"""Task recipes, physical batching and speculative error isolation; no model calls."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from lev._data import thaw
from lev._request_adapters import CONFIGURABLE_DECISION_ADAPTER, normalize_decision_options
from lev.backends import DecisionAPIBackend, InferenceContext, InferenceError, DecisionBatchPlan
from lev.backends.base import normalize_call, encode_payload
from lev.backends.config import backends_from_env
from lev.client import LevClient
from lev.compiler import compile_program
from lev.plan_lowering import lower_to_plan
from lev.preview import preview_request, main
from lev.runtime import LevRuntime, RunInput
from lev.server.releases import Releases, manifest
from lev.server.worker import ObservedBackend, backends as worker_backends
from lev._dispatch_pool import RequestSlots
from lev.test_decision_wire import answers_for
from lev.test_request_adapters import configuration, spec
from lev.test_runtime import FakeBackend, mixed


def recipe(criteria='preserve', examples='preserve', mode='full', **state):
    return {'criteria': criteria, 'examples': examples, 'state': {'mode': mode, **state}}


def plain(mode='full', **state):
    return recipe('text', 'omit', mode, **state)


def backend(options=None, **kwargs):
    return DecisionAPIBackend('http://offline.invalid/decision', 'secret', 'unchanged-model',
                              request_adapter=CONFIGURABLE_DECISION_ADAPTER, request_adapter_options=options, **kwargs)


def call(context, identity='a', name='tone', operation='choice', examples=True):
    kwargs = {'instructions': '独立任务', 'field_name': name}
    if operation == 'choice': kwargs['criteria'] = {'甲': '类别甲', '乙': '类别乙'}
    if operation == 'score': kwargs['criteria'] = ['低', '高']
    if examples:
        kwargs['examples'] = [{'state': {'text': '示例一', 'meta': ['仅示例']},
                               'answer': {'choice': '甲', 'score': 0, 'noul': False}[operation]},
                              {'state': {'text': '示例二'},
                               'answer': {'choice': '乙', 'score': 1, 'noul': True}[operation]}]
    return normalize_call(identity, operation, (), kwargs, context)


class RepresentationTests(unittest.TestCase):
    def test_preserve_unwrap_and_projection_without_mutating_inputs(self):
        context = InferenceContext({'text': '当前', 'meta': ['当前元数据']})
        c = call(context)
        before = thaw(c.arguments)
        for criteria, examples in [('preserve', 'preserve'), ('preserve', 'omit'), ('text', 'omit')]:
            for state in [{'mode': 'full'}, {'mode': 'fields_object', 'fields': ['text']}, {'mode': 'field_string', 'field': 'text'}]:
                r = {'criteria': criteria, 'examples': examples, 'state': state}
                b = backend({'tasks': {'tone': r}})
                value = b.preview_batch(context, [c])['payload']
                self.assertEqual(list(value['questions']['tone']['criteria']), ['甲', '乙'])
                expected = {'full': thaw(context.state), 'fields_object': {'text': '当前'}, 'field_string': '当前'}[state['mode']]
                self.assertEqual(value['state'], expected)
                criterion = value['questions']['tone']['criteria']['甲']
                if criteria == 'text': self.assertEqual(criterion, '类别甲')
                else:
                    self.assertEqual(criterion['what'], '类别甲')
                    if examples == 'preserve':
                        projected = {'full': {'text': '示例一', 'meta': ['仅示例']}, 'fields_object': {'text': '示例一'}, 'field_string': '示例一'}[state['mode']]
                        self.assertEqual(criterion['examples'], [projected])
                    else: self.assertNotIn('examples', criterion)
                self.assertEqual(thaw(c.arguments), before)
                self.assertEqual(thaw(context.state), {'text': '当前', 'meta': ['当前元数据']})

    def test_task_matching_is_exact_with_default_for_unnamed_tasks(self):
        ctx = InferenceContext({'text': '输入'})
        b = backend({'tasks': {'tone': plain()}})
        calls = [call(ctx, str(i), name) for i, name in enumerate(['tone', ' tone ', 'tone_extra', None])]
        payload = b.preview_batch(ctx, calls)['payload']
        self.assertEqual(payload['questions']['tone']['criteria']['甲'], '类别甲')
        for key in [' tone ', 'tone_extra', '3']:
            self.assertIn('examples', payload['questions'][key]['criteria']['甲'])
        self.assertEqual(len(b.preview_batches(ctx, calls)['requests']), 1)

    def test_empty_and_missing_projection_and_example_errors_are_local(self):
        for state in ['正文', [], {}, {'text': []}]:
            ctx = InferenceContext(state)
            b = backend({'default': plain('field_string', field='text')})
            with self.subTest(state=state), self.assertRaises(InferenceError) as caught:
                b.preview_batch(ctx, [call(ctx)])
            self.assertEqual(caught.exception.code, 'invalid_state_projection')
            self.assertEqual(caught.exception.call_id, 'a')
            self.assertEqual(caught.exception.metadata['dispatch_status'], 'not_sent')
            self.assertIsNone(b._client)
        ctx = InferenceContext({'text': ''})
        b = backend({'default': plain('field_string', field='text')})
        self.assertEqual(b.preview_batch(ctx, [call(ctx)])['payload']['state'], '')
        kept = backend({'default': recipe(mode='field_string', field='text')})
        c = normalize_call('x', 'choice', ('任务', {'甲': '甲'}),
                           {'examples': [{'state': {'other': 'missing'}, 'answer': '甲'}]}, ctx)
        with self.assertRaises(InferenceError): kept.preview_batch(ctx, [c])
        self.assertEqual(b.preview_batch(ctx, [c])['payload']['state'], '')

    def test_actual_shared_state_controls_compatibility_and_prefix_order(self):
        opts = {'tasks': {'tone': plain('field_string', field='text')}}
        b = backend(opts)
        ctx = InferenceContext({'text': '输入', 'brand': ['甲']})
        calls = [call(ctx, 'a', 'tone'), call(ctx, 'b', 'brand'), call(ctx, 'c', 'tone')]
        plan = b.preview_batches(ctx, calls)
        self.assertEqual([list(v['evidence']['wire_call_ids'].values()) for v in plan['requests']], [['a'], ['b'], ['c']])
        self.assertEqual([v['payload']['state'] for v in plan['requests']], ['输入', thaw(ctx.state), '输入'])
        with self.assertRaises(InferenceError) as caught: b.preview_batch(ctx, calls[:2])
        self.assertEqual(caught.exception.code, 'incompatible_decision_batch')
        # Different view definitions may share the same actual object bytes.
        ctx = InferenceContext({'text': '输入'})
        b = backend({'tasks': {'tone': plain('fields_object', fields=['text'])}})
        self.assertEqual(len(b.plan_batch(ctx, [call(ctx, 'a', 'tone'), call(ctx, 'b', 'brand')], max_calls=32).calls), 2)
        # Object key order is model-visible and must not be canonicalized away.
        ctx = InferenceContext({'x': 1, 'text': '输入'})
        b = backend({'tasks': {'tone': plain('fields_object', fields=['text', 'x'])}})
        self.assertEqual(len(b.plan_batch(ctx, [call(ctx, 'a', 'tone'), call(ctx, 'b', 'brand')], max_calls=32).calls), 1)

    def test_limits_validation_and_duplicate_names_are_not_hidden(self):
        ctx = InferenceContext({'text': '输入'})
        calls = [call(ctx, str(i), 'n' + str(i)) for i in range(5)]
        b = backend()
        self.assertEqual([len(p['payload']['questions']) for p in b.preview_batches(ctx, calls, max_calls=2)['requests']], [2, 2, 1])
        for limit in [0, -1, True, 1.5, None]:
            with self.assertRaises(ValueError): b.plan_batch(ctx, calls, max_calls=limit)
        duplicate = [call(ctx, 'a'), call(ctx, 'b')]
        self.assertIsNone(b.plan_batch(ctx, duplicate, max_calls=32).prepared)
        with self.assertRaisesRegex(ValueError, 'duplicate model-visible'): b.preview_batch(ctx, duplicate)

    def test_invalid_options_are_rejected_before_transport_and_detached(self):
        invalid = [False, [], {'extra': {}}, {'default': {}}, {'tasks': []}, {'tasks': {'': plain()}},
                   {'default': recipe('text', 'preserve')}, {'default': recipe(mode='unknown')},
                   {'default': recipe(mode='field_string', field=' ')}, {'default': recipe(mode='full', field='x')},
                   {'default': recipe(mode='fields_object', fields=[])}, {'default': recipe(mode='fields_object', fields=['x', 'x'])}]
        for options in invalid:
            with self.subTest(options=options), patch('lev.backends.base.HTTPBackend.__init__', side_effect=AssertionError('transport')):
                with self.assertRaises(ValueError): backend(options)
        source = {'tasks': {'tone': plain()}}
        b = backend(source)
        source['tasks']['tone']['criteria'] = 'preserve'
        self.assertEqual(b.request_adapter_options['tasks']['tone']['criteria'], 'text')
        with self.assertRaises(TypeError): b.request_adapter_options['tasks']['tone']['criteria'] = 'preserve'
        with self.assertRaises(AttributeError): b.request_adapter_options = {}
        with self.assertRaises(ValueError): DecisionAPIBackend('http://offline.invalid', 'secret', 'm', request_adapter_options=source)


class DecisionHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_score_legends_and_noul_without_invented_descriptions(self):
        ctx = InferenceContext({'text': '输入'})
        sent = []
        def handle(req):
            body = json.loads(req.content); sent.append(body)
            return httpx.Response(200, json={'answers': answers_for(body)})
        b = backend({'default': plain()}, transport=httpx.MockTransport(handle))
        try:
            calls = [call(ctx, 's', 'score', 'score'), call(ctx, 'n', 'boolean', 'noul')]
            result = await b.infer_batch(ctx, calls)
            self.assertEqual(result.responses['s'].legend, {0: '低', 1: '高'})
            self.assertEqual(result.responses['n'].noul, .8)
            self.assertEqual(sent[0]['questions']['score']['criteria'], ['低', '高'])
            self.assertNotIn('criteria', sent[0]['questions']['boolean'])
        finally: await b.aclose()

    async def test_prepared_requests_bound_to_options_calls_context_and_backend(self):
        ctx = InferenceContext({'text': '输入'})
        calls = [call(ctx)]
        received = []
        def handle(req):
            received.append(req.content)
            return httpx.Response(200, json={'answers': answers_for(json.loads(req.content))})
        a = backend(transport=httpx.MockTransport(handle))
        b = backend({'default': plain()}, transport=httpx.MockTransport(handle))
        try:
            prepared = a.plan_batch(ctx, calls, max_calls=32).prepared
            self.assertIsNotNone(prepared)
            await b.infer_batch(ctx, calls, prepared=prepared)
            self.assertEqual(received[-1], encode_payload(b.preview_batch(ctx, calls)['payload']))
            self.assertNotEqual(received[-1], prepared.body)
            # Valid preparation is sent without reconstructing the body.
            plan = b.plan_batch(ctx, calls, max_calls=32)
            with patch.object(b, '_batch_payload', side_effect=AssertionError('rebuilt')):
                await b.infer_batch(ctx, calls, prepared=plan.prepared)
            self.assertEqual(received[-1], plan.prepared.body)
            changed = [call(ctx, 'other', 'other')]
            await b.infer_batch(ctx, changed, prepared=plan.prepared)
            self.assertIn('other', json.loads(received[-1])['questions'])
            other_ctx = InferenceContext({'text': '另一个输入'})
            await b.infer_batch(other_ctx, calls, prepared=plan.prepared)
            self.assertEqual(json.loads(received[-1])['state'], {'text': '另一个输入'})
        finally:
            await a.aclose(); await b.aclose()

    async def test_incompatible_direct_batch_sends_nothing_and_partial_answers_keep_ids(self):
        ctx = InferenceContext({'text': '输入', 'meta': []}); sent = []
        calls = [call(ctx, 'a', 'tone'), call(ctx, 'b', 'other')]
        def handle(req):
            sent.append(req.content); body = json.loads(req.content)
            answers = answers_for(body); answers.pop(next(reversed(answers)))
            return httpx.Response(200, json={'answers': answers})
        b = backend({'tasks': {'tone': plain('field_string', field='text')}}, transport=httpx.MockTransport(handle))
        try:
            with self.assertRaises(InferenceError): await b.infer_batch(ctx, calls)
            self.assertFalse(sent)
        finally: await b.aclose()
        b = backend(transport=httpx.MockTransport(handle))
        try:
            result = await b.infer_batch(ctx, calls, allow_partial=True)
            self.assertEqual(set(result.responses) | set(result.errors), {'a', 'b'})
            self.assertEqual(len(sent), 1)
            self.assertTrue(all(e.call_id == k for k, e in result.errors.items()))
        finally: await b.aclose()


def ordered_tasks(lev):
    a = lev.choice('body first', {'yes': '成立', 'no': '不成立'}, field_name='body').choice
    b = lev.noul('brand', field_name='brand').noul
    c = lev.choice('body last', {'yes': '成立', 'no': '不成立'}, field_name='body').choice
    return [a, b, c]


def projection_branch(lev):
    a = lev.choice('gate', {'yes': '成立', 'no': '不成立'}, field_name='gate').choice
    if a == 'no':
        return lev.noul('bad projection', field_name='bad').noul
    return lev.noul('good projection', field_name='good').noul


def selected_invalid(lev):
    return lev.noul('selected invalid', field_name='bad').noul


class DecisionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def run_program(self, function, b, *, mode='speculative', **options):
        async with LevRuntime(b, FakeBackend(), trace=True, **options) as runtime:
            results = [r async for r in await runtime.submit(lower_to_plan(compile_program(function)),
                                                           [RunInput(0, {'text': '输入', 'brand': ['甲']})], mode=mode)]
        return results[0], runtime.trace

    async def test_ordered_split_all_modes_and_one_slot_per_physical_request(self):
        for mode in ['eager', 'strict', 'speculative']:
            sent, received, active, peak = [], [], 0, 0
            async def handle(req):
                nonlocal active, peak
                body = json.loads(req.content); sent.append(body); received.append(req.content); active += 1; peak = max(peak, active)
                await asyncio.sleep(.005); active -= 1
                return httpx.Response(200, json={'answers': answers_for(body)})
            slots = RequestSlots(decision=1, generation=1)
            b = backend({'tasks': {'body': plain('field_string', field='text')}}, transport=httpx.MockTransport(handle))
            ctx = InferenceContext({'text': '输入', 'brand': ['甲']})
            calls = [normalize_call('a', 'choice', ('body first', {'yes': '成立', 'no': '不成立'}), {'field_name': 'body'}, ctx),
                     normalize_call('b', 'noul', ('brand',), {'field_name': 'brand'}, ctx),
                     normalize_call('c', 'choice', ('body last', {'yes': '成立', 'no': '不成立'}), {'field_name': 'body'}, ctx)]
            preview = b.preview_batches(ctx, calls)
            observed = ObservedBackend(b, [], lambda: True)
            result, trace = await self.run_program(ordered_tasks, observed, mode=mode, request_slots=slots)
            self.assertEqual(result.status, 'success', result.error)
            self.assertEqual(result.value, ['yes', .8, 'yes'])
            self.assertEqual([list(v['questions']) for v in sent], [['body'], ['brand'], ['body']])
            self.assertEqual(received, [encode_payload(v['payload']) for v in preview['requests']])
            self.assertEqual(len([e for e in trace if e['event'] == 'http_submitted']), 3)
            self.assertEqual(peak, 1); self.assertEqual(slots.active['decision'], 0)

    async def test_projection_errors_are_isolated_and_discarded_when_not_adopted(self):
        for function, expected in [(projection_branch, 'success'), (selected_invalid, 'error')]:
            sent = []
            def handle(req):
                body = json.loads(req.content); sent.append(body)
                return httpx.Response(200, json={'answers': answers_for(body)})
            b = backend({'tasks': {'bad': plain('field_string', field='missing')}}, transport=httpx.MockTransport(handle))
            result, trace = await self.run_program(function, b)
            self.assertEqual(result.status, expected, result.error)
            self.assertFalse(any('bad' in v['questions'] for v in sent))
            if function is projection_branch:
                self.assertEqual(result.value, .8)
                self.assertTrue(any(e['event'] == 'speculation_error_deferred' for e in trace))
            else: self.assertEqual(result.error.code, 'invalid_state_projection')

    async def test_custom_backend_without_planning_keeps_existing_batching(self):
        b = FakeBackend()
        result, trace = await self.run_program(mixed, b, mode='eager', max_batch_questions=2)
        self.assertEqual(result.status, 'success', result.error)
        self.assertEqual([len(v[1]) for v in b.batches], [2, 1])

    async def test_split_partial_response_error_is_only_observed_on_selected_path(self):
        for selected in ['yes', 'no']:
            sent = []
            def handle(req):
                body = json.loads(req.content); sent.append(body); answers = answers_for(body)
                if 'gate' in answers:
                    answers['gate'].update(choice=selected, probabilities={'yes': float(selected == 'yes'), 'no': float(selected == 'no')})
                if 'bad' in answers: answers['bad']['noul'] = 'invalid'
                return httpx.Response(200, json={'answers': answers})
            b = backend({'tasks': {'bad': plain('field_string', field='text'), 'good': plain('field_string', field='text')}},
                        transport=httpx.MockTransport(handle))
            result, trace = await self.run_program(projection_branch, ObservedBackend(b, [], lambda: True))
            self.assertEqual(result.status, 'success' if selected == 'yes' else 'error', result.error)
            self.assertEqual(sum('bad' in v['questions'] for v in sent), 1)
            if selected == 'yes': self.assertEqual(result.value, .8)
            else: self.assertEqual(result.error.code, 'invalid_response')

    async def test_runtime_question_limit_and_invalid_planner_release_slots(self):
        for limit in [1, 2]:
            sent = []
            def handle(req):
                body = json.loads(req.content); sent.append(body)
                return httpx.Response(200, json={'answers': answers_for(body)})
            b = backend(transport=httpx.MockTransport(handle))
            result, trace = await self.run_program(mixed, b, mode='eager', max_batch_questions=limit)
            self.assertEqual(result.status, 'success', result.error)
            self.assertEqual([len(v['questions']) for v in sent], [1, 1, 1] if limit == 1 else [2, 1])
        slots = RequestSlots(decision=1, generation=1)
        b = backend(transport=httpx.MockTransport(lambda req: self.fail('HTTP')))
        def invalid(context, calls, *, max_calls):
            return DecisionBatchPlan(tuple(reversed(calls)), None)
        with patch.object(b, 'plan_batch', side_effect=invalid), self.assertRaises(InferenceError) as caught:
            await self.run_program(mixed, b, mode='eager', request_slots=slots)
        self.assertEqual(caught.exception.code, 'runtime_failed')
        self.assertEqual(slots.active['decision'], 0)

    async def test_cancellation_drains_inflight_slot_without_sending_other_prefixes(self):
        entered, release = asyncio.Event(), asyncio.Event(); sent = []
        async def handle(req):
            body = json.loads(req.content); sent.append(body); entered.set(); await release.wait()
            return httpx.Response(200, json={'answers': answers_for(body)})
        slots = RequestSlots(decision=1, generation=1)
        b = backend({'tasks': {'body': plain('field_string', field='text')}}, transport=httpx.MockTransport(handle))
        runtime = LevRuntime(b, FakeBackend(), request_slots=slots)
        await runtime.__aenter__()
        try:
            stream = await runtime.submit(lower_to_plan(compile_program(ordered_tasks)), [RunInput(0, {'text': '输入'})], mode='speculative')
            await asyncio.wait_for(entered.wait(), 2)
            await stream.aclose()
            self.assertEqual(len(sent), 1)
            self.assertEqual(slots.active['decision'], 1)
            release.set()
        finally:
            release.set(); await runtime.aclose()
        self.assertEqual(len(sent), 1)
        self.assertEqual(slots.active['decision'], 0)


class OptionsAndPreviewTests(unittest.TestCase):
    def test_release_normalization_binding_and_worker_ignores_environment(self):
        cfg = configuration()
        cfg['decision'].update(request_adapter=CONFIGURABLE_DECISION_ADAPTER, request_adapter_options={'tasks': {'tone': plain()}})
        m = manifest(cfg)
        normalized = normalize_decision_options(CONFIGURABLE_DECISION_ADAPTER, cfg['decision']['request_adapter_options'])
        self.assertEqual(m['decision']['request_adapter_options'], normalized)
        explicit = deepcopy(cfg); explicit['decision']['request_adapter_options'] = normalized
        self.assertEqual(Releases({'x': cfg}, 'x').default, Releases({'x': explicit}, 'x').default)
        changed = deepcopy(cfg); changed['decision']['request_adapter_options']['tasks']['tone'] = plain('field_string', field='text')
        self.assertNotEqual(Releases({'x': cfg}, 'x').default, Releases({'x': changed}, 'x').default)
        with patch.dict(os.environ, {'LEV_DECISION_AUTH_TOKEN': 'secret', 'LEV_GENERATE_AUTH_TOKEN': 'secret',
                                     'LEV_DECISION_REQUEST_ADAPTER_OPTIONS': 'invalid'}, clear=True):
            d, _ = worker_backends(m)
        self.assertEqual(thaw(d.request_adapter_options), normalized)
        bad = deepcopy(cfg); bad['decision']['request_adapter_options'] = None
        with self.assertRaises(ValueError): manifest(bad)

    def test_existing_job_retains_recipe_when_default_release_changes(self):
        from lev.server.store import Store
        from lev.test_jobs_server import payload
        a = configuration()
        a['decision'].update(request_adapter=CONFIGURABLE_DECISION_ADAPTER, request_adapter_options={'tasks': {'tone': plain()}})
        b = deepcopy(a)
        b['decision']['request_adapter_options']['tasks']['tone'] = plain('field_string', field='text')
        old, new = Releases({'x': a}, 'x'), Releases({'x': b}, 'x')
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'jobs.sqlite')
            first = store.submit('alice', 'same', payload(), old)
            repeated = store.submit('alice', 'same', payload(), new)
            fresh = store.submit('alice', 'fresh', payload(), new)
            self.assertEqual(first['release_id'], repeated['release_id'])
            self.assertEqual(fresh['release_id'], new.default)
            self.assertNotEqual(first['release_id'], fresh['release_id'])

    def test_environment_json_precedence_and_client_forwarding(self):
        env = {f'LEV_{p}_{k}': v for p in ['DECISION', 'GENERATE'] for k, v in
               [('BASE_URL', 'http://offline.invalid'), ('AUTH_TOKEN', 'secret'), ('MODEL', 'unchanged')]}
        env['LEV_DECISION_REQUEST_ADAPTER'] = CONFIGURABLE_DECISION_ADAPTER
        opts = {'tasks': {'tone': plain()}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'env'
            path.write_text("LEV_DECISION_REQUEST_ADAPTER_OPTIONS='" + json.dumps(opts) + "'\n")
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(thaw(backends_from_env(env_file=path)[0].request_adapter_options)['tasks'], opts['tasks'])
            env['LEV_DECISION_REQUEST_ADAPTER_OPTIONS'] = '{"default":{},"default":{}}'
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError): backends_from_env(env_file=path)
                with LevClient.from_env(env_file=path, decision_request_adapter_options=opts) as client:
                    self.assertEqual(thaw(client.runtime.decision_backend.request_adapter_options)['tasks'], opts['tasks'])
            for invalid in ['null', '[]', '{"x":NaN}', '']:
                env['LEV_DECISION_REQUEST_ADAPTER_OPTIONS'] = invalid
                with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError): backends_from_env()


if __name__ == '__main__':
    unittest.main()
