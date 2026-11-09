"""Frozen pre-refactor wire compatibility, explicit binding and offline preview."""

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from annoscript._data import thaw
from annoscript._request_adapters import DEFAULT_DECISION_ADAPTER, DEFAULT_GENERATION_ADAPTER
from annoscript.backends import DecisionAPIBackend, LLMAPIBackend, InferenceContext, InferenceError
from annoscript.backends._batch_wire import batch_calls
from annoscript.backends.base import normalize_call, encode_payload
from annoscript.backends.config import backends_from_env
from annoscript.client import AnnoClient
from annoscript.preview import preview_request, main
from annoscript.server.releases import Releases, manifest
from annoscript.server.worker import backends as worker_backends

BASELINE = Path(__file__).with_name('fixtures') / 'request-adapters-v1.json'


def configuration():
    return {'decision': {'base_url': 'http://offline.invalid/decision', 'model': 'unchanged-decider', 'revision': 'fixed'},
            'generation': {'base_url': 'http://offline.invalid/v1', 'model': 'unchanged-generator', 'revision': 'fixed'}}


def spec(stage='generation', **extra):
    value = {'format': 'lev-request-spec', 'version': 1, 'stage': stage, 'model': 'unchanged-model',
             'state': {'text': '北京广州'}, 'calls': [{'id': 'x', 'operation': 'generate',
                 'arguments': {'instructions': '标题', 'max_length': 24}}]}
    if stage == 'decision':
        value['calls'][0].update(operation='choice', arguments={'instructions': '类别', 'criteria': {'甲': '甲类', '乙': '乙类'}})
    value.update(extra)
    return value


def answer(call):
    if call.operation == 'generate':
        return '标题'
    if call.operation == 'generate_list':
        return ['维度']
    return ['北京']


class FrozenWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_111_pre_refactor_bodies_default_and_explicit_on_real_send_path(self):
        fixture = json.loads(BASELINE.read_text())
        self.assertEqual(fixture['source_head'], 'synthetic-request-adapter-v1')
        self.assertEqual(len(fixture['cases']), 111)
        for index, case in enumerate(fixture['cases']):
            for explicit in (False, True):
                with self.subTest(case=index, explicit=explicit):
                    context = InferenceContext(case['state'], case.get('system_prompt'))
                    calls = [normalize_call(x['id'], x['operation'], (), x['kwargs'], context) for x in case['calls']]
                    original = [thaw(c.arguments) for c in calls]
                    received = []

                    def handle(request):
                        received.append(request.content)
                        body = json.loads(request.content)
                        if case['stage'] == 'decision':
                            answers = {}
                            for key, q in body['questions'].items():
                                value = {'type': q['type']}
                                if q['type'] == 'noul':
                                    value['noul'] = .8
                                elif q['type'] == 'choice':
                                    labels = list(q['criteria'])
                                    value.update(choice=labels[0], confidence=1., probabilities={k: float(k == labels[0]) for k in labels})
                                else:
                                    value.update(score=1., confidence=1., probabilities={'0': 0., '1': 1.},
                                                 legend={str(i): v for i, v in enumerate(q['criteria'])})
                                answers[key] = value
                            data = {'answers': answers}
                        elif case['mode'] == 'batch':
                            text = json.dumps({'answers': {k: answer(c) for k, c in batch_calls(calls).items()}}, ensure_ascii=False)
                            data = {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}
                        else:
                            value = answer(calls[0])
                            text = (json.dumps(value, ensure_ascii=False) if calls[0].arguments.get('nullable') else value) if calls[0].operation == 'generate' else json.dumps({'items': value}, ensure_ascii=False)
                            data = {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}
                        return httpx.Response(200, json=data)

                    options = deepcopy(case['options'])
                    if explicit:
                        options['request_adapter'] = DEFAULT_DECISION_ADAPTER if case['stage'] == 'decision' else DEFAULT_GENERATION_ADAPTER
                    if options.get('request_policy'):
                        options['_token_counter'] = lambda payload: 10
                    cls = DecisionAPIBackend if case['stage'] == 'decision' else LLMAPIBackend
                    backend = cls('http://offline.invalid/v1', 'transport-only-secret', fixture['model'],
                                  transport=httpx.MockTransport(handle), **options)
                    try:
                        if case['stage'] == 'decision':
                            preview = backend.preview_batch(context, calls)
                            await backend.infer_batch(context, calls)
                        elif case['mode'] == 'batch':
                            preview = backend.preview_batch(context, calls, max_tokens=case['max_tokens'])
                            await backend.infer_batch(context, calls, max_tokens=case['max_tokens'])
                        else:
                            # Frozen envelopes used an explicit physical budget. max_num_tokens
                            # is absent from the prompt, so pin that budget for the public path.
                            item = case['calls'][0]
                            pinned = normalize_call(item['id'], item['operation'], (),
                                                    {**item['kwargs'], 'max_num_tokens': case['max_tokens']}, context)
                            preview = backend.preview(context, pinned, repair_feedback=case.get('repair_feedback'))
                            await backend.infer(context, pinned, repair_feedback=case.get('repair_feedback'))
                        self.assertEqual(len(received), 1)
                        self.assertEqual(len(received[0]), case['expected_bytes'])
                        self.assertEqual(hashlib.sha256(received[0]).hexdigest(), case['expected_sha256'])
                        self.assertEqual(encode_payload(preview['payload']), received[0])
                        self.assertEqual(preview['body_sha256'], case['expected_sha256'])
                        self.assertNotIn('transport-only-secret', json.dumps(preview))
                        self.assertEqual([thaw(c.arguments) for c in calls], original)
                    finally:
                        await backend.aclose()

    async def test_preview_errors_match_preflight_without_dispatch(self):
        from annoscript.test_generation_policy import policy
        context = InferenceContext('')
        call = normalize_call('e', 'extract', ('来源',), {'min_items': 1, 'max_items': 2, 'max_length_per_item': 4}, context)
        backend = LLMAPIBackend('http://offline.invalid', 'secret', 'model', profile='qwen_vllm',
                                transport=httpx.MockTransport(lambda req: self.fail('unexpected HTTP')))
        try:
            for invoke in (lambda: backend.preview(context, call), lambda: backend.preview_batch(context, [call], max_tokens=32)):
                with self.assertRaises(InferenceError) as caught:
                    invoke()
                self.assertEqual(caught.exception.code, 'unsatisfiable_constraints')
            with self.assertRaises(InferenceError) as caught:
                await backend.infer(context, call)
            self.assertEqual(caught.exception.code, 'unsatisfiable_constraints')
            self.assertIsNone(backend._client)
        finally:
            await backend.aclose()
        context = InferenceContext('北京')
        call = normalize_call('e', 'generate', ('标题',), {'max_length': 4}, context)
        backend = LLMAPIBackend('http://offline.invalid', 'secret', 'model', profile='qwen_vllm',
                                request_policy=policy(), _token_counter=lambda p: 8190,
                                transport=httpx.MockTransport(lambda req: self.fail('unexpected HTTP')))
        try:
            with self.assertRaises(InferenceError) as caught:
                backend.preview(context, call)
            self.assertEqual(caught.exception.code, 'context_length_exceeded')
            with self.assertRaises(InferenceError):
                await backend.infer(context, call)
            self.assertIsNone(backend._client)
        finally:
            await backend.aclose()


class AdapterBindingTests(unittest.TestCase):
    def test_release_materializes_defaults_and_rejects_unbound_or_cross_stage_recipes(self):
        cfg = configuration()
        implicit = manifest(cfg)
        explicit = deepcopy(cfg)
        explicit['decision']['request_adapter'] = DEFAULT_DECISION_ADAPTER
        explicit['generation']['request_adapter'] = DEFAULT_GENERATION_ADAPTER
        self.assertEqual(implicit, manifest(explicit))
        self.assertEqual(implicit['decision']['request_adapter'], DEFAULT_DECISION_ADAPTER)
        self.assertEqual(implicit['generation']['request_adapter'], DEFAULT_GENERATION_ADAPTER)
        self.assertEqual(cfg, configuration())
        for stage in ('decision', 'generation'):
            for invalid in ('unknown', '', None, False, [], DEFAULT_GENERATION_ADAPTER if stage == 'decision' else DEFAULT_DECISION_ADAPTER):
                altered = deepcopy(cfg)
                altered[stage]['request_adapter'] = invalid
                with self.subTest(stage=stage, invalid=invalid), self.assertRaises(ValueError):
                    manifest(altered)

    def test_adapter_implementation_changes_release_identity_and_existing_job_stays_bound(self):
        from annoscript.server.store import Store
        from annoscript.test_jobs_server import payload
        cfg = {'test': configuration()}
        old = Releases(cfg, 'test')
        read_bytes = Path.read_bytes

        def changed_adapter(path):
            data = read_bytes(path)
            return data + b'\n# revision\n' if path.parent.name == 'adapters' and path.name == 'decision.py' else data

        with patch.object(Path, 'read_bytes', changed_adapter):
            new = Releases(cfg, 'test')
        self.assertNotEqual(old.default, new.default)
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'jobs.sqlite')
            first = store.submit('alice', 'same', payload(), old)
            repeated = store.submit('alice', 'same', payload(), new)
            fresh = store.submit('alice', 'fresh', payload(), new)
            self.assertEqual(first['release_id'], repeated['release_id'])
            self.assertEqual(fresh['release_id'], new.default)

    def test_worker_uses_bound_adapter_despite_poisoned_environment(self):
        cfg = manifest(configuration())
        environment = {'ANNO_DECISION_AUTH_TOKEN': 'secret', 'ANNO_GENERATE_AUTH_TOKEN': 'secret',
                       'ANNO_DECISION_REQUEST_ADAPTER': 'unknown', 'ANNO_GENERATE_REQUEST_ADAPTER': 'unknown'}
        with patch.dict(os.environ, environment, clear=True):
            decision, generation = worker_backends(cfg)
        self.assertEqual(decision.request_adapter, DEFAULT_DECISION_ADAPTER)
        self.assertEqual(generation.request_adapter, DEFAULT_GENERATION_ADAPTER)
        # Existing manifests without recipe fields retain the original recipe.
        cfg['decision'].pop('request_adapter')
        cfg['generation'].pop('request_adapter')
        with patch.dict(os.environ, environment, clear=True):
            decision, generation = worker_backends(cfg)
        self.assertEqual(decision.request_adapter, DEFAULT_DECISION_ADAPTER)
        self.assertEqual(generation.request_adapter, DEFAULT_GENERATION_ADAPTER)

    def test_environment_precedence_and_client_forwarding(self):
        env = {f'ANNO_{p}_{k}': v for p in ('DECISION', 'GENERATE') for k, v in
               [('BASE_URL', 'http://offline.invalid'), ('AUTH_TOKEN', 'secret'), ('MODEL', 'same-model')]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'env'
            path.write_text('ANNO_DECISION_REQUEST_ADAPTER=unknown\nANNO_GENERATE_REQUEST_ADAPTER=unknown\n')
            with patch.dict(os.environ, env, clear=True):
                d, g = backends_from_env()
                self.assertEqual((d.request_adapter, g.request_adapter), (DEFAULT_DECISION_ADAPTER, DEFAULT_GENERATION_ADAPTER))
                with self.assertRaises(ValueError):
                    backends_from_env(env_file=path)
                with AnnoClient.from_env(env_file=path, decision_request_adapter=DEFAULT_DECISION_ADAPTER,
                                        generation_request_adapter=DEFAULT_GENERATION_ADAPTER) as client:
                    self.assertEqual(client.runtime.decision_backend.request_adapter, DEFAULT_DECISION_ADAPTER)
                    self.assertEqual(client.runtime.llm_backend.request_adapter, DEFAULT_GENERATION_ADAPTER)
            env.update(ANNO_DECISION_REQUEST_ADAPTER=DEFAULT_DECISION_ADAPTER, ANNO_GENERATE_REQUEST_ADAPTER=DEFAULT_GENERATION_ADAPTER)
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(backends_from_env(env_file=path)[1].request_adapter, DEFAULT_GENERATION_ADAPTER)
                for value in ('', False, 0, []):
                    with self.assertRaises(ValueError):
                        backends_from_env(generation_request_adapter=value)
        for cls, wrong in [(DecisionAPIBackend, DEFAULT_GENERATION_ADAPTER), (LLMAPIBackend, DEFAULT_DECISION_ADAPTER)]:
            with patch('annoscript.backends.base.HTTPBackend.__init__', side_effect=AssertionError('transport initialized')):
                with self.assertRaises(ValueError):
                    cls('http://offline.invalid', 'secret', 'model', request_adapter=wrong)


class OfflinePreviewTests(unittest.TestCase):
    def test_preview_is_deterministic_detached_and_does_not_load_credentials_or_dispatch(self):
        for stage in ('decision', 'generation'):
            request = spec(stage)
            original = deepcopy(request)
            with patch('annoscript.backends.base.HTTPBackend.post', side_effect=AssertionError('HTTP')),\
                 patch('annoscript.backends.config.backends_from_env', side_effect=AssertionError('credentials')),\
                 patch('socket.socket.connect', side_effect=AssertionError('socket')),\
                 patch.dict(os.environ, {'ANNO_DECISION_REQUEST_ADAPTER': 'unknown', 'ANNO_GENERATE_REQUEST_ADAPTER': 'unknown'}, clear=True):
                left, right = preview_request(request), preview_request(request)
            self.assertEqual(left, right)
            self.assertEqual(request, original)
            self.assertEqual(hashlib.sha256(encode_payload(left['payload'])).hexdigest(), left['body_sha256'])
            left['payload']['model'] = 'modified-preview'
            self.assertEqual(preview_request(request), right)

    def test_invalid_specs_and_collisions_are_rejected(self):
        cases = [spec(version=True), spec(stage=[]), spec(options={'auth_token': 'secret'}),
                 spec(request_adapter=DEFAULT_DECISION_ADAPTER), spec(mode='single', calls=[]),
                 spec(mode='batch'), spec(max_tokens=0), spec(repair_feedback=None), spec(unrecognized=True)]
        for change in [lambda v: v['calls'].append(deepcopy(v['calls'][0])),
                       lambda v: v['calls'][0].update(id=''),
                       lambda v: v['calls'][0].update(arguments={'instructions': 'x', 'source': 'old-api'})]:
            v = spec(); change(v); cases.append(v)
        collision = spec('decision')
        collision['calls'][0]['arguments']['field_name'] = 'same'
        other = deepcopy(collision['calls'][0]); other['id'] = 'y'
        collision['calls'].append(other); cases.append(collision)
        cross_stage = spec(mode='batch', max_tokens=64)
        cross_stage['calls'] = spec('decision')['calls']
        cases.append(cross_stage)
        for case in cases:
            with self.subTest(case=case), self.assertRaises((ValueError, TypeError)):
                preview_request(case)

    def test_cli_files_stdin_strict_json_and_input_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'spec.json', Path(folder) / 'preview.json'
            raw = json.dumps(spec(), ensure_ascii=False)
            source.write_text(raw)
            self.assertEqual(main([str(source), '-o', str(target)]), 0)
            self.assertEqual(json.loads(target.read_text()), preview_request(spec()))
            with patch('sys.stderr', new_callable=io.StringIO):
                self.assertEqual(main([str(source), '-o', str(source)]), 2)
            self.assertEqual(source.read_text(), raw)
            with patch('sys.stdin', io.StringIO(raw)), patch('sys.stdout', new_callable=io.StringIO) as out:
                self.assertEqual(main(['-']), 0)
            self.assertEqual(json.loads(out.getvalue()), preview_request(spec()))
            for raw in ['{"version":1,"version":1}', '{"x":NaN}']:
                source.write_text(raw)
                with patch('sys.stderr', new_callable=io.StringIO):
                    self.assertEqual(main([str(source)]), 2)


if __name__ == '__main__':
    unittest.main()
