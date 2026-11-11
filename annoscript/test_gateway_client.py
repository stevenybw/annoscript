"""Base-dependency gateway client regression, including Python 3.10."""
import asyncio
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import httpx
from annoscript._gateway import GatewaySession
from annoscript._test_synthetic import CannedBackend
from annoscript.backends.base import BackendFatalError, InferenceContext, normalize_call
from annoscript.client import AnnoClient
from annoscript_examples.quick_start import SAMPLE, main
from annoscript_examples.request import annotate

ANSWER = {'category': 'note', 'highlight': True, 'clarity': 1., 'title': 'Library opening',
          'tags': ['library'], 'spans': ['library']}


def transport(requests):
    answers = CannedBackend([(SAMPLE, ANSWER)])
    def handle(request):
        requests.append(request)
        if request.url.path == '/api/playground/session':
            return httpx.Response(200, json={'token': 'session-token', 'timeout_seconds': 120})
        if request.url.path == '/api/playground/session/close':
            return httpx.Response(200, json={'status': 'closed'})
        assert request.url.path == '/api/playground/operation'
        assert request.headers['authorization'] == 'Bearer session-token'
        body = json.loads(request.content)
        assert set(body) <= {'state', 'system_prompt', 'calls', 'max_tokens', 'repair_feedback'}
        context = InferenceContext(body['state'], body.get('system_prompt'))
        calls = [normalize_call(c['id'], c['operation'], (), c['arguments'], context) for c in body['calls']]
        return httpx.Response(200, json={'responses': {c.id: asdict(answers.answer(context, c)) for c in calls},
                                       'metadata': {'usage': {'input_tokens': 10, 'output_tokens': 5}}})
    return httpx.MockTransport(handle)


class GatewayClientTests(unittest.TestCase):
    def test_base_install_six_operations_modes_batches_and_cleanup(self):
        for mode in ('eager', 'strict', 'speculative'):
            with self.subTest(mode=mode):
                requests = []
                with AnnoClient.server('https://offline.invalid', 'server-token', mode=mode) as client:
                    session = client.runtime.llm_backend.session
                    session.transport = transport(requests)
                    self.assertIsNone(session.http)
                    self.assertEqual(list(client.submit(annotate, [])), [])
                    self.assertFalse(requests)
                    results = list(client.submit(annotate, [SAMPLE]))
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0].status, 'success', results[0].error)
                    self.assertEqual(results[0].value, ANSWER)
                self.assertTrue(session.closed)
                self.assertTrue(session.http.is_closed)
                self.assertEqual(requests[0].headers['authorization'], 'Bearer server-token')
                self.assertEqual(sum(r.url.path == '/api/playground/session' for r in requests), 1)
                self.assertEqual(sum(r.url.path == '/api/playground/session/close' for r in requests), 1)
                calls = [c['operation'] for r in requests if r.url.path.endswith('/operation')
                         for c in json.loads(r.content)['calls']]
                self.assertEqual(set(calls), {'choice', 'score', 'noul', 'generate', 'generate_list', 'extract'})

    def test_actual_quick_start_entrypoint_on_base_dependencies(self):
        requests = []
        original = AnnoClient.server
        def offline():
            client = original('https://offline.invalid')
            client.runtime.llm_backend.session.transport = transport(requests)
            return client
        from contextlib import redirect_stdout
        from io import StringIO
        output = StringIO()
        with patch.object(AnnoClient, 'server', side_effect=offline), redirect_stdout(output):
            self.assertEqual(main(), 0)
        self.assertEqual(json.loads(output.getvalue()), ANSWER)


class GatewayTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_expiry_renews_once_for_concurrent_operations(self):
        stale = asyncio.Event()
        opened, rejected = [], []

        async def handle(request):
            if request.url.path.endswith('/session'):
                opened.append(str(len(opened)))
                return httpx.Response(200, json={'token': opened[-1], 'timeout_seconds': 86400,
                                                'idle_timeout_seconds': 600})
            if request.url.path.endswith('/close'):
                return httpx.Response(200, json={'status': 'closed'})
            if request.headers['authorization'] == 'Bearer 0':
                rejected.append(request)
                if len(rejected) == 2:
                    stale.set()
                await stale.wait()
                return httpx.Response(401, json={'code': 'session_expired'})
            return httpx.Response(200, json={'responses': {}})

        session = GatewaySession('https://offline.invalid', None, 1, transport=httpx.MockTransport(handle))
        body = {'state': SAMPLE, 'calls': [{'id': 'q', 'operation': 'generate', 'arguments': {
            'instructions': 'Write.', 'max_length': 30}}]}
        try:
            results = await asyncio.gather(session.operation(body), session.operation(body))
            self.assertEqual(results, [{'responses': {}}, {'responses': {}}])
            self.assertEqual(opened, ['0', '1'])
            self.assertIsNone(session.expires)
        finally:
            await session.aclose()

    async def test_idle_renewal_stops_after_second_expiry(self):
        opened = []

        def handle(request):
            if request.url.path.endswith('/session'):
                opened.append(str(len(opened)))
                return httpx.Response(200, json={'token': opened[-1], 'timeout_seconds': 86400,
                                                'idle_timeout_seconds': 600})
            if request.url.path.endswith('/close'):
                return httpx.Response(200, json={'status': 'closed'})
            return httpx.Response(401, json={'code': 'session_expired'})

        session = GatewaySession('https://offline.invalid', None, 1, transport=httpx.MockTransport(handle))
        body = {'state': SAMPLE, 'calls': [{'id': 'q', 'operation': 'generate', 'arguments': {
            'instructions': 'Write.', 'max_length': 30}}]}
        try:
            with self.assertRaisesRegex(BackendFatalError, 'session_expired'):
                await session.operation(body)
            self.assertEqual(opened, ['0', '1'])
        finally:
            await session.aclose()

    async def test_timeout_cancels_request_without_retry_or_reopening(self):
        requests, cancelled = [], []
        async def slow(request):
            requests.append(request.url.path)
            if request.url.path.endswith('/session'):
                return httpx.Response(200, json={'token': 'session-token', 'timeout_seconds': 120})
            if request.url.path.endswith('/close'):
                return httpx.Response(200, json={'status': 'closed'})
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.append(True)
        session = GatewaySession('https://offline.invalid', None, .02, transport=httpx.MockTransport(slow))
        body = {'state': SAMPLE, 'calls': [{'id': 'q', 'operation': 'generate', 'arguments': {
            'instructions': 'Write.', 'max_length': 30}}]}
        try:
            for _ in range(2):
                with self.assertRaisesRegex(BackendFatalError, 'gateway_timeout'):
                    await session.operation(body)
        finally:
            await session.aclose()
        self.assertEqual(cancelled, [True])
        self.assertEqual(requests, ['/api/playground/session', '/api/playground/operation', '/api/playground/session/close'])


if __name__ == '__main__':
    unittest.main()
