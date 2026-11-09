"""Offline operation gateway acceptance; providers never leave HTTP fixtures."""
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from unittest.mock import patch

import httpx
import uvicorn
from annoscript._gateway import GatewaySession, GatewayBackend
from annoscript._test_evaluation_http import http_backends
from annoscript.backends.base import InferenceContext, normalize_call, BackendFatalError
from annoscript.client import AnnoClient
from annoscript.server.gateway import create_app, app_from_env
from annoscript.server.quota import Quota
from annoscript_examples.quick_start import SAMPLE
from annoscript_examples.request import annotate

ANSWER = {'category': 'note', 'highlight': True, 'clarity': 1., 'title': 'Library opening',
          'tags': ['library'], 'spans': ['library']}


@contextmanager
def loopback(app):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='critical', access_log=False))
        thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError('loopback gateway failed to start')
                time.sleep(.01)
            yield f'http://127.0.0.1:{port}'
        finally:
            server.should_exit = True
            thread.join(5)
            if thread.is_alive():
                raise RuntimeError('gateway did not stop')


def body(operation='choice'):
    args = {'instructions': 'Choose.', 'criteria': {'a': 'A', 'b': 'B'}} if operation == 'choice' else {
        'instructions': 'Write.', 'max_length': 30, 'max_num_tokens': 32}
    return {'state': 'synthetic', 'calls': [{'id': 'q', 'operation': operation, 'arguments': args}], 'max_tokens': 32}


class GatewayClientTests(unittest.TestCase):
    def test_real_client_loopback_six_operations_batches_and_budgets(self):
        requests = []
        with TemporaryDirectory() as directory:
            app = create_app(quota=Quota(Path(directory)/'q.db', 'fixture', minute=100), access_token='server-token',
                backend_factory=lambda **kw: http_backends([(SAMPLE, ANSWER)], requests=requests))
            with loopback(app) as url:
                with AnnoClient.server(url, 'server-token') as client:
                    session = client.runtime.decision_backend.session
                    self.assertIsNone(session.http)
                    self.assertEqual(app.state.editor_sessions.items, {})
                    self.assertEqual(list(client.submit(annotate, [])), [])
                    self.assertIsNone(session.http)
                    result = list(client.submit(annotate, [SAMPLE]))[0]
                    self.assertEqual(result.status, 'success', result.error)
                    self.assertEqual(result.value, ANSWER)
                    self.assertEqual(len(app.state.editor_sessions.items), 1)
                    self.assertEqual(app.state.active, 0)
                    self.assertIs(session, client.runtime.llm_backend.session)
                self.assertEqual(app.state.editor_sessions.items, {})
                self.assertEqual(app.state.active, 0)
        self.assertTrue(any('questions' in r and len(r['questions']) == 3 for r in requests))
        generations = [r for r in requests if 'messages' in r]
        self.assertTrue(generations)
        self.assertTrue(all(1 <= r['max_tokens'] <= 4096 for r in generations))

    def test_constructor_close_and_no_provider_environment_read(self):
        with patch.dict(os.environ, {'ANNO_GENERATE_AUTH_TOKEN': 'must-not-transfer'}):
            with AnnoClient.server() as client:
                session = client.runtime.llm_backend.session
                self.assertIsNone(session.auth)
                self.assertIsNone(session.http)
            self.assertTrue(session.closed)

    def test_default_auth_and_startup_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'TOKEN'):
            create_app(quota=object())
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'HMAC'):
                app_from_env()
        with TemporaryDirectory() as directory:
            app = create_app(quota=Quota(Path(directory)/'q', 'fixture'), access_token='server-token')
            with loopback(app) as url, httpx.Client(base_url=url) as http:
                self.assertEqual(http.post('/api/playground/session', json={}).status_code, 401)
                self.assertEqual(http.post('/api/playground/session', json={}, headers={'Authorization': 'Bearer wrong'}).status_code, 401)
                self.assertEqual(app.state.editor_sessions.items, {})
                self.assertEqual(http.get('/healthz').status_code, 200)


class AsyncGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory()
        self.calls, self.finished = [], []
        self.block = asyncio.Event()
        from annoscript.anno import ChoiceResponse, GenerateResponse
        from annoscript.backends.base import InferenceOutput, BatchResult
        owner = self
        class Backend:
            model = 'fixture'
            async def infer_batch(self, context, calls):
                owner.calls.append('decision')
                try:
                    await owner.block.wait()
                    return BatchResult({c.id: ChoiceResponse('a', 1., {'a': 1., 'b': 0.}) for c in calls})
                finally:
                    owner.finished.append('decision')
            async def infer(self, context, call, *, max_tokens_override=None, repair_feedback=None):
                owner.calls.append(max_tokens_override)
                try:
                    await owner.block.wait()
                    return InferenceOutput(GenerateResponse('Synthetic'))
                finally:
                    owner.finished.append('generation')
            async def aclose(self): pass
        self.app = create_app(quota=Quota(Path(self.directory.name)/'q', 'fixture', minute=100), anonymous=True,
            concurrency=1, backend_factory=lambda **kw: (Backend(), Backend()))
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(self.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.life.__aexit__(None, None, None)
        self.directory.cleanup()

    async def session(self):
        value = (await self.http.post('/api/playground/session', json={})).json()
        return value['token'], {'Authorization': 'Bearer ' + value['token']}

    async def started(self, count):
        for _ in range(100):
            if len(self.calls) >= count: return
            await asyncio.sleep(.005)
        self.fail('fixture did not start')

    async def test_empty_leases_dual_backend_one_slot_and_release(self):
        token, headers = await self.session()
        other, second = await self.session()
        self.assertEqual(self.app.state.active, 0)
        decision = asyncio.create_task(self.http.post('/api/playground/operation', json=body(), headers=headers))
        await self.started(1)
        self.assertEqual(self.app.state.active, 1)
        generation = asyncio.create_task(self.http.post('/api/playground/operation', json=body('generate'), headers=headers))
        await self.started(2)
        self.assertEqual(self.app.state.active, 1)
        blocked = await self.http.post('/api/playground/operation', json=body(), headers=second)
        self.assertEqual(blocked.json()['code'], 'busy')
        self.assertEqual(self.app.state.editor_sessions.items[other].requests, 0)
        self.block.set()
        self.assertEqual([r.status_code for r in await asyncio.gather(decision, generation)], [200, 200])
        self.assertEqual(self.app.state.active, 0)
        self.assertEqual((await self.http.post('/api/playground/operation', json=body(), headers=second)).status_code, 200)
        for h in (headers, headers, second):
            await self.http.post('/api/playground/session/close', json={}, headers=h)
        self.assertEqual(self.app.state.active, 0)

    async def test_close_expiry_cancel_and_no_double_release(self):
        token, headers = await self.session()
        request = asyncio.create_task(self.http.post('/api/playground/operation', json=body(), headers=headers))
        await self.started(1)
        await self.http.post('/api/playground/session/close', json={}, headers=headers)
        self.assertEqual((await request).json()['code'], 'cancelled')
        self.app.state.editor_sessions.expire()
        await self.http.post('/api/playground/session/close', json={}, headers=headers)
        self.assertEqual(self.finished, ['decision'])
        self.assertEqual(self.app.state.active, 0)
        token, headers = await self.session()
        self.app.state.editor_sessions.items[token].expires = time.monotonic() - 1
        self.assertEqual((await self.http.post('/api/playground/operation', json=body(), headers=headers)).status_code, 401)
        self.assertEqual(self.app.state.active, 0)

    async def test_expiry_while_running_times_out_and_releases(self):
        token, headers = await self.session()
        self.app.state.editor_sessions.items[token].expires = time.monotonic() + .03
        response = await self.http.post('/api/playground/operation', json=body(), headers=headers)
        self.assertEqual(response.json()['code'], 'timeout')
        self.app.state.editor_sessions.expire()
        self.assertEqual(self.app.state.active, 0)

    async def test_budget_validation_and_no_renewal_after_error(self):
        session = GatewaySession('http://test', None, 1, transport=httpx.ASGITransport(self.app))
        backend = GatewayBackend(session)
        context = InferenceContext('synthetic')
        call = normalize_call('q', 'generate', (), {'instructions': 'Write.', 'max_length': 30, 'max_num_tokens': 4096}, context)
        self.block.set()
        for _ in range(2):
            await backend.infer(context, call, max_tokens_override=4096)
        with self.assertRaisesRegex(BackendFatalError, 'run_budget'):
            await backend.infer(context, call, max_tokens_override=32)
        token = session.token
        with self.assertRaisesRegex(BackendFatalError, 'run_budget'):
            await backend.infer(context, call, max_tokens_override=32)
        self.assertEqual(session.token, token)
        self.assertEqual(len(self.calls), 2)
        await backend.aclose()
        self.assertEqual(self.app.state.active, 0)

    async def test_real_provider_single_operations_and_exact_budgets(self):
        requests = []
        app = create_app(quota=Quota(Path(self.directory.name)/'real', 'fixture', minute=100), anonymous=True,
            backend_factory=lambda **kw: http_backends([(SAMPLE, ANSWER)], requests=requests))
        async with app.router.lifespan_context(app):
            session = GatewaySession('http://test', None, 1, transport=httpx.ASGITransport(app))
            backend = GatewayBackend(session)
            context = InferenceContext(SAMPLE)
            specs = [('choice', 'category', {'criteria': {'note': 'A note', 'other': 'Other'}}),
                ('noul', 'highlight', {}), ('score', 'clarity', {'criteria': ['low', 'high']}),
                ('generate', 'title', {'max_length': 64}),
                ('generate_list', 'tags', {'max_items': 3, 'max_length_per_item': 32}),
                ('extract', 'spans', {'source_field': 'text', 'max_items': 3, 'max_length_per_item': 64})]
            for operation, field, args in specs:
                call = normalize_call(field, operation, (), {'instructions': 'Annotate.', 'field_name': field, **args}, context)
                await backend.infer(context, call, max_tokens_override=48)
                if operation in ('generate', 'generate_list', 'extract'):
                    self.assertEqual(requests[-1]['max_tokens'], 48)
            await backend.aclose()
            self.assertEqual(len(requests), 6)
            self.assertEqual(app.state.active, 0)

    async def test_network_timeout_does_not_retry_or_renew(self):
        requests = []
        async def handle(request):
            requests.append(request.url.path)
            if request.url.path.endswith('/session'):
                return httpx.Response(200, json={'token': 'fixture', 'timeout_seconds': 120})
            if request.url.path.endswith('/close'):
                return httpx.Response(200, json={'status': 'closed'})
            await asyncio.sleep(1)
        session = GatewaySession('http://test', None, .02, transport=httpx.MockTransport(handle))
        for _ in range(2):
            with self.assertRaisesRegex(BackendFatalError, 'gateway_timeout'):
                await session.operation(body())
        await session.aclose()
        self.assertEqual(requests.count('/api/playground/session'), 1)
        self.assertEqual(requests.count('/api/playground/operation'), 1)


if __name__ == '__main__':
    unittest.main()
