"""Client-side model proxies for the bounded operation gateway (no provider keys)."""
import asyncio
import math
import time

import httpx
from annoscript._data import thaw
from annoscript.backends.base import (BackendFatalError, BatchResult, InferenceOutput,
    LLMBatchPlan, RESPONSE_TYPES, encode_payload, validate_response)
from annoscript.backends._generation_policy import effective_budget


class GatewaySession:
    def __init__(self, base_url, auth_token, timeout, *, transport=None):
        url = httpx.URL(base_url)
        if url.scheme not in ('http', 'https') or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError('server URL must be HTTP(S) without credentials, query or fragment')
        if auth_token is not None and (type(auth_token) is not str or not auth_token or '\n' in auth_token or '\r' in auth_token):
            raise ValueError('auth_token must be a nonempty server access token or None')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.url, self.auth, self.timeout = base_url.rstrip('/'), auth_token, timeout
        self.transport, self.http, self.token = transport, None, None
        self.lock, self.closed, self.failure = asyncio.Lock(), False, None
        self.expires = None

    def fail(self, code):
        self.failure = code
        raise BackendFatalError('AnnoScript server: ' + code + '. No automatic retry; close this client.', code=code)

    async def post(self, route, body, bearer):
        try:
            response = await asyncio.wait_for(
                self.http.post(self.url + route, content=encode_payload(body),
                    headers={'Content-Type': 'application/json', **({'Authorization': 'Bearer ' + bearer} if bearer else {})}),
                timeout=self.timeout)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            self.fail('gateway_timeout')
        except httpx.RequestError:
            self.fail('gateway_unavailable')
        try:
            value = response.json()
        except ValueError:
            self.fail('invalid_gateway_response')
        if response.status_code != 200:
            known = {'unauthorized', 'session_expired', 'busy', 'operation_busy', 'run_budget',
                     'daily_limit', 'ip_daily_limit', 'rate_limit', 'timeout', 'cancelled',
                     'invalid_input', 'invalid_operation', 'model_error'}
            code = value.get('code') if type(value) is dict else None
            self.fail(code if code in known else 'gateway_http_' + str(response.status_code))
        return value

    async def ensure(self):
        async with self.lock:
            if self.closed:
                self.fail('client_closed')
            if self.failure:
                self.fail(self.failure)
            if self.token is not None:
                if time.monotonic() >= self.expires:
                    self.fail('session_expired')
                return
            if self.http is None:
                self.http = httpx.AsyncClient(timeout=self.timeout,
                    transport=self.transport or httpx.AsyncHTTPTransport(retries=0),
                    follow_redirects=False, trust_env=False)
            # A failed/ambiguous creation must never silently consume a second quota.
            self.failure = 'session_creation_failed'
            value = await self.post('/api/playground/session', {}, self.auth)
            if (type(value) is not dict or type(value.get('token')) is not str or not value['token']
                or type(value.get('timeout_seconds')) not in (int, float)
                or not 0 < value['timeout_seconds'] <= 86400):
                self.fail('invalid_gateway_response')
            self.token = value['token']
            self.expires = time.monotonic() + value['timeout_seconds']
            self.failure = None

    async def operation(self, body):
        # Fail locally before admitting a session when even the envelope is too large.
        if len(encode_payload(body)) > 65536 or len(encode_payload(body['state'])) > 16384:
            self.fail('invalid_operation')
        from annoscript._gateway_validation import validate_operation
        try:
            validate_operation(body)
        except (ValueError, TypeError, KeyError, RecursionError):
            self.fail('invalid_operation')
        await self.ensure()
        return await self.post('/api/playground/operation', body, self.token)

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        if self.http is not None:
            try:
                if self.token:
                    # Cleanup is bounded, best effort, and never creates a session.
                    await self.post('/api/playground/session/close', {}, self.token)
            except BackendFatalError:
                pass
            finally:
                await self.http.aclose()


def payload(context, calls, budget, feedback=None):
    body = {'state': thaw(context.state), 'system_prompt': context.system_prompt,
            'calls': [{'id': c.id, 'operation': c.operation, 'arguments': thaw(c.arguments)} for c in calls],
            'max_tokens': budget}
    if feedback:
        body['repair_feedback'] = feedback
    return body


class GatewayBackend:
    model = 'annoscript-server'
    supports_llm_batch = True

    def __init__(self, session):
        self.session = session

    def resolve_token_budget(self, call, fallback):
        return effective_budget(call, 2)

    def plan_batch(self, context, calls, *, max_calls, max_input_bytes, max_output_tokens, default_tokens):
        selected, budget, plan = [], 64, None
        for call in calls[:min(max_calls, 32)]:
            budget += effective_budget(call, 2) + 32
            if budget > min(max_output_tokens, 4096):
                break
            candidate = selected + [call]
            size = len(encode_payload(payload(context, candidate, budget)))
            if size > min(max_input_bytes, 65536):
                break
            selected = candidate
            if len(selected) > 1:
                plan = LLMBatchPlan(tuple(selected), budget, size)
        return plan

    async def infer_batch(self, context, calls, *, max_tokens=1024):
        value = await self.session.operation(payload(context, calls, max_tokens))
        try:
            responses = {}
            if set(value['responses']) != {c.id for c in calls}:
                raise ValueError('response IDs differ')
            for call in calls:
                row = dict(value['responses'][call.id])
                if call.operation == 'score':
                    for key in ('probabilities', 'legend'):
                        row[key] = {int(k): v for k, v in row[key].items()}
                responses[call.id] = validate_response(call, RESPONSE_TYPES[call.operation](**row))
            return BatchResult(responses, value.get('metadata', {}))
        except (KeyError, TypeError, ValueError):
            self.session.fail('invalid_gateway_response')

    async def infer(self, context, call, *, max_tokens_override=None, repair_feedback=None):
        budget = effective_budget(call, 2, max_tokens_override) if call.operation in ('generate', 'generate_list', 'extract') else 1024
        if not repair_feedback:
            result = await self.infer_batch(context, [call], max_tokens=budget)
            return InferenceOutput(result.responses[call.id], result.metadata)
        # Repair is supported by the wire protocol, but never retries quota/transport failures.
        value = await self.session.operation(payload(context, [call], budget, repair_feedback))
        return InferenceOutput(validate_response(call, RESPONSE_TYPES[call.operation](**value['responses'][call.id])),
                               value.get('metadata', {}))

    async def aclose(self):
        await self.session.aclose()
