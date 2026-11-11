"""Bounded model operations for Python running locally or in a browser Web Worker.

No Python source, IR, imports, provider URLs or credentials are accepted here.
"""
import asyncio
from dataclasses import asdict, dataclass, field
import json
import secrets
import time

from fastapi import Request
from fastapi.responses import JSONResponse


IDLE_TIMEOUT = 600


@dataclass
class Session:
    expires: float
    requests: int = 0
    output_budget: int = 8192
    tasks: set = field(default_factory=set)
    inflight: set = field(default_factory=set)
    slot: bool = False


class Sessions:
    def __init__(self, *, unlimited=False):
        self.items = {}
        self.unlimited = unlimited

    def close(self, token):
        item = self.items.pop(token, None)
        if item:
            for task in item.tasks:
                task.cancel()
            item.tasks.clear()

    def expire(self):
        for token, item in list(self.items.items()):
            if self.expired(item):
                self.close(token)

    def expired(self, item):
        return item.expires <= time.monotonic() and (not self.unlimited or not item.tasks)

    async def reap_idle(self):
        while True:
            await asyncio.sleep(30)
            self.expire()


async def wait_for_disconnect(request):
    while (await request.receive())["type"] != "http.disconnect":
        pass


def install_gateway(app, allowed, limit, duration, backend_factory=None, *, access_token=None, unlimited=False):
    sessions = Sessions(unlimited=unlimited)
    app.state.editor_sessions = sessions

    def response(code, status=400):
        return JSONResponse({"code": code}, status_code=status, headers={"Cache-Control": "no-store"})

    async def read(request):
        if request.headers.get("origin") and request.headers["origin"] not in allowed:
            raise ValueError("origin")
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise ValueError("content type")
        raw = bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw) > 65536:
                raise ValueError("body too large")
        data = json.loads(raw)
        if type(data) is not dict:
            raise ValueError("object required")
        return data

    def token(request):
        value = request.headers.get("authorization", "")
        return value[7:] if value.startswith("Bearer ") else ""

    @app.post("/api/playground/session")
    async def begin(request: Request):
        if access_token is not None and not secrets.compare_digest(token(request), access_token):
            return response("unauthorized", 401)
        try:
            if await read(request) != {}:
                return response("invalid_input", 422)
        except (ValueError, RecursionError):
            return response("invalid_input", 422)
        if not unlimited:
            sessions.expire()
            denial = app.state.quota.admit(request.client.host if request.client else "unknown")
            if denial:
                return response(denial[0], 429)
        identity = secrets.token_urlsafe(32)
        sessions.items[identity] = Session(time.monotonic() + (IDLE_TIMEOUT if unlimited else duration))
        if unlimited:
            # Older SDKs require a positive timeout_seconds of at most one day.
            info = {"timeout_seconds": 86400, "idle_timeout_seconds": IDLE_TIMEOUT,
                    "max_requests": None, "max_output_tokens": None}
        else:
            info = {"timeout_seconds": duration, "max_requests": 12, "max_output_tokens": 8192}
        return JSONResponse({"token": identity, **info}, headers={"Cache-Control": "no-store"})

    @app.post("/api/playground/session/close")
    async def close(request: Request):
        try:
            await read(request)
        except (ValueError, RecursionError):
            return response("invalid_input", 422)
        sessions.close(token(request))
        return {"status": "closed"}

    @app.post("/api/playground/operation")
    async def operation(request: Request):
        if not unlimited:
            sessions.expire()
        identity = token(request)
        item = sessions.items.get(identity)
        if item is None or sessions.expired(item):
            return response("session_expired", 401)
        try:
            data = await read(request)
            from annoscript._gateway_validation import validate_operation
            context, calls, decision, budget, feedback = validate_operation(data)
        except (ValueError, TypeError, KeyError, RecursionError):
            return response("invalid_operation", 422)
        # Reading the body can yield to a concurrent close or expiry.
        if sessions.items.get(identity) is not item or sessions.expired(item):
            return response("session_expired", 401)
        kind = 'decision' if decision else 'generation'
        if not unlimited:
            if kind in item.inflight:
                return response("operation_busy", 429)
            if item.requests >= 12 or (not decision and budget > item.output_budget):
                return response("run_budget", 429)
            if not item.slot:
                if app.state.active >= limit:
                    return response("busy", 429)
                item.slot = True
                app.state.active += 1
            item.requests += 1
            if not decision:
                item.output_budget -= budget
            item.inflight.add(kind)

        async def infer():
            from annoscript.backends.config import backends_from_env
            from annoscript._data import thaw
            backends = (backend_factory or backends_from_env)(timeout=45, disable_thinking=True)
            backend = backends[0 if decision else 1]
            try:
                if not decision and len(calls) == 1:
                    result = await backend.infer(context, calls[0], max_tokens_override=budget,
                                                 **({"repair_feedback": feedback} if feedback else {}))
                    values, metadata = {calls[0].id: result.response}, result.metadata
                else:
                    result = await backend.infer_batch(context, calls, **({} if decision else {"max_tokens": budget}))
                    if result.errors:
                        raise ValueError("model result failed")
                    values, metadata = result.responses, result.metadata
                usage = {k: v for k, v in metadata.get("usage", {}).items()
                         if k in {"input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"}
                         and type(v) is int and v >= 0}
                return {"responses": {key: asdict(value) for key, value in values.items()},
                        "metadata": {"usage": usage}, "model": backend.model}
            finally:
                await asyncio.gather(*(b.aclose() for b in backends))

        task = asyncio.create_task(infer())
        item.tasks.add(task)
        disconnect = asyncio.create_task(wait_for_disconnect(request))
        try:
            timeout = 45 if unlimited else max(.01, min(45, item.expires - time.monotonic()))
            done, _ = await asyncio.wait({task, disconnect}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                return response("cancelled", 409)
            if task not in done:
                return response("timeout", 504)
            return JSONResponse(task.result(), headers={"Cache-Control": "no-store"})
        except asyncio.CancelledError:
            return response("cancelled", 409)
        except TimeoutError:
            return response("timeout", 504)
        except Exception:
            return response("model_error", 502)
        finally:
            if not task.done() and task in item.tasks:
                # Claim cancellation before awaiting cleanup so close cannot interrupt it again.
                item.tasks.remove(task)
                if unlimited:
                    item.expires = time.monotonic() + IDLE_TIMEOUT
                task.cancel()
            disconnect.cancel()
            try:
                await asyncio.gather(task, disconnect, return_exceptions=True)
            finally:
                item.tasks.discard(task)
                if unlimited:
                    item.expires = time.monotonic() + IDLE_TIMEOUT
                else:
                    item.inflight.discard(kind)
                    if not item.inflight and item.slot:
                        item.slot = False
                        app.state.active -= 1

    return sessions


def create_app(*, backend_factory=None, quota=None, access_token=None, anonymous=False,
               origins=(), concurrency=4, duration=120):
    """Build a single-process gateway. Authentication is required by default."""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    if not anonymous and (type(access_token) is not str or not access_token.strip()):
        raise ValueError('ANNO_SERVER_TOKEN is required; anonymous mode must be explicit')
    if type(concurrency) is not int or concurrency < 1 or not 0 < duration <= 86400:
        raise ValueError('invalid gateway concurrency or duration')
    if quota is None:
        raise ValueError('persistent quota is required')

    @asynccontextmanager
    async def lifespan(app):
        app.state.quota, app.state.active = quota, 0
        yield
        tasks = [task for item in sessions.items.values() for task in item.tasks]
        for identity in list(sessions.items):
            sessions.close(identity)
        await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    sessions = install_gateway(app, set(origins), concurrency, duration, backend_factory,
                               access_token=None if anonymous else access_token)

    @app.get('/healthz')
    async def health():
        from annoscript import __version__
        return {'status': 'ok', 'service': 'annoscript-gateway', 'version': __version__}
    return app


def app_from_env():
    """Validate configuration without contacting any model service."""
    import os
    from annoscript._env import normalize
    from annoscript.backends.config import backends_from_env
    from annoscript.server.quota import Quota
    values = normalize(os.environ)
    secret = values.get('ANNO_QUOTA_HMAC_SECRET', '')
    if len(secret) < 16:
        raise ValueError('ANNO_QUOTA_HMAC_SECRET must contain at least 16 characters')
    token = values.get('ANNO_SERVER_TOKEN')
    anonymous = values.get('ANNO_SERVER_ANONYMOUS', '0')
    if anonymous not in ('0', '1'):
        raise ValueError('ANNO_SERVER_ANONYMOUS must be 0 or 1')
    if anonymous != '1' and not token:
        raise ValueError('ANNO_SERVER_TOKEN is required')
    # Constructors validate URLs, adapters and models; they do not open connections.
    backends_from_env(timeout=45)
    quota = Quota(values.get('ANNO_QUOTA_DB', 'anno-quota.sqlite'), secret,
        daily=int(values.get('ANNO_SERVER_DAILY_LIMIT', '1000')),
        ip_daily=int(values.get('ANNO_SERVER_IP_DAILY_LIMIT', '50')),
        minute=int(values.get('ANNO_SERVER_IP_MINUTE_LIMIT', '5')))
    return create_app(quota=quota, access_token=token, anonymous=anonymous == '1',
        origins=filter(None, values.get('ANNO_SERVER_ORIGINS', '').split(',')),
        concurrency=int(values.get('ANNO_SERVER_CONCURRENCY', '4')))
