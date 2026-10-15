"""Short-request JSON API. No model backend is constructed by this process."""

import asyncio
from contextlib import asynccontextmanager, suppress
import hmac

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from lev.job_protocol import decode, integer, nonempty
from lev.remote import RemoteProtocolError
from lev.server.store import ServiceError


def create_app(store, releases, tokens):
    if not tokens or any(not nonempty(k) or not nonempty(v) for k, v in tokens.items()):
        raise ValueError("service tokens must map to nonempty principals")

    @asynccontextmanager
    async def lifespan(app):
        async def maintain():
            while True:
                await asyncio.to_thread(store.maintain)
                await asyncio.sleep(1)
        await asyncio.to_thread(store.maintain)
        task = asyncio.create_task(maintain())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None,
                  redirect_slashes=False)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse({"error": {"code": error.code, "message": str(error)}}, status_code=error.status)

    def principal(request: Request):
        auth = request.headers.get("Authorization", "")
        if not auth.lower().startswith("bearer "):
            raise ServiceError("authentication_failed", "service authentication required", 401)
        supplied = auth[7:]
        for token, identity in tokens.items():
            if hmac.compare_digest(supplied.encode(), token.encode()):
                return identity
        raise ServiceError("authentication_failed", "invalid service token", 401)

    def key(request):
        value = request.headers.get("Idempotency-Key")
        if not nonempty(value) or len(value) > 256:
            raise ServiceError("invalid_idempotency_key", "a nonempty Idempotency-Key of at most 256 characters is required")
        return value

    async def body(request, *, empty=False):
        limit = max(m["policy"]["max_request_bytes"] for m in releases.manifests.values())
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > limit:
                raise ServiceError("request_too_large", "request exceeds service body limit", 413)
        if not data and empty:
            return {}
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            raise ServiceError("invalid_content_type", "expected application/json", 415)
        try:
            value = decode(bytes(data))
        except RemoteProtocolError as exc:
            raise ServiceError("invalid_json", str(exc)) from None
        if type(value) is not dict:
            raise ServiceError("invalid_request", "request must be an object")
        return value

    def envelope(job):
        return {"protocol_version": 1, "job": job}

    @app.post("/v1/jobs", status_code=202)
    async def submit(request: Request, identity=Depends(principal)):
        idem = key(request)
        payload = await body(request)
        try:
            job = await asyncio.to_thread(store.submit, identity, idem, payload, releases)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ServiceError("invalid_request", "invalid job request or IR") from exc
        return envelope(job)

    @app.get("/v1/jobs:lookup")
    def lookup(request: Request, identity=Depends(principal)):
        return envelope(store.lookup(identity, key(request)))

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str, identity=Depends(principal)):
        return envelope(store.status(identity, job_id))

    @app.get("/v1/jobs/{job_id}/results")
    def results(job_id: str, request: Request, identity=Depends(principal)):
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            raise ServiceError("invalid_limit", "invalid page limit") from None
        return store.results(identity, job_id, request.query_params.get("cursor"), limit)

    @app.post("/v1/jobs/{job_id}:resume")
    async def resume(job_id: str, request: Request, identity=Depends(principal)):
        idem = key(request)
        payload = await body(request)
        if set(payload) != {"pause_generation"} or not integer(payload["pause_generation"], 1):
            raise ServiceError("invalid_request", "resume requires a positive pause_generation")
        return envelope(await asyncio.to_thread(store.resume, identity, job_id, idem, payload["pause_generation"], releases))

    @app.post("/v1/jobs/{job_id}:cancel")
    async def cancel(job_id: str, request: Request, identity=Depends(principal)):
        if await body(request, empty=True):
            raise ServiceError("invalid_request", "cancel body must be empty")
        return envelope(await asyncio.to_thread(store.cancel, identity, job_id))

    @app.get("/v1/releases/{release_id}")
    def release(release_id: str, identity=Depends(principal)):
        return {"protocol_version": 1, **store.release(release_id, releases)}

    @app.post("/v1/sessions", status_code=201)
    async def create_session(request: Request, identity=Depends(principal)):
        idem = key(request)
        if await body(request, empty=True):
            raise ServiceError("invalid_request", "session body must be empty")
        return {"protocol_version": 1, **await asyncio.to_thread(store.create_session, identity, idem)}

    @app.post("/v1/sessions/{session_id}:heartbeat")
    async def heartbeat(session_id: str, request: Request, identity=Depends(principal)):
        if await body(request, empty=True):
            raise ServiceError("invalid_request", "heartbeat body must be empty")
        return {"protocol_version": 1, **await asyncio.to_thread(store.heartbeat_session, identity, session_id)}

    @app.post("/v1/sessions/{session_id}:close")
    async def close_session(session_id: str, request: Request, identity=Depends(principal)):
        if await body(request, empty=True):
            raise ServiceError("invalid_request", "close body must be empty")
        return {"protocol_version": 1, **await asyncio.to_thread(store.close_session, identity, session_id)}

    return app
