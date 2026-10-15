"""Python 3.10+ durable job HTTP client. Closing it does not cancel durable jobs."""

from __future__ import annotations

import math
import random
import threading
import time
from urllib.parse import quote

from lev.job_protocol import (decode, encode, nonempty, retry_after_seconds,
                              validate_page, validate_request, validate_status)
from lev.remote import (RemoteProtocolError, RemoteSubmissionError,
                        RemoteTransportError, ir_sha256, _error)


class RemoteJobPausedError(RemoteSubmissionError):
    def __init__(self, job):
        reason = job["pause_reason"]
        super().__init__(reason["code"], reason["message"])
        self.job = job


class RemoteCancellationError(RemoteTransportError):
    """Cancellation was requested but could not be confirmed before closing."""


def job_id(job):
    value = job.get("job_id") if type(job) is dict else job
    if not nonempty(value):
        raise ValueError("job must be a job status or nonempty job ID")
    return value


class LevJobClient:
    def __init__(self, base_url, auth_token=None, *, timeout=30.0, transport=None,
                 _clock=time.monotonic, _wall_clock=time.time, _sleep=time.sleep,
                 _random=random.random):
        import httpx
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        if auth_token is not None and (not nonempty(auth_token) or "\r" in auth_token or "\n" in auth_token):
            raise ValueError("auth_token must be a nonempty single-line string")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url, self.timeout = str(url).rstrip("/"), timeout
        self.clock, self.wall_clock, self.sleep, self.random = _clock, _wall_clock, _sleep, _random
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth_token is not None:
            headers["Authorization"] = auth_token if auth_token.lower().startswith("bearer ") else "Bearer " + auth_token
        self._http = httpx.Client(headers=headers, timeout=timeout, follow_redirects=False,
                                  transport=transport if transport is not None else httpx.HTTPTransport(retries=0))
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise RuntimeError("client is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self._http.close()

    def _request(self, method, path, *, payload=None, key=None, params=None,
                 budget=120.0, attempts=5):
        # httpx timeouts are per network phase; DNS or a slowly progressing
        # response must not turn a bounded reconnect budget into an endless wait.
        # A late request can still reach the server, so all mutations are either
        # idempotent or guarded by a closed ownership session.
        outcome, done = [], threading.Event()
        def run():
            try:
                outcome.append((True, self._request_with_retries(method, path, payload=payload,
                    key=key, params=params, budget=budget, attempts=attempts)))
            except BaseException as exc:
                outcome.append((False, exc))
            finally:
                done.set()
        threading.Thread(target=run, name="lev-job-http", daemon=True).start()
        if not done.wait(max(0.0, budget)):
            raise RemoteTransportError("remote reconnect deadline exceeded")
        success, value = outcome[0]
        if not success:
            raise value
        return value

    def _request_with_retries(self, method, path, *, payload=None, key=None, params=None,
                              budget=120.0, attempts=5):
        import httpx
        if self.closed:
            raise RuntimeError("client is closed")
        if key is not None and (not nonempty(key) or len(key) > 256 or "\r" in key or "\n" in key):
            raise ValueError("invalid idempotency key")
        content = encode(payload) if payload is not None else None
        deadline = self.clock() + budget
        for attempt in range(attempts):
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise RemoteTransportError("remote reconnect budget exhausted")
            retry_after = None
            try:
                with self._http.stream(method, self.base_url + path, content=content, params=params,
                                       headers={"Idempotency-Key": key} if key is not None else None,
                                       timeout=min(self.timeout, remaining)) as response:
                    chunks = []
                    for chunk in response.iter_bytes():
                        if self.clock() > deadline:
                            raise httpx.ReadTimeout("request deadline exceeded")
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    if not 200 <= response.status_code < 300:
                        code, message = "http_error", f"remote HTTP error {response.status_code}"
                        try:
                            error = _error(decode(raw)["error"])
                            code, message = error.code, error.message
                        except (RemoteProtocolError, KeyError, TypeError):
                            pass
                        error = RemoteSubmissionError(code, message, http_status=response.status_code,
                                                      retry_after=response.headers.get("Retry-After"))
                        if response.status_code not in (408, 429) and not 500 <= response.status_code <= 599:
                            raise error
                        failure, retry_after = error, error.retry_after
                    else:
                        if response.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json":
                            raise RemoteProtocolError("expected an application/json response")
                        value = decode(raw)
                        if type(value) is not dict or type(value.get("protocol_version")) is not int or value["protocol_version"] != 1:
                            raise RemoteProtocolError("unsupported job protocol version")
                        return value
            except httpx.RequestError as exc:
                failure = RemoteTransportError("remote HTTP request interrupted")
                failure.__cause__ = exc
            if attempt + 1 == attempts:
                raise failure
            delay = max(self.random() * min(10.0, 0.5 * 2 ** attempt),
                        retry_after_seconds(retry_after, self.wall_clock()))
            if self.clock() + delay >= deadline:
                raise failure
            self.sleep(delay)
        raise RemoteTransportError("remote reconnect budget exhausted")

    @staticmethod
    def _status(response, expected_id=None):
        if set(response) != {"protocol_version", "job"}:
            raise RemoteProtocolError("invalid job envelope")
        job = validate_status(response["job"])
        if expected_id is not None and job["job_id"] != expected_id:
            raise RemoteProtocolError("response belongs to another job")
        return job

    @staticmethod
    def _binding(expected, actual):
        if type(expected) is dict:
            for field in ("job_id", "ir_sha256", "release_id", "mode", "input_indices"):
                if expected[field] != actual[field]:
                    raise RemoteProtocolError("immutable job binding changed")
        return actual

    def submit_ir(self, ir, inputs, *, idempotency_key, mode="eager", release_id=None,
                  application_revision=None, session_id=None):
        ir = ir.to_dict() if hasattr(ir, "to_dict") else ir
        payload = {"protocol_version": 1, "ir": ir, "ir_sha256": ir_sha256(ir), "mode": mode, "inputs": inputs}
        for name, value in (("release_id", release_id), ("application_revision", application_revision), ("session_id", session_id)):
            if value is not None:
                payload[name] = value
        payload = validate_request(payload)
        job = self._status(self._request("POST", "/v1/jobs", payload=payload, key=idempotency_key))
        if (job["ir_sha256"] != payload["ir_sha256"] or job["mode"] != mode
                or job["input_indices"] != [item["index"] for item in payload["inputs"]]
                or (release_id is not None and job["release_id"] != release_id)):
            raise RemoteProtocolError("accepted job does not match submitted IR, inputs or release")
        return job

    def lookup(self, idempotency_key, **options):
        return self._status(self._request("GET", "/v1/jobs:lookup", key=idempotency_key, **options))

    def status(self, job):
        jid = job_id(job)
        return self._binding(job, self._status(self._request("GET", f"/v1/jobs/{quote(jid, safe='')}"), jid))

    def fetch_results(self, job, *, cursor=None, limit=100):
        jid = job_id(job)
        params = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        page = validate_page(self._request("GET", f"/v1/jobs/{quote(jid, safe='')}/results", params=params), jid, cursor)
        self._binding(job, page["job"])
        return page

    def resume(self, job, *, pause_generation, idempotency_key):
        jid = job_id(job)
        return self._binding(job, self._status(self._request("POST", f"/v1/jobs/{quote(jid, safe='')}:resume",
                                          payload={"pause_generation": pause_generation}, key=idempotency_key), jid))

    def cancel(self, job, **options):
        jid = job_id(job)
        result = self._status(self._request("POST", f"/v1/jobs/{quote(jid, safe='')}:cancel", payload={}, **options), jid)
        self._binding(job, result)
        if result["state"] not in ("completed", "cancelled", "failed"):
            raise RemoteProtocolError("cancellation was not persisted")
        return result

    def _session(self, response):
        if (set(response) != {"protocol_version", "session_id", "state", "expires_at", "heartbeat_interval"}
                or not nonempty(response["session_id"]) or response["state"] not in ("active", "closed", "expired")
                or any(type(response[k]) not in (int, float) or not math.isfinite(response[k]) or response[k] <= 0
                       for k in ("expires_at", "heartbeat_interval"))):
            raise RemoteProtocolError("invalid session response")
        return response

    def _create_session(self, key):
        response = self._session(self._request("POST", "/v1/sessions", payload={}, key=key))
        if response["state"] != "active":
            raise RemoteSubmissionError("session_closed", "client session has ended")
        return response

    def _heartbeat(self, session_id):
        response = self._session(self._request("POST", f"/v1/sessions/{quote(session_id, safe='')}:heartbeat",
                                               payload={}, attempts=1, budget=30))
        if response["session_id"] != session_id or response["state"] != "active":
            raise RemoteProtocolError("invalid session heartbeat")
        return response

    def _close_session(self, session_id, budget):
        response = self._session(self._request("POST", f"/v1/sessions/{quote(session_id, safe='')}:close",
                                               payload={}, attempts=1, budget=budget))
        if response["session_id"] != session_id or response["state"] == "active":
            raise RemoteProtocolError("session cancellation was not persisted")
        return response
