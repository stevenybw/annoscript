"""Python 3.10+ synchronous IR-over-HTTP client; no local execution imports.

The wire contract is documented in remote_protocol.md. A stream owns its HTTP
response; the client owns the connection pool. Neither layer retries requests.
"""

from __future__ import annotations

import hashlib
import json
import math

from lev._client_common import Result, ResultError, ResultStream, prepare_inputs
from lev._data import snapshot
from lev.compiler import compile_program


PROTOCOL_VERSION = 1


class RemoteSubmissionError(RuntimeError):
    """The HTTP service rejected or aborted a submission."""

    def __init__(self, code, message, *, http_status=None, retry_after=None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.retry_after = retry_after


class RemoteTransportError(RuntimeError):
    """Network failure; remote work may already have started or completed."""


class RemoteProtocolError(RuntimeError):
    """Malformed, mismatched or incomplete remote results."""


def ir_sha256(ir):
    """Hash the complete wire IR, including diagnostic source coordinates."""
    encoded = json.dumps(ir, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("non-finite JSON number")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        return snapshot(value)
    except (ValueError, UnicodeError, RecursionError):
        raise RemoteProtocolError("response must contain finite UTF-8 JSON without duplicate keys") from None


def _error(value):
    if (type(value) is not dict or set(value) != {"code", "message"}
            or type(value["code"]) is not str or not value["code"]
            or type(value["message"]) is not str):
        raise RemoteProtocolError("error must contain a nonempty code and a string message")
    return ResultError(**value)


class RemoteLevClient:
    """Compile and optimize locally, then submit IR to a remote executor.

    Construction does not open connections or read credentials. ``timeout`` is
    a positive HTTP phase timeout, not a total execution deadline. ``transport``
    accepts an httpx synchronous transport, primarily for offline tests.

    Client/stream lifecycle and iteration are single-threaded. Use separate
    clients for independent worker threads. Close streams when stopping early.
    """

    def __init__(self, base_url, auth_token=None, *, mode="eager", optimize=True,
                 timeout=300.0, transport=None):
        import httpx

        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        if auth_token is not None and (type(auth_token) is not str or not auth_token.strip()
                                       or "\r" in auth_token or "\n" in auth_token):
            raise ValueError("auth_token must be a nonempty single-line string or None")
        if mode not in ("eager", "strict", "speculative"):
            raise ValueError("mode must be eager, strict or speculative")
        if type(optimize) is not bool:
            raise TypeError("optimize must be a boolean")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url = str(url).rstrip("/")
        self.mode, self.optimize, self.timeout = mode, optimize, timeout
        self._token, self._transport = auth_token, transport
        self._http = None
        self._streams = set()
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise RuntimeError("client is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        failure = None
        for stream in list(self._streams):
            try:
                stream.close()
            except BaseException as exc:
                failure = failure or exc
        if self._http is not None:
            try:
                self._http.close()
            except BaseException as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def _start(self):
        import httpx

        if self._http is None:
            headers = {"Accept": "application/x-ndjson", "Content-Type": "application/json"}
            if self._token is not None:
                headers["Authorization"] = (self._token if self._token.lower().startswith("bearer ")
                                             else "Bearer " + self._token)
            self._http = httpx.Client(headers=headers, timeout=self.timeout,
                                      follow_redirects=False,
                                      transport=self._transport if self._transport is not None
                                      else httpx.HTTPTransport(retries=0))

    def submit(self, program, inputs, *, ordered=False):
        """Return indexed results, including local invalid_input failures.

        All checks and compilation finish before HTTP dispatch. No valid inputs
        means no HTTP work. Compiler/checker defects abort submission. Remote
        failures can occur during submit or iteration; previously yielded items
        remain delivered. A normal exhausted stream has one result per input.
        """
        import httpx

        if self.closed:
            raise RuntimeError("client is closed")
        if type(ordered) is not bool:
            raise TypeError("ordered must be a boolean")
        valid, errors = prepare_inputs(program, inputs)
        ir = compile_program(program, optimize=self.optimize).to_dict()
        digest = ir_sha256(ir)
        response = None
        if valid:
            payload = {"protocol_version": PROTOCOL_VERSION, "ir": ir, "ir_sha256": digest,
                       "mode": self.mode,
                       "inputs": [{"index": index, "state": state} for index, state in valid]}
            content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._start()
            try:
                request = self._http.build_request("POST", self.base_url + "/v1/runs:batch", content=content)
                response = self._http.send(request, stream=True)
                if response.status_code != 200:
                    response.read()
                    code, message = "http_error", f"remote HTTP error {response.status_code}"
                    try:
                        data = _json(response.content)
                        error = _error(data.get("error")) if type(data) is dict else None
                        if error is not None:
                            code, message = error.code, error.message
                    except RemoteProtocolError:
                        pass
                    raise RemoteSubmissionError(code, message, http_status=response.status_code,
                                                retry_after=response.headers.get("Retry-After"))
                media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if media_type != "application/x-ndjson":
                    raise RemoteProtocolError("expected application/x-ndjson response")
            except BaseException as exc:
                if response is not None:
                    response.close()
                if isinstance(exc, httpx.RequestError):
                    raise RemoteTransportError("remote HTTP request failed") from exc
                raise
        stream = _RemoteStream(self, response, digest, {i for i, _ in valid}, errors, ordered)
        self._streams.add(stream)
        return stream


class _RemoteStream(ResultStream):
    def __init__(self, client, response, digest, expected, errors, ordered):
        super().__init__()
        self.client, self.response, self.digest = client, response, digest
        self.expected, self.seen = expected, set()
        self.local = iter(Result(i, "error", error=e) for i, e in sorted(errors.items()))
        self.ordered, self.index, self.buffer = ordered, 0, {}
        self.accepted = False
        self.frames = self._frames() if response is not None else iter(())

    def _frames(self):
        # Split bytes before decoding so malformed UTF-8 is rejected rather than
        # replaced by httpx's text decoder. Require newline-terminated NDJSON.
        pending = bytearray()
        for chunk in self.response.iter_bytes():
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                line = bytes(pending[:newline])
                del pending[:newline + 1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                try:
                    yield _json(line.decode("utf-8"))
                except UnicodeError:
                    raise RemoteProtocolError("response is not UTF-8") from None
        if pending:
            raise RemoteProtocolError("unterminated NDJSON frame")

    def _remote_result(self):
        import httpx

        try:
            for frame in self.frames:
                if type(frame) is not dict:
                    raise RemoteProtocolError("frame must be an object")
                kind = frame.get("type")
                if kind == "error":
                    if set(frame) != {"type", "error"}:
                        raise RemoteProtocolError("invalid error frame")
                    error = _error(frame["error"])
                    raise RemoteSubmissionError(error.code, error.message,
                                                http_status=self.response.status_code,
                                                retry_after=self.response.headers.get("Retry-After"))
                if not self.accepted:
                    if (set(frame) != {"type", "protocol_version", "ir_sha256"}
                            or kind != "accepted" or type(frame["protocol_version"]) is not int
                            or frame["protocol_version"] != PROTOCOL_VERSION
                            or frame["ir_sha256"] != self.digest):
                        raise RemoteProtocolError("missing or mismatched accepted frame")
                    self.accepted = True
                    continue
                if kind == "done":
                    if set(frame) != {"type"} or self.seen != self.expected:
                        raise RemoteProtocolError("done before all submitted inputs completed")
                    # Verify the response really ends. Do not accept extra results
                    # or an error after the completion marker.
                    end = object()
                    if next(self.frames, end) is not end:
                        raise RemoteProtocolError("frame after done")
                    raise StopIteration
                if kind != "result" or set(frame) != {"type", "index", "status", "value", "error"}:
                    raise RemoteProtocolError("invalid result frame")
                index = frame["index"]
                if type(index) is not int or index not in self.expected or index in self.seen:
                    raise RemoteProtocolError("unknown or duplicate result index")
                error = None if frame["error"] is None else _error(frame["error"])
                try:
                    result = Result(index, frame["status"], frame["value"], error)
                except ValueError as exc:
                    raise RemoteProtocolError(str(exc)) from None
                self.seen.add(index)
                return result
        except httpx.RequestError as exc:
            raise RemoteTransportError("remote result stream failed") from exc
        raise RemoteProtocolError("remote result stream ended without done")

    def _pull(self):
        try:
            return next(self.local)
        except StopIteration:
            if self.response is None:
                raise
        return self._remote_result()

    def _next_result(self):
        if not self.ordered:
            return self._pull()
        while self.index not in self.buffer:
            result = self._pull()
            self.buffer[result.index] = result
        result = self.buffer.pop(self.index)
        self.index += 1
        return result

    def _close(self):
        try:
            if self.response is not None:
                self.response.close()
        finally:
            self.frames = iter(())
            self.local = iter(())
            self.buffer.clear()
            self.client._streams.discard(self)
