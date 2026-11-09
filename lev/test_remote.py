"""Offline HTTP protocol and import-boundary tests (Python 3.10+)."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import httpx

from lev.client import precondition, require, system_prompt
from lev.compiler import CompileError, compile_program
from lev.remote import (RemoteLevClient, RemoteProtocolError, RemoteSubmissionError,
                        RemoteTransportError, ir_sha256)


def check(state):
    require(state != "invalid", message="invalid record")


@system_prompt("共享规则")
@precondition(check)
def checked(lev):
    return lev.generate(lev.state, max_length=512).text


def constant(lev):
    unused = 1
    return None


def unsupported(lev):
    return [x for x in lev.state]


def success(index, value=None):
    return {"type": "result", "index": index, "status": "success", "value": value, "error": None}


def accepted(payload):
    return {"type": "accepted", "protocol_version": 1, "ir_sha256": payload["ir_sha256"]}


def encode(frames):
    return b"".join((json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n").encode() for f in frames)


class Chunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.pulled = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.pulled += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


class RemoteTests(unittest.TestCase):
    def client(self, frames=None, **options):
        self.requests, self.responses = [], []

        def handle(request):
            payload = json.loads(request.content)
            self.requests.append((request, payload))
            values = frames(payload) if frames else [accepted(payload)] + [
                success(item["index"], item["state"]) for item in reversed(payload["inputs"])
            ] + [{"type": "done"}]
            stream = values if isinstance(values, Chunks) else Chunks([encode(values)])
            self.responses.append(stream)
            return httpx.Response(200, headers={"Content-Type": "application/x-ndjson"}, stream=stream)

        client = RemoteLevClient("http://offline.test/prefix/", transport=httpx.MockTransport(handle), **options)
        self.addCleanup(client.close)
        return client

    def test_ir_only_request_and_indexed_results(self):
        for ordered in (False, True):
            for mode in ("strict", "eager"):
                with self.subTest(ordered=ordered, mode=mode), self.client(mode=mode, auth_token="test-token") as client:
                    inputs = ["甲", "invalid", {"nested": [1, None]}, float("nan"), "乙"]
                    results = list(client.submit(checked, inputs, ordered=ordered))
                    self.assertEqual([r.index for r in results], [0, 1, 2, 3, 4] if ordered else [1, 3, 4, 2, 0])
                    by_index = {r.index: r for r in results}
                    for i in (1, 3):
                        self.assertEqual(by_index[i].error.code, "invalid_input")
                    for i in (0, 2, 4):
                        self.assertEqual(by_index[i].value, inputs[i])
                    request, payload = self.requests[0]
                    self.assertEqual(request.method, "POST")
                    self.assertEqual(str(request.url), "http://offline.test/prefix/v1/runs:batch")
                    self.assertEqual(request.headers["Authorization"], "Bearer test-token")
                    self.assertEqual(set(payload), {"protocol_version", "ir", "ir_sha256", "mode", "inputs"})
                    self.assertEqual(payload["mode"], mode)
                    self.assertEqual(payload["ir"], compile_program(checked).to_dict())
                    self.assertEqual(payload["ir_sha256"], ir_sha256(payload["ir"]))
                    self.assertEqual([item["index"] for item in payload["inputs"]], [0, 2, 4])
                    self.assertNotIn("invalid record", request.content.decode())
                    self.assertNotIn("test-token", request.content.decode())
                    self.assertTrue(self.responses[0].closed)

    def test_empty_invalid_only_and_checker_defect_make_no_http_calls(self):
        with self.client() as client:
            self.assertEqual(list(client.submit(checked, [])), [])
            self.assertEqual([r.error.code for r in client.submit(checked, ["invalid", 1, {1: "bad"}])],
                             ["invalid_input"] * 3)
            for bad in (lambda state: 42, lambda state: 1 / 0):
                with patch("lev._client_common._PRECONDITIONS", {checked: (bad,)}):
                    with self.assertRaises((TypeError, ZeroDivisionError)):
                        client.submit(checked, ["first", "second"])
            with self.assertRaises(CompileError):
                client.submit(unsupported, ["first"])
            self.assertEqual(self.requests, [])
            self.assertIsNone(client._http)

    def test_snapshot_and_optimize_switch(self):
        for optimize in (False, True):
            with self.client(optimize=optimize) as client:
                inputs = [{"items": ["original"]}]
                before = deepcopy(inputs)
                stream = client.submit(constant, inputs)
                inputs[0]["items"].append("changed")
                self.assertEqual(list(stream)[0].value, before[0])
                payload = self.requests[0][1]
                self.assertEqual(payload["inputs"][0]["state"], before[0])
                self.assertEqual(payload["ir"], compile_program(constant, optimize=optimize).to_dict())
        self.assertNotEqual(compile_program(constant, optimize=False).to_json(), compile_program(constant).to_json())

    def test_values_and_per_item_errors(self):
        def frames(payload):
            return [accepted(payload), success(2, [None, {"中文": 1}]),
                    {"type": "result", "index": 1, "status": "error", "value": None,
                     "error": {"code": "constraint_failed", "message": "failed"}},
                    success(0), {"type": "done"}]
        with self.client(frames) as client:
            results = list(client.submit(constant, ["a", "b", "c"], ordered=True))
            self.assertEqual(results[0].status, "success")
            self.assertIsNone(results[0].value)
            self.assertEqual(results[1].error.code, "constraint_failed")
            self.assertEqual(results[2].value, [None, {"中文": 1}])

    def test_protocol_rejects_mismatches_duplicates_missing_and_malformed_results(self):
        mutations = [
            lambda p: [],
            lambda p: [success(0), {"type": "done"}],
            lambda p: [{**accepted(p), "protocol_version": True}],
            lambda p: [{**accepted(p), "protocol_version": 2}],
            lambda p: [{**accepted(p), "ir_sha256": "wrong"}],
            lambda p: [accepted(p), accepted(p)],
            lambda p: [accepted(p), success(0), success(0), {"type": "done"}],
            lambda p: [accepted(p), success(4)],
            lambda p: [accepted(p), success(True)],
            lambda p: [accepted(p), {"type": "done"}],
            lambda p: [accepted(p), success(0)],
            lambda p: [accepted(p), {"type": "result", "index": 0, "status": "success"}],
            lambda p: [accepted(p), {**success(0), "status": "unknown"}],
            lambda p: [accepted(p), {**success(0), "error": {"code": "x", "message": "bad"}}],
            lambda p: [accepted(p), {**success(0), "status": "error"}],
            lambda p: [accepted(p), success(0), {"type": "done"}, success(0)],
            lambda p: [accepted(p), success(0), {"type": "done"}, None],
        ]
        for frames in mutations:
            for ordered in (False, True):
                with self.subTest(frames=mutations.index(frames), ordered=ordered), self.client(frames) as client:
                    stream = client.submit(constant, ["a"], ordered=ordered)
                    with self.assertRaises(RemoteProtocolError):
                        list(stream)
                    self.assertTrue(stream.closed)
                    self.assertTrue(self.responses[0].closed)
                    self.assertEqual(len(self.requests), 1)

    def test_json_and_utf8_errors_and_chunk_boundaries(self):
        invalid = [b"{}", b"\n", b"not json\n", b"\xff\n", b"[]\n",
                   b'{"type":"done","type":"done"}\n', b'{"x":NaN}\n',
                   b'{"x":1e999}\n']
        for data in invalid:
            with self.subTest(data=data), self.client(lambda p: Chunks([encode([accepted(p)]), data])) as client:
                with self.assertRaises(RemoteProtocolError):
                    list(client.submit(constant, ["a"]))
        def bytes_one_at_a_time(p):
            data = encode([accepted(p), success(0, "中文"), {"type": "done"}]).replace(b"\n", b"\r\n")
            return Chunks([data[i:i+1] for i in range(len(data))])
        with self.client(bytes_one_at_a_time) as client:
            self.assertEqual(list(client.submit(constant, ["a"]))[0].value, "中文")

    def test_http_rejection_and_submission_error_preserve_metadata(self):
        for status in (302, 401, 403, 429, 503):
            requests = []
            def handle(request):
                requests.append(request)
                return httpx.Response(status, headers={"Retry-After": "12", "Location": "http://elsewhere.test/"},
                                      json={"error": {"code": "rejected", "message": "try later"}})
            with RemoteLevClient("http://offline.test", transport=httpx.MockTransport(handle)) as client:
                with self.assertRaises(RemoteSubmissionError) as caught:
                    client.submit(constant, ["a"])
                self.assertEqual((caught.exception.code, caught.exception.http_status, caught.exception.retry_after),
                                 ("rejected", status, "12"))
                self.assertEqual(len(requests), 1)
        with self.client(lambda p: [accepted(p), success(0), {"type": "error", "error": {
            "code": "backend_unavailable", "message": "unavailable"}}]) as client:
            stream = client.submit(constant, ["a", "b"])
            self.assertEqual(next(stream).index, 0)
            with self.assertRaises(RemoteSubmissionError) as caught:
                next(stream)
            self.assertEqual(caught.exception.code, "backend_unavailable")
            self.assertTrue(stream.closed)

    def test_transport_failures_are_not_retried(self):
        for error_type in (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError):
            calls = []
            def handle(request):
                calls.append(request)
                raise error_type("injected")
            with RemoteLevClient("http://offline.test", transport=httpx.MockTransport(handle)) as client:
                with self.assertRaises(RemoteTransportError):
                    client.submit(constant, ["a"])
                self.assertEqual(len(calls), 1)
            with self.client(lambda p: Chunks([encode([accepted(p), success(0)]), error_type("injected")])) as client:
                stream = client.submit(constant, ["a", "b"])
                self.assertEqual(next(stream).index, 0)
                with self.assertRaises(RemoteTransportError):
                    next(stream)
                self.assertTrue(stream.closed)
                self.assertEqual(len(self.requests), 1)

    def test_response_headers_and_errors_before_acceptance(self):
        for media_type in (None, "application/json", "text/plain"):
            response = httpx.Response(200, headers={} if media_type is None else {"Content-Type": media_type},
                                      stream=Chunks([b'{}\n']))
            with RemoteLevClient("http://offline.test", transport=httpx.MockTransport(lambda r: response)) as client:
                with self.assertRaises(RemoteProtocolError):
                    client.submit(constant, ["a"])
                self.assertTrue(response.is_closed)
        with self.client(lambda p: [{"type": "error", "error": {"code": "unsupported_ir", "message": "version"}}]) as client:
            with self.assertRaises(RemoteSubmissionError) as caught:
                list(client.submit(constant, ["a"]))
            self.assertEqual(caught.exception.code, "unsupported_ir")
        response = httpx.Response(503, headers={"Retry-After": "10"}, content="<html>private debug details</html>")
        with RemoteLevClient("http://offline.test", transport=httpx.MockTransport(lambda r: response)) as client:
            with self.assertRaises(RemoteSubmissionError) as caught:
                client.submit(constant, ["a"])
            self.assertEqual(caught.exception.http_status, 503)
            self.assertEqual(caught.exception.retry_after, "10")
            self.assertNotIn("private", str(caught.exception))

    def test_close_is_lazy_idempotent_and_client_closes_all_streams(self):
        def frames(p):
            return Chunks([encode([accepted(p)]), encode([success(0)]),
                           encode([success(1)]), encode([{"type": "done"}])])
        with self.client(frames) as client:
            a = client.submit(constant, ["a", "b"])
            self.assertEqual(self.responses[0].pulled, 0)
            self.assertEqual(next(a).index, 0)
            self.assertEqual(self.responses[0].pulled, 2)
            a.close()
            a.close()
            self.assertEqual(self.responses[0].pulled, 2)
            b = client.submit(constant, ["a", "b"])
            client.close()
            self.assertTrue(b.closed)
            self.assertEqual(list(a), [])
            self.assertEqual(list(b), [])
            self.assertTrue(all(response.closed for response in self.responses))
            with self.assertRaises(RuntimeError):
                client.submit(constant, [])

    def test_cleanup_failure_still_closes_other_streams(self):
        client = self.client()
        streams = [client.submit(constant, ["a"]) for _ in range(2)]
        with patch.object(self.responses[0], "close", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                client.close()
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(self.responses[1].closed)
        self.assertTrue(client._http.is_closed)

    def test_input_and_configuration_types(self):
        for options in ({"mode": "other"}, {"optimize": 1}, {"timeout": 0}, {"timeout": True},
                        {"timeout": float("inf")}, {"auth_token": "a\nb"}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                RemoteLevClient("http://offline.test", **options)
        for url in ("file:///tmp/x", "http://u:p@host", "http://host/?token=x", "http://host/#x"):
            with self.assertRaises(ValueError):
                RemoteLevClient(url)
        with self.client() as client:
            with self.assertRaises(TypeError):
                client.submit(constant, ("a",))
            with self.assertRaises(TypeError):
                client.submit(constant, ["a"], ordered=1)


class ImportBoundaryTests(unittest.TestCase):
    def run_isolated(self, code):
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
