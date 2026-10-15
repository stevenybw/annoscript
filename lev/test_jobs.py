"""Client-only durable job tests: deterministic HTTP, no execution imports."""

from copy import deepcopy
import gc
import json
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
import weakref

import httpx

from lev.client import LevClient, precondition, require
from lev.compiler import compile_program
from lev.job_protocol import cursor_for, cursor_seq, request_sha256, validate_page
from lev.jobs import (LevJobClient, RemoteCancellationError, RemoteJobPausedError,
                      RemoteProtocolError, RemoteSubmissionError, RemoteTransportError)


def generate(lev):
    return lev.generate(lev.state).text


def check(state):
    require(state != "invalid", message="invalid record")


@precondition(check)
def checked(lev):
    return lev.generate(lev.state).text


class Clock:
    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Wire:
    def __init__(self, *, values=None, reverse=False):
        self.requests, self.jobs, self.keys, self.sessions = [], {}, {}, {}
        self.values, self.reverse = values, reverse
        self.failures, self.page_transform = [], None
        self.lose_accept = False
        self.paused = False
        self.job_counter = 0

    def response(self, value, status=200, headers=None):
        return httpx.Response(status, json=value, headers=headers)

    def session(self, sid):
        return {"protocol_version": 1, "session_id": sid, "state": self.sessions[sid],
                "expires_at": 1300.0, "heartbeat_interval": 30.0}

    def __call__(self, request):
        self.requests.append(request)
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, BaseException):
                raise failure
            if failure is not None:
                return failure
        path = request.url.path
        if path == "/v1/sessions":
            sid = request.headers["Idempotency-Key"]
            self.sessions.setdefault(sid, "active")
            return self.response(self.session(sid), 201)
        if path.startswith("/v1/sessions/"):
            sid, op = path.rsplit("/", 1)[1].split(":")
            if op == "close":
                self.sessions[sid] = "closed"
                for jid in self.jobs:
                    self.jobs[jid]["job"]["state"] = "cancelled"
            return self.response(self.session(sid))
        if path == "/v1/jobs" and request.method == "POST":
            payload = json.loads(request.content)
            key = request.headers["Idempotency-Key"]
            if key not in self.keys:
                self.job_counter += 1
                jid = str(self.job_counter)
                items = payload["inputs"]
                values = self.values if self.values is not None else [v["state"] for v in items]
                rows = [{"index": item["index"], "status": "success", "value": value,
                         "error": None, "attempts": 1} for item, value in zip(items, values)]
                if self.reverse:
                    rows.reverse()
                for n, row in enumerate(rows, 1):
                    row["seq"] = n
                job = {"job_id": jid, "ir_sha256": payload["ir_sha256"], "release_id": "fixed-release",
                       "mode": payload["mode"], "input_indices": [i["index"] for i in items],
                       "state": "paused" if self.paused else "completed",
                       "counts": {"total": len(items), "completed": 0 if self.paused else len(rows),
                                  "success": 0 if self.paused else len(rows), "error": 0},
                       "pause_generation": 1 if self.paused else 0,
                       "pause_reason": {"code": "model_timeout", "message": "budget exhausted"} if self.paused else None,
                       "created_at": 1000.0, "updated_at": 1000.0}
                self.keys[key] = jid
                self.jobs[jid] = {"job": job, "rows": [] if self.paused else rows, "request": payload}
            job = self.jobs[self.keys[key]]["job"]
            if self.lose_accept:
                self.lose_accept = False
                raise httpx.ReadError("lost acceptance")
            return self.response({"protocol_version": 1, "job": job}, 202)
        if path == "/v1/jobs:lookup":
            job = self.jobs[self.keys[request.headers["Idempotency-Key"]]]["job"]
            return self.response({"protocol_version": 1, "job": job})
        jid = path.split("/")[3].split(":")[0]
        job = self.jobs[jid]["job"]
        if path.endswith(":cancel"):
            job["state"], job["pause_reason"] = "cancelled", None
            job["counts"].update(completed=job["counts"]["total"], error=job["counts"]["total"], success=0)
            return self.response({"protocol_version": 1, "job": job})
        if path.endswith("/results"):
            after = cursor_seq(request.url.params.get("cursor"), jid)
            # Small pages exercise cursors and ordered buffers across pages.
            rows = self.jobs[jid]["rows"][after:after + 1]
            last = after + len(rows)
            more = last < job["counts"]["completed"]
            page = {"protocol_version": 1, "job": deepcopy(job), "results": deepcopy(rows),
                    "next_cursor": cursor_for(jid, last), "has_more": more,
                    "done": job["state"] in ("completed", "cancelled", "failed") and not more}
            if self.page_transform:
                self.page_transform(page)
            return self.response(page)
        return self.response({"protocol_version": 1, "job": job})


class ClientTests(unittest.TestCase):
    def client(self, wire=None, **kwargs):
        wire = wire or Wire()
        client = LevClient.remote("http://test", transport=httpx.MockTransport(wire), **kwargs)
        clock = Clock()
        low = client._remote.session.client
        low.clock, low.wall_clock, low.sleep, low.random = clock, clock, clock.sleep, lambda: 1.0
        self.addCleanup(client.close)
        return client, wire, clock

    def test_empty_all_invalid_and_defects_do_not_contact_server(self):
        client, wire, _ = self.client()
        self.assertEqual(list(client.submit(checked, [])), [])
        result = list(client.submit(checked, ["invalid", float("nan")]))
        self.assertEqual([r.error.code for r in result], ["invalid_input"] * 2)
        with patch("lev._client_common._PRECONDITIONS", {checked: (lambda state: 1 / 0,)}):
            with self.assertRaises(ZeroDivisionError):
                client.submit(checked, ["a"])
        with patch("lev._remote_jobs.compile_program", side_effect=ValueError("compile")):
            with self.assertRaisesRegex(ValueError, "compile"):
                client.submit(checked, ["a"])
        self.assertEqual(wire.requests, [])

    def test_results_indices_order_and_json_values(self):
        for ordered in (False, True):
            client, wire, _ = self.client(Wire(values=[None, [1, None], {"中文": True}], reverse=True))
            result = list(client.submit(checked, ["a", "invalid", "b", "c"], ordered=ordered))
            self.assertEqual([r.index for r in result], [0, 1, 2, 3] if ordered else [1, 3, 2, 0])
            values = {r.index: r.value for r in result}
            self.assertIsNone(values[0])
            self.assertEqual(values[2], [1, None])
            self.assertEqual(values[3], {"中文": True})
            self.assertEqual([v["index"] for v in wire.jobs["1"]["request"]["inputs"]], [0, 2, 3])

    def test_lost_acceptance_uses_identical_request_and_one_compile(self):
        client, wire, clock = self.client()
        wire.lose_accept = True
        with patch("lev._remote_jobs.compile_program", wraps=compile_program) as compile:
            self.assertEqual(list(client.submit(generate, ["a"]))[0].value, "a")
            self.assertEqual(compile.call_count, 1)
        posts = [r for r in wire.requests if r.url.path == "/v1/jobs"]
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0].content, posts[1].content)
        self.assertEqual(posts[0].headers["Idempotency-Key"], posts[1].headers["Idempotency-Key"])
        self.assertEqual(wire.job_counter, 1)
        self.assertTrue(clock.sleeps)
        list(client.submit(generate, ["a"]))
        self.assertEqual(wire.job_counter, 2)

    def test_result_disconnect_does_not_duplicate_delivery(self):
        client, wire, _ = self.client(Wire(reverse=True))
        stream = client.submit(generate, ["a", "b", "c"])
        first = next(stream)
        wire.failures = [httpx.ReadError("disconnected")]
        result = [first, *list(stream)]
        self.assertEqual([r.index for r in result], [2, 1, 0])
        gets = [r for r in wire.requests if r.url.path.endswith("/results")]
        self.assertEqual(gets[1].url, gets[2].url)

    def test_auth_and_protocol_errors_do_not_retry(self):
        for response, exception in ((httpx.Response(401, json={"error": {"code": "auth", "message": "denied"}}), RemoteSubmissionError),
                                    (httpx.Response(200, content=b"{bad", headers={"Content-Type": "application/json"}), RemoteProtocolError)):
            wire = Wire()
            wire.failures = [response]
            with LevJobClient("http://test", transport=httpx.MockTransport(wire)) as client:
                with self.assertRaises(exception):
                    client.status("anything")
                self.assertEqual(len(wire.requests), 1)

    def test_rate_limit_retry_after_and_exhaustion(self):
        wire, clock = Wire(), Clock()
        wire.failures = [httpx.Response(429, json={"error": {"code": "limited", "message": "later"}}, headers={"Retry-After": "3"})] * 5
        with LevJobClient("http://test", transport=httpx.MockTransport(wire), _clock=clock,
                          _wall_clock=clock, _sleep=clock.sleep, _random=lambda: 0) as client:
            with self.assertRaises(RemoteSubmissionError) as caught:
                client.status("anything")
            self.assertEqual(caught.exception.http_status, 429)
            self.assertEqual(caught.exception.retry_after, "3")
            self.assertEqual(clock.sleeps, [3] * 4)
            self.assertEqual(len(wire.requests), 5)

    def test_timeout_retries_are_bounded(self):
        wire, clock = Wire(), Clock()
        wire.failures = [httpx.ReadTimeout("timeout")] * 5
        with LevJobClient("http://test", transport=httpx.MockTransport(wire), _clock=clock,
                          _sleep=clock.sleep, _random=lambda: 0) as client:
            with self.assertRaises(RemoteTransportError):
                client.status("anything")
            self.assertEqual(len(wire.requests), 5)

    def test_total_deadline_bounds_a_transport_ignoring_phase_timeout(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        def slow(request):
            started.set()
            release.wait(2)
            finished.set()
            raise httpx.ReadTimeout("late transport")
        with LevJobClient("http://test", transport=httpx.MockTransport(slow)) as client:
            try:
                with self.assertRaisesRegex(RemoteTransportError, "deadline"):
                    client._request("GET", "/v1/jobs/unknown", budget=0.03, attempts=1)
                self.assertTrue(started.is_set())
                self.assertFalse(finished.is_set())
            finally:
                release.set()
                self.assertTrue(finished.wait(1))

    def test_duplicate_index_across_pages_fails_after_partial_delivery(self):
        client, wire, _ = self.client()
        stream = client.submit(generate, ["a", "b"])
        self.assertEqual(next(stream).index, 0)
        wire.page_transform = lambda page: page["results"][0].update(index=0)
        with self.assertRaises(RemoteProtocolError):
            next(stream)

    def test_lost_acceptance_exhaustion_cancels_unknown_accepted_job(self):
        wire = Wire()
        def disconnect(request):
            response = wire(request)
            if request.url.path == "/v1/jobs":
                raise httpx.ReadError("response always lost")
            return response
        client = LevClient.remote("http://test", transport=httpx.MockTransport(disconnect))
        client._remote.session.client.sleep = lambda _: None
        with self.assertRaises(RemoteTransportError):
            client.submit(generate, ["a"])
        self.assertEqual(wire.job_counter, 1)
        self.assertTrue(client.closed)
        self.assertEqual(wire.jobs["1"]["job"]["state"], "cancelled")
        client.close()

    def test_duplicate_unknown_missing_and_changed_bindings_fail(self):
        transforms = [lambda p: p["results"][0].update(index=99),
                      lambda p: p.update(next_cursor=cursor_for(p["job"]["job_id"], 9)),
                      lambda p: p["job"].update(release_id="different"),
                      lambda p: p["job"].update(state="running"),
                      lambda p: p.update(done=True),
                      lambda p: p["results"].append(deepcopy(p["results"][0]))]
        for transform in transforms:
            client, wire, _ = self.client()
            stream = client.submit(generate, ["a", "b"])
            wire.page_transform = transform
            with self.assertRaises(RemoteProtocolError):
                list(stream)
            self.assertTrue(stream.closed)

    def test_pause_is_explicit_and_closes_submission(self):
        wire = Wire()
        wire.paused = True
        client, _, _ = self.client(wire)
        stream = client.submit(generate, ["a"])
        with self.assertRaises(RemoteJobPausedError):
            list(stream)
        self.assertEqual(wire.jobs["1"]["job"]["state"], "cancelled")

    def test_one_stream_close_then_client_close(self):
        client, wire, _ = self.client()
        first, second = client.submit(generate, ["a"]), client.submit(generate, ["b"])
        first.close()
        self.assertEqual(wire.jobs["1"]["job"]["state"], "cancelled")
        self.assertNotEqual(wire.jobs["2"]["job"]["state"], "cancelled")
        client.close()
        self.assertTrue(second.closed)
        self.assertEqual(set(wire.sessions.values()), {"closed"})
        client.close()

    def test_destructor_collects_shell_even_with_live_stream_and_heartbeat(self):
        wire = Wire()
        client = LevClient.remote("http://test", transport=httpx.MockTransport(wire))
        stream = client.submit(generate, ["a"])
        session, ref = client._remote.session, weakref.ref(client)
        del client
        gc.collect()
        self.assertIsNone(ref())
        self.assertTrue(session.closed)
        self.assertTrue(stream.closed)
        self.assertEqual(set(wire.sessions.values()), {"closed"})
        session.thread.join(1)
        self.assertFalse(session.thread.is_alive())

    def test_explicit_close_timeout_is_reported_and_business_exception_survives(self):
        client, wire, _ = self.client()
        stream = client.submit(generate, ["a"])
        with patch.object(client._remote.session.client, "_close_session", side_effect=httpx.ReadTimeout("timeout")):
            with self.assertRaises(RemoteCancellationError):
                client.close()
        self.assertTrue(stream.closed)
        client, wire, _ = self.client()
        with patch.object(client._remote, "close", side_effect=RemoteCancellationError("cleanup")):
            with self.assertLogs("lev.client", level="WARNING"):
                with self.assertRaisesRegex(ValueError, "business"):
                    with client:
                        raise ValueError("business")
        # The patched close deliberately bypassed real resource cleanup.
        client._remote.session.close(1)

    def test_close_low_level_client_does_not_cancel_job(self):
        wire = Wire()
        with LevJobClient("http://test", transport=httpx.MockTransport(wire)) as client:
            ir = compile_program(generate)
            job = client.submit_ir(ir, [{"index": 2, "state": "a"}], idempotency_key="stable")
        self.assertFalse(any(r.url.path.endswith(":cancel") for r in wire.requests))
        with LevJobClient("http://test", transport=httpx.MockTransport(wire)) as client:
            self.assertEqual(client.lookup("stable"), job)

    def test_input_object_order_is_part_of_request_identity(self):
        payload = {"protocol_version": 1, "ir_sha256": "hash", "mode": "eager",
                   "inputs": [{"index": 0, "state": {"a": 1, "b": 2}}]}
        other = deepcopy(payload)
        other["inputs"][0]["state"] = {"b": 2, "a": 1}
        self.assertNotEqual(request_sha256(payload), request_sha256(other))

    def test_fresh_process_remote_import_isolation_and_reconnect(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.startswith(("lev.runtime", "lev.backends", "lev.plan", "lev.server", "fastapi", "uvicorn", "xgrammar", "vllm")):
            raise AssertionError("forbidden import: " + name)
sys.meta_path.insert(0, Block())
from lev.test_jobs import Wire, checked, Clock
from lev.client import LevClient
import httpx
w = Wire(); w.lose_accept = True
with LevClient.remote("http://test", transport=httpx.MockTransport(w)) as c:
    c._remote.session.client.sleep = lambda _: None
    r = list(c.submit(checked, ["a", "invalid"], ordered=True))
    assert [v.index for v in r] == [0, 1]
    assert r[0].value == "a" and r[1].error.code == "invalid_input"
assert w.job_counter == 1
'''
        subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
