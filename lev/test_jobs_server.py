"""Offline SQLite, API and real Runtime integration; no external services."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from lev._client_common import Result, ResultError
from lev.backends import BackendFatalError
from lev.backends.base import InferenceError
from lev.compiler import compile_program
from lev.job_protocol import cursor_for, validate_page
from lev.jobs import LevJobClient, RemoteSubmissionError
from lev.remote import ir_sha256
from lev.server.api import create_app
from lev.server.releases import Releases
from lev.server.store import ServiceError, Store
from lev.server.worker import Worker
from lev.test_jobs import Clock, generate
from lev.test_runtime import FakeBackend


def configurations(**policy):
    return {"test": {"decision": {"base_url": "http://offline.invalid", "model": "fixture", "revision": "fixed-1"},
                     "generation": {"base_url": "http://offline.invalid", "model": "fixture", "revision": "fixed-1"},
                     "policy": policy}}


def payload(states=("a", "b"), *, mode="eager", program=generate, **options):
    ir = compile_program(program).to_dict()
    return {"protocol_version": 1, "ir": ir, "ir_sha256": ir_sha256(ir), "mode": mode,
            "inputs": [{"index": i * 2, "state": value} for i, value in enumerate(states)], **options}


def input_keys(lev):
    result = []
    for key in lev.state:
        result = result + [key]
    return result


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name, "jobs.sqlite")
        self.clock = Clock()
        self.store = Store(self.path, clock=self.clock)
        self.releases = Releases(configurations(), "test")

    def submit(self, states=("a", "b"), key="job", **options):
        return self.store.submit("alice", key, payload(states, **options), self.releases)

    def claim(self):
        return self.store.claim("worker", self.releases)

    def test_repeated_and_concurrent_idempotency_and_principal_isolation(self):
        request = payload()
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs = list(pool.map(lambda _: self.store.submit("alice", "same", request, self.releases), range(12)))
        self.assertEqual(len({j["job_id"] for j in jobs}), 1)
        bob = self.store.submit("bob", "same", request, self.releases)
        self.assertNotEqual(bob["job_id"], jobs[0]["job_id"])
        with self.assertRaises(ServiceError) as caught:
            self.store.status("bob", jobs[0]["job_id"])
        self.assertEqual(caught.exception.status, 404)
        with self.assertRaises(ServiceError) as caught:
            self.store.submit("alice", "same", payload(("changed",)), self.releases)
        self.assertEqual(caught.exception.code, "idempotency_conflict")

    def test_new_default_does_not_rebind_repeated_submission(self):
        original = self.submit()
        config = configurations()
        config["test"]["generation"]["revision"] = "fixed-2"
        changed = Releases(config, "test")
        old = self.store.submit("alice", "job", payload(), changed)
        self.assertEqual(old["release_id"], original["release_id"])
        self.clock.now += 1
        fresh = self.store.submit("alice", "new", payload(), changed)
        self.assertNotEqual(fresh["release_id"], original["release_id"])
        self.store.claim("worker", changed)
        self.assertEqual(self.store.status("alice", original["job_id"])["pause_reason"]["code"], "release_unavailable")

    def test_profile_normalization_version_binding_and_unavailable_release(self):
        from lev.server.releases import build_sha256
        original = self.submit()
        config = configurations()
        config["test"]["generation"]["profile"] = "legacy"
        self.assertEqual(Releases(config, "test").default, self.releases.default)
        config["test"]["generation"]["profile"] = "qwen_vllm"
        changed = Releases(config, "test")
        self.assertNotEqual(changed.default, self.releases.default)
        replay = self.store.submit("alice", "job", payload(), changed)
        self.assertEqual(replay["release_id"], original["release_id"])
        self.assertIsNone(self.store.claim("worker", changed))
        self.assertEqual(self.store.status("alice", original["job_id"])["pause_reason"]["code"], "release_unavailable")
        config["test"]["disable_thinking"] = False
        self.assertNotEqual(Releases(config, "test").default, changed.default)
        # Grammar lives in source included by the existing build digest.
        before = build_sha256()
        read_bytes = Path.read_bytes
        def changed_grammar(path):
            content = read_bytes(path)
            return content + b"\n# changed grammar\n" if path.name == "_grammar.py" else content
        with patch.object(Path, "read_bytes", changed_grammar):
            self.assertNotEqual(before, build_sha256())

    def test_invalid_release_profile_rejected(self):
        for profile in (None, "", "unknown", 1, False, []):
            config = configurations()
            config["test"]["generation"]["profile"] = profile
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                Releases(config, "test")
        config = configurations()
        config["test"]["decision"]["profile"] = "qwen_vllm"
        with self.assertRaises(ValueError):
            Releases(config, "test")

    def test_batch_configuration_digest_validation_and_existing_job_binding(self):
        from lev.backends.base import LLM_BATCH_DEFAULTS
        original = self.submit()
        for name, default in LLM_BATCH_DEFAULTS.items():
            config = configurations()
            config["test"]["runtime"] = {name: default}
            self.assertEqual(Releases(config, "test").default, self.releases.default)
            config["test"]["runtime"][name] = 0 if name == "llm_batch_repair_attempts" else default + 1
            changed = Releases(config, "test")
            self.assertNotEqual(changed.default, self.releases.default)
            self.assertEqual(self.store.submit("alice", "job", payload(), changed)["release_id"], original["release_id"])
            for invalid in (True, -1, 1.5, None, 2 if name == "llm_batch_repair_attempts" else 0):
                config["test"]["runtime"][name] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    Releases(config, "test")

    def test_restart_reclaim_and_stale_result_fencing(self):
        job = self.submit(("a",))
        first = self.claim()
        self.clock.now += 61
        restarted = Store(self.path, clock=self.clock)
        second = restarted.claim("replacement", self.releases)
        self.assertEqual(second["attempt"], 2)
        self.assertNotEqual(first["token"], second["token"])
        self.assertFalse(self.store.finish(first, Result(0, "success", "stale")))
        self.assertTrue(restarted.finish(second, Result(0, "success", None)))
        page = restarted.results("alice", job["job_id"])
        self.assertTrue(page["done"])
        self.assertIsNone(page["results"][0]["value"])
        self.assertEqual(page["results"][0]["attempts"], 2)

    def test_crash_after_result_commit_never_reexecutes(self):
        job = self.submit(("a",))
        claim = self.claim()
        self.store.finish(claim, Result(0, "success", {"saved": True}))
        self.clock.now += 61
        self.assertIsNone(Store(self.path, clock=self.clock).claim("replacement", self.releases))
        self.assertEqual(self.store.lookup("alice", "job")["state"], "completed")

    def test_result_pages_completion_order_and_cursor_validation(self):
        job = self.submit()
        first, second = self.claim(), self.claim()
        empty = self.store.results("alice", job["job_id"])
        self.assertFalse(empty["done"])
        self.store.finish(second, Result(2, "success", [None, "中文"]))
        page = self.store.results("alice", job["job_id"], limit=1)
        validate_page(page, job["job_id"])
        self.assertEqual(page["results"][0]["index"], 2)
        self.assertFalse(page["done"])
        self.store.finish(first, Result(0, "error", error=ResultError("invalid_response", "invalid")))
        final = self.store.results("alice", job["job_id"], page["next_cursor"])
        validate_page(final, job["job_id"], page["next_cursor"])
        self.assertTrue(final["done"])
        self.assertEqual(final["results"][0]["index"], 0)
        self.assertEqual(final, self.store.results("alice", job["job_id"], page["next_cursor"]))
        for cursor in ("bad", cursor_for("another", 0), cursor_for(job["job_id"], 100)):
            with self.assertRaises(ServiceError):
                self.store.results("alice", job["job_id"], cursor)

    def test_retry_exhaustion_pause_and_idempotent_resume(self):
        job = self.submit()
        first, second = self.claim(), self.claim()
        self.store.finish(second, Result(2, "success", "kept"))
        error = {"code": "model_timeout", "message": "temporary"}
        for n in range(3):
            self.store.attempt_failed(first, error)
            if n < 2:
                first = self.claim()
        paused = self.store.status("alice", job["job_id"])
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["counts"]["completed"], 1)
        self.assertIsNone(self.claim())
        self.store.resume("alice", job["job_id"], "resume", paused["pause_generation"], self.releases)
        final = self.claim()
        self.assertEqual((final["index"], final["attempt"], final["round_attempt"]), (0, 4, 1))
        # Repeating a lost resume response cannot grant another retry budget.
        self.store.resume("alice", job["job_id"], "resume", paused["pause_generation"], self.releases)
        self.store.finish(final, Result(0, "success", "fixed"))
        self.assertEqual(self.store.status("alice", job["job_id"])["state"], "completed")
        with self.assertRaises(ServiceError):
            self.store.resume("alice", job["job_id"], "new-resume", paused["pause_generation"], self.releases)

    def test_abandoned_attempts_consume_budget(self):
        job = self.submit(("a",))
        for n in range(3):
            self.assertIsNotNone(self.claim())
            self.clock.now += 61
            self.store.maintain()
        self.assertIsNone(self.claim())
        self.assertEqual(self.store.status("alice", job["job_id"])["state"], "paused")

    def test_cancellation_preserves_results_and_rejects_late_worker(self):
        job = self.submit()
        first, second = self.claim(), self.claim()
        self.store.finish(first, Result(0, "success", "done"))
        cancelled = self.store.cancel("alice", job["job_id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertFalse(self.store.finish(second, Result(2, "success", "late")))
        rows = self.store.results("alice", job["job_id"])["results"]
        self.assertEqual(rows[0]["value"], "done")
        self.assertEqual(rows[1]["error"]["code"], "cancelled")
        self.assertEqual(cancelled, self.store.cancel("alice", job["job_id"]))

    def test_session_close_catches_unknown_job_and_prevents_late_submit(self):
        session = self.store.create_session("alice", "session")
        self.assertEqual(session, self.store.create_session("alice", "session"))
        job = self.submit(session_id=session["session_id"])
        claim = self.claim()
        self.store.close_session("alice", session["session_id"])
        self.assertEqual(self.store.lookup("alice", "job")["state"], "cancelled")
        self.assertFalse(self.store.renew(claim))
        with self.assertRaises(ServiceError):
            self.submit(key="late", session_id=session["session_id"])
        with self.assertRaises(ServiceError):
            self.store.heartbeat_session("alice", session["session_id"])
        self.assertEqual(self.store.create_session("alice", "session")["state"], "closed")

    def test_session_expiry_cancels_even_after_process_restart(self):
        session = self.store.create_session("alice", "session")
        job = self.submit(session_id=session["session_id"])
        self.clock.now += 200
        self.store.heartbeat_session("alice", session["session_id"])
        self.clock.now += 200
        self.assertEqual(self.store.status("alice", job["job_id"])["state"], "queued")
        self.clock.now += 101
        restarted = Store(self.path, clock=self.clock)
        self.assertIsNone(restarted.claim("worker", self.releases))
        rows = restarted.results("alice", job["job_id"])["results"]
        self.assertEqual({row["error"]["code"] for row in rows}, {"session_expired"})

    def test_close_submit_race_leaves_no_runnable_orphan(self):
        session = self.store.create_session("alice", "session")
        def submit():
            try:
                return self.submit(session_id=session["session_id"])
            except ServiceError as error:
                self.assertEqual(error.code, "session_closed")
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(submit)
            b = pool.submit(self.store.close_session, "alice", session["session_id"])
            a.result(), b.result()
        self.assertIsNone(self.claim())

    def test_retention_tombstone_and_paused_job_retention(self):
        job = self.submit()
        self.store.cancel("alice", job["job_id"])
        other = self.submit(("c",), key="paused")
        self.store.attempt_failed(self.claim(), {"code": "authentication_failed", "message": "bad config"}, pause=True)
        self.clock.now += 8 * 86400
        for operation in (lambda: self.store.status("alice", job["job_id"]),
                          lambda: self.submit(), lambda: self.store.lookup("alice", "job")):
            with self.assertRaises(ServiceError) as error:
                operation()
            self.assertEqual(error.exception.status, 410)
        self.assertEqual(self.store.status("alice", other["job_id"])["state"], "paused")

    def test_schema_and_input_structure_rejected(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA user_version=99")
        with self.assertRaises(RuntimeError):
            Store(self.path)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA user_version=2")
        bad = payload()
        bad["inputs"][1]["index"] = 0
        with self.assertRaises(ValueError):
            self.store.submit("alice", "bad", bad, self.releases)

    def test_concurrent_database_initialization_is_atomic(self):
        path = Path(self.folder.name, "new.sqlite")
        with ThreadPoolExecutor(max_workers=6) as pool:
            stores = list(pool.map(lambda _: Store(path, clock=self.clock), range(6)))
        self.assertEqual(len(stores), 6)
        self.assertEqual(stores[0].create_session("alice", "key"), stores[-1].create_session("alice", "key"))

    def test_initialization_recovers_from_transient_journal_busy(self):
        connect = sqlite3.connect

        class BusyOnce(sqlite3.Connection):
            def execute(self, statement, *args, **kwargs):
                if statement == "PRAGMA journal_mode=WAL" and not getattr(self, "injected", False):
                    self.injected = True
                    error = sqlite3.OperationalError("database is locked")
                    error.sqlite_errorcode = sqlite3.SQLITE_BUSY | (1 << 8)
                    raise error
                return super().execute(statement, *args, **kwargs)

        with patch("lev.server.store.sqlite3.connect",
                   side_effect=lambda *a, **kw: connect(*a, factory=BusyOnce, **kw)):
            store = Store(Path(self.folder.name, "busy.sqlite"), clock=self.clock)
            first = store.create_session("alice", "stable-key")
            self.assertEqual(first, store.create_session("alice", "stable-key"))

    def test_initialization_preserves_non_busy_database_error(self):
        connect = sqlite3.connect
        error = sqlite3.OperationalError("injected storage error")
        error.sqlite_errorcode = sqlite3.SQLITE_IOERR
        attempts = []

        class Broken(sqlite3.Connection):
            def execute(self, statement, *args, **kwargs):
                if statement == "PRAGMA journal_mode=WAL":
                    attempts.append(statement)
                    raise error
                return super().execute(statement, *args, **kwargs)

        with patch("lev.server.store.sqlite3.connect",
                   side_effect=lambda *a, **kw: connect(*a, factory=Broken, **kw)):
            with self.assertRaises(sqlite3.OperationalError) as caught:
                Store(Path(self.folder.name, "broken.sqlite"), clock=self.clock)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(attempts), 1)

    def test_input_key_order_conflicts_even_when_values_match(self):
        self.submit(({"a": 1, "b": 2},))
        with self.assertRaises(ServiceError) as error:
            self.submit(({"b": 2, "a": 1},))
        self.assertEqual(error.exception.code, "idempotency_conflict")

    def test_foreign_session_and_foreign_resume_are_rejected(self):
        session = self.store.create_session("bob", "session")
        with self.assertRaises(ServiceError) as error:
            self.submit(session_id=session["session_id"])
        self.assertEqual(error.exception.status, 404)
        job = self.submit(("a",))
        self.store.attempt_failed(self.claim(), {"code": "model_timeout", "message": "timeout"}, pause=True)
        with self.assertRaises(ServiceError) as error:
            self.store.resume("bob", job["job_id"], "resume", 1, self.releases)
        self.assertEqual(error.exception.status, 404)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.clock = Clock()
        self.store = Store(Path(self.folder.name, "jobs.sqlite"), clock=self.clock)
        self.releases = Releases(configurations(), "test")
        self.http = TestClient(create_app(self.store, self.releases, {"test-token": "alice"}))
        self.addCleanup(self.http.close)

    def client(self):
        def handle(request):
            return self.http.request(request.method, str(request.url), content=request.content, headers=request.headers)
        client = LevJobClient("http://testserver", "test-token", transport=httpx.MockTransport(handle))
        self.addCleanup(client.close)
        return client

    def run_claim(self, factory=None):
        if factory is None:
            def factory(manifest):
                backend = FakeBackend()
                return backend, backend
        worker = Worker(self.store, self.releases, backend_factory=factory, random_source=lambda: 0)
        claim = self.store.claim("worker", self.releases)
        self.assertIsNotNone(claim)
        asyncio.run(worker.execute(claim))
        return claim

    def test_http_ir_deserialization_runtime_and_indexed_results(self):
        client = self.client()
        ir = compile_program(generate)
        for mode in ("eager", "strict"):
            job = client.submit_ir(ir, [{"index": 7, "state": "hello"}], idempotency_key=mode, mode=mode)
            self.run_claim()
            page = client.fetch_results(job)
            self.assertTrue(page["done"])
            self.assertEqual((page["results"][0]["index"], page["results"][0]["value"]), (7, "hello"))
            self.assertEqual(client.lookup(mode)["job_id"], job["job_id"])

    def test_input_object_iteration_order_survives_http_journal_and_runtime(self):
        client = self.client()
        job = client.submit_ir(compile_program(input_keys), [
            {"index": 0, "state": {"a": 1, "b": 2}},
            {"index": 1, "state": {"b": 2, "a": 1}},
        ], idempotency_key="ordered-keys")
        self.run_claim()
        self.run_claim()
        self.assertEqual([r["value"] for r in client.fetch_results(job)["results"]], [["a", "b"], ["b", "a"]])

    def test_http_auth_bad_json_version_digest_and_limits(self):
        response = self.http.get("/v1/jobs/unknown")
        self.assertEqual(response.status_code, 401)
        headers = {"Authorization": "Bearer test-token", "Content-Type": "application/json", "Idempotency-Key": "bad"}
        for content in (b'{"duplicate":1,"duplicate":2}', b'{"x":NaN}', b'invalid'):
            self.assertEqual(self.http.post("/v1/jobs", content=content, headers=headers).status_code, 400)
        for field, value in (("protocol_version", 2), ("ir_sha256", "wrong")):
            request = payload()
            request[field] = value
            self.assertEqual(self.http.post("/v1/jobs", json=request, headers=headers).status_code, 400)
        oversized = payload(["a"] * 501)
        self.assertEqual(self.http.post("/v1/jobs", json=oversized, headers=headers).status_code, 413)

    def test_transient_failures_pause_and_can_be_resumed(self):
        client = self.client()
        job = client.submit_ir(compile_program(generate), [{"index": 0, "state": "hello"}], idempotency_key="retry")
        class Failing(FakeBackend):
            async def infer(self, *args):
                raise InferenceError("limited", code="model_http_error", metadata={"http_status": 429, "retry_after": "2"})
        for n in range(3):
            self.run_claim(lambda _: (Failing(), Failing()))
            self.clock.now += 2
        paused = client.status(job)
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(client.fetch_results(job)["results"], [])
        client.resume(job, pause_generation=paused["pause_generation"], idempotency_key="resume")
        self.run_claim()
        self.assertEqual(client.fetch_results(job)["results"][0]["attempts"], 4)

    def test_real_http_backend_retry_metadata_reaches_journal(self):
        from lev.backends.llm import LLMAPIBackend
        client = self.client()
        job = client.submit_ir(compile_program(generate), [{"index": 0, "state": "hello"}], idempotency_key="headers")
        def factory(manifest):
            llm = LLMAPIBackend("http://offline.invalid", "fixture-token", "fixture-model", profile="qwen_vllm",
                transport=httpx.MockTransport(lambda request: httpx.Response(429, json={}, headers={"Retry-After": "7"})))
            return FakeBackend(), llm
        self.run_claim(factory)
        self.assertIsNone(self.store.claim("worker", self.releases))
        with closing(sqlite3.connect(self.store.path)) as db:
            data = json.loads(db.execute("SELECT metadata FROM attempts WHERE job_id=?", (job["job_id"],)).fetchone()[0])
        self.assertEqual(data, {"http_status": 429, "retry_after": "7"})
        self.clock.now += 7
        self.run_claim()
        self.assertTrue(client.fetch_results(job)["done"])

    def test_worker_uses_bound_profile_despite_new_default_and_environment(self):
        from lev.backends.llm import LLMAPIBackend
        from lev.server.worker import backends
        config = configurations()
        config["test"]["generation"]["profile"] = "qwen_vllm"
        qwen = Releases(config, "test")
        job = self.store.submit("alice", "qwen", payload(("hello",)), qwen)
        # Both releases remain available; the server's new default is legacy.
        self.releases.manifests.update(qwen.manifests)
        bodies = []
        def handle(request):
            body = json.loads(request.content)
            bodies.append(body)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]})
        def create(*args, **kwargs):
            return LLMAPIBackend(*args, **kwargs, transport=httpx.MockTransport(handle))
        with patch.dict("os.environ", {"LEV_DECISION_AUTH_TOKEN": "fixture", "LEV_GENERATE_AUTH_TOKEN": "fixture",
                                       "LEV_GENERATE_PROFILE": "legacy"}), \
                patch("lev.backends.llm.LLMAPIBackend", side_effect=create):
            claim = self.run_claim(backends)
        self.assertEqual(claim["release_id"], job["release_id"])
        self.assertEqual(bodies[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("thinking", bodies[0])
        self.assertEqual(len(bodies), 1)
        self.assertTrue(self.store.results("alice", job["job_id"])["done"])

    def test_cancelled_attempt_cannot_start_another_model_call(self):
        from lev.server.worker import ObservedBackend
        backend = FakeBackend()
        observed = ObservedBackend(backend, [], lambda: False)
        async def run():
            with self.assertRaises(asyncio.CancelledError):
                await observed.infer(None, None)
            with self.assertRaises(asyncio.CancelledError):
                await observed.infer_batch(None, None)
        asyncio.run(run())
        self.assertEqual(backend.calls, [])

    def test_auth_failure_pauses_immediately_and_invalid_response_is_terminal(self):
        client = self.client()
        class Auth(FakeBackend):
            async def infer(self, *args):
                raise BackendFatalError("bad token", code="authentication_failed")
        job = client.submit_ir(compile_program(generate), [{"index": 0, "state": "a"}], idempotency_key="auth")
        self.run_claim(lambda _: (Auth(), Auth()))
        self.assertEqual(client.status(job)["state"], "paused")
        client.cancel(job)
        class Invalid(FakeBackend):
            async def infer(self, *args):
                raise InferenceError("invalid output", code="invalid_response")
        job = client.submit_ir(compile_program(generate), [{"index": 0, "state": "a"}], idempotency_key="invalid")
        self.run_claim(lambda _: (Invalid(), Invalid()))
        result = client.fetch_results(job)["results"][0]
        self.assertEqual((result["error"]["code"], result["attempts"]), ("invalid_response", 1))


if __name__ == "__main__":
    unittest.main()
