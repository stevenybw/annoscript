"""Metric persistence, explicit migration and real HTTP Backend/Runtime tests."""

import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import httpx

from annoscript._client_common import Result
from annoscript._test_evaluation_http import http_backends, llm_calls, llm_content
from annoscript.compiler import compile_program
from annoscript.metrics import validate_metrics
from annoscript.server.migrate import migrate
from annoscript.server.store import SCHEMA_V1, ServiceError, Store
from annoscript.server.worker import Worker
from annoscript.server.releases import Releases
from annoscript.test_jobs import generate
from annoscript.test_jobs_server import ExecutionTests, StoreTests, payload, configurations


def reused(anno):
    a = anno.generate("same request", max_length=512)
    b = anno.generate("same request", max_length=512)
    return [a.text, b.text]


def concurrent(anno):
    a = anno.generate("first", max_length=512)
    b = anno.generate("second", max_length=512)
    return [a.text, b.text]


class MetricStoreTests(StoreTests):
    # Only new methods are discovered in this subclass (the existing suite runs
    # independently); inheritance provides a small setup/helper harness.
    def test_metric_fencing_retry_unknown_and_once_only(self):
        job = self.submit(states=("a",))
        claim = self.claim()
        self.store.meter_attempt(claim, True)
        self.store.request_started(claim, "r1", "generation", 1)
        self.store.request_started(claim, "r1", "generation", 1)
        self.store.request_finished(claim, "r1", seconds=1, http_status=200, usage={"prompt_tokens": 10, "completion_tokens": 2})
        self.store.request_finished(claim, "r1", seconds=9, http_status=200, usage={"prompt_tokens": 100})
        self.store.request_started(claim, "uncertain", "generation", 1)
        self.clock.now += 61
        replacement = self.claim()
        self.assertFalse(self.store.request_finished(claim, "uncertain", seconds=1, http_status=200, usage={}))
        self.store.meter_attempt(replacement, True)
        self.store.request_started(replacement, "r2", "generation", 1)
        self.store.request_finished(replacement, "r2", seconds=2, http_status=200, usage={"prompt_tokens": 7, "completion_tokens": 3})
        self.store.finish(replacement, Result(0, "success", "a"), elapsed_seconds=3)
        value = self.store.metrics("alice", job["job_id"])
        validate_metrics(value, job["job_id"])
        g = value["summary"]["backends"]["generation"]
        self.assertEqual((g["requests"], g["unknown_requests"], g["input_tokens"]["observed"]), (3, 1, 17))
        self.assertIsNone(g["input_tokens"]["total"])
        self.assertIsNone(value["summary"]["attempts"]["seconds"]["total"])
        self.assertEqual(value["summary"]["attempts"]["count"], 2)
        self.assertEqual(Store(self.path, clock=self.clock).metrics("alice", job["job_id"]), value)

    def test_cancel_and_expiration_authorization_and_queue_epoch(self):
        self.store.monotonic = lambda: 10
        job = self.submit(states=("a",))
        self.store.monotonic = lambda: 13
        claim = self.claim()
        self.store.meter_attempt(claim, True)
        self.store.request_started(claim, "r", "decision", 4)
        self.store.cancel("alice", job["job_id"])
        self.assertFalse(self.store.request_finished(claim, "r", seconds=1, http_status=200, usage={}))
        value = self.store.metrics("alice", job["job_id"])
        self.assertEqual(value["summary"]["queue_wait_seconds"]["total"], 3)
        self.assertEqual(value["summary"]["backends"]["decision"]["unknown_requests"], 1)
        with self.assertRaises(ServiceError) as error:
            self.store.metrics("bob", job["job_id"])
        self.assertEqual(error.exception.status, 404)
        self.clock.now += 8 * 86400
        with self.assertRaises(ServiceError) as error:
            self.store.metrics("alice", job["job_id"])
        self.assertEqual(error.exception.status, 410)
        job2 = self.submit(states=("b",), key="new")
        self.store.clock_epoch = "different-boot"
        claim2 = self.claim()
        self.store.meter_attempt(claim2, True)
        self.store.finish(claim2, Result(0, "success", "b"), elapsed_seconds=1)
        self.assertIsNone(self.store.metrics("alice", job2["job_id"])["summary"]["queue_wait_seconds"]["total"])

    def legacy_copy(self):
        path = Path(self.folder.name, "legacy.sqlite")
        with closing(sqlite3.connect(path)) as old, closing(sqlite3.connect(self.path)) as new:
            old.executescript(SCHEMA_V1)
            for table in ("releases", "sessions", "jobs", "inputs", "attempts", "results", "resume_ops"):
                names = [r[1] for r in old.execute(f"PRAGMA table_info({table})")]
                rows = new.execute(f"SELECT {','.join(names)} FROM {table}").fetchall()
                old.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in names)})", rows)
            old.commit()
        return path

    def test_explicit_migration_preserves_binding_backup_and_history(self):
        job = self.submit(states=("a",))
        self.store.finish(self.claim(), Result(0, "success", "a"))
        legacy = self.legacy_copy()
        with self.assertRaisesRegex(RuntimeError, "schema 1"):
            Store(legacy)
        backup = Path(self.folder.name, "backup.sqlite")
        migrate(legacy, backup)
        upgraded = Store(legacy, clock=self.clock)
        self.assertEqual(upgraded.lookup("alice", "job"), job | {"state": "completed", "counts": {"total": 1, "completed": 1, "success": 1, "error": 0}})
        value = upgraded.metrics("alice", job["job_id"])
        self.assertFalse(value["summary"]["history_complete"])
        self.assertIsNone(value["summary"]["backends"]["generation"]["total_requests"])
        self.assertEqual(upgraded.results("alice", job["job_id"])["results"][0]["value"], "a")
        with closing(sqlite3.connect(backup)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM results").fetchone()[0], 1)

    def test_migration_rolls_back_and_does_not_replace_backup(self):
        legacy = self.legacy_copy()
        backup = Path(self.folder.name, "backup.sqlite")
        def fail(statement):
            raise RuntimeError("simulated failure")
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            migrate(legacy, backup, _after_statement=fail)
        with closing(sqlite3.connect(legacy)) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertNotIn("metrics_available", [r[1] for r in db.execute("PRAGMA table_info(jobs)")])
        with self.assertRaises(FileExistsError):
            migrate(legacy, backup)


class MetricExecutionTests(ExecutionTests):
    def test_runtime_reuse_is_one_physical_request(self):
        requests = []
        client = self.client()
        job = client.submit_ir(compile_program(reused, optimize=False), [{"index": 0, "state": "hello"}], idempotency_key="reuse")
        self.run_claim(lambda _: http_backends(requests=requests))
        self.assertEqual(len(requests), 1)
        self.assertEqual(client.metrics(job)["summary"]["backends"]["generation"]["total_requests"], 1)

    def test_concurrent_http_sums_are_not_attempt_wall_time(self):
        from annoscript.backends import DecisionAPIBackend
        from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
        started = []
        async def handle(request):
            started.append(request)
            while len(started) < 2:
                await asyncio.sleep(0)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}],
                                            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        def factory(_):
            transport = httpx.MockTransport(handle)
            return (DecisionAPIBackend("http://offline", "fixture", "fixture", transport=transport),
                    LLMAPIBackend("http://offline", "fixture", "fixture", transport=transport))
        # This test needs two independent physical HTTP requests, irrespective
        # of the default batching optimization covered by the car tests.
        config = configurations()
        config["test"]["runtime"] = {"max_llm_batch_calls": 1}
        configured = Releases(config, "test")
        self.releases.manifests.update(configured.manifests)
        self.releases.default = configured.default
        client = self.client()
        job = client.submit_ir(compile_program(concurrent), [{"index": 0, "state": "hello"}], idempotency_key="concurrent")
        ticks = iter((0., 0., 10., 10.))
        from annoscript.server.metrics import RequestMeter
        original = RequestMeter.__init__
        def meter_init(instance, *args, **kwargs):
            original(instance, *args, **kwargs)
            instance.clock = lambda: next(ticks)
        with patch.object(RequestMeter, "__init__", meter_init):
            self.run_claim(factory)
        summary = client.metrics(job)["summary"]
        self.assertEqual(summary["backends"]["generation"]["http_seconds"]["total"], 20)
        self.assertLess(summary["attempts"]["seconds"]["total"], 20)

    def test_mixed_batch_and_local_repair_meter_each_http_once(self):
        from annoscript.test_llm_batch import mixed_generation, answer, completion
        requests = []
        def fail_one(body):
            content = llm_content(body, answer)
            if llm_calls(body)[0]:
                value = json.loads(content)
                del value["answers"][next(iter(value["answers"]))]
                content = json.dumps(value)
            return completion(content)
        client = self.client()
        job = client.submit_ir(compile_program(mixed_generation), [{"index": 0, "state": "广州与北京"}],
                               idempotency_key="batch-repair-metrics")
        self.run_claim(lambda _: http_backends(requests=requests, override=fail_one))
        self.assertEqual(client.fetch_results(job)["results"][0]["status"], "success")
        self.assertEqual([len(llm_calls(r)[1]) for r in requests], [3, 1])
        g = client.metrics(job)["summary"]["backends"]["generation"]
        self.assertEqual((g["requests"], g["questions"], g["unknown_requests"]), (2, 4, 0))
        self.assertEqual((g["input_tokens"]["total"], g["output_tokens"]["total"]), (26, 10))


    def test_content_repair_budget_resets_only_with_new_execution_attempt(self):
        from annoscript.test_llm_batch import mixed_generation, answer, completion
        requests, repairs = [], []
        def response(body):
            content = llm_content(body, answer)
            if llm_calls(body)[0]:
                value = json.loads(content)
                del value["answers"][next(iter(value["answers"]))]
                return completion(json.dumps(value))
            repairs.append(body)
            if len(repairs) == 1:
                return httpx.Response(429, json={"usage": {"prompt_tokens": 4, "completion_tokens": 0}},
                                      headers={"Retry-After": "0"})
            return completion(content)
        client = self.client()
        job = client.submit_ir(compile_program(mixed_generation), [{"index": 0, "state": "广州与北京"}],
                               idempotency_key="repair-per-attempt")
        for _ in range(2):
            self.run_claim(lambda _: http_backends(requests=requests, override=response))
        self.assertEqual(client.fetch_results(job)["results"][0]["attempts"], 2)
        self.assertEqual(client.fetch_results(job)["results"][0]["status"], "success")
        self.assertEqual([len(llm_calls(r)[1]) for r in requests], [3, 1, 3, 1])
        metrics = client.metrics(job)["summary"]["backends"]["generation"]
        self.assertEqual((metrics["requests"], metrics["questions"], metrics["input_tokens"]["total"]), (4, 8, 43))

    def test_retry_usage_survives_pause_and_resume(self):
        client = self.client()
        job = client.submit_ir(compile_program(generate), [{"index": 0, "state": "hello"}], idempotency_key="retry-metrics")
        def fail(body):
            return httpx.Response(429, json={"usage": {"prompt_tokens": 4, "completion_tokens": 0}}, headers={"Retry-After": "0"})
        for _ in range(3):
            self.run_claim(lambda _: http_backends(override=fail))
        status = client.status(job)
        self.assertEqual(status["state"], "paused")
        self.assertEqual(client.metrics(job)["summary"]["backends"]["generation"]["input_tokens"]["total"], 12)
        client.resume(job, pause_generation=status["pause_generation"], idempotency_key="resume-metrics")
        self.run_claim(lambda _: http_backends())
        metrics = client.metrics(job)
        self.assertEqual(metrics["summary"]["attempts"]["count"], 4)
        self.assertEqual(metrics["summary"]["backends"]["generation"]["input_tokens"]["total"], 19)

    def test_timeout_and_missing_usage_are_not_zero(self):
        for name in ("timeout", "missing"):
            client = self.client()
            job = client.submit_ir(compile_program(generate), [{"index": 0, "state": "hello"}], idempotency_key=name)
            def response(body):
                if name == "timeout":
                    raise httpx.ReadTimeout("offline")
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]})
            self.run_claim(lambda _: http_backends(override=response))
            g = client.metrics(job)["summary"]["backends"]["generation"]
            self.assertIsNone(g["input_tokens"]["total"])
            self.assertEqual(g["input_tokens"]["unknown"], 1)
            self.assertEqual(g["unknown_requests"], int(name == "timeout"))
            client.cancel(job)


# Reuse setup helpers, not the inherited test cases themselves.
for cls, base in ((MetricStoreTests, StoreTests), (MetricExecutionTests, ExecutionTests)):
    for name in base.__dict__:
        if name.startswith("test_") and name not in cls.__dict__:
            setattr(cls, name, None)
del StoreTests, ExecutionTests, cls, base
