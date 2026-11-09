"""Native nullable scalars, default compatibility and mixed runtime requests."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from annoscript._data import thaw
from annoscript._test_evaluation_http import FixtureLLMBackend, llm_calls
from annoscript.backends import InferenceContext, InferenceError, LLMAPIBackend
from annoscript.backends.base import normalize_call, validate_response
from annoscript.compiler import compile_program
from annoscript.ir import Program
from annoscript.anno import GenerateResponse
from annoscript.plan import ExecutionPlan
from annoscript.plan_lowering import lower_to_plan
from annoscript.runtime import AnnoRuntime, RunInput
from annoscript.server.releases import Releases
from annoscript.server.store import Store
from annoscript.server.worker import Worker
from annoscript.test_jobs_server import configurations, payload
from annoscript.test_runtime import FakeBackend


def optional_calls(anno):
    normal = anno.generate("same", min_length=1, max_length=24, max_num_tokens=64).text
    optional = anno.generate("same", min_length=1, max_length=24, max_num_tokens=64, nullable=True).text
    again = anno.generate("same", min_length=1, max_length=24, max_num_tokens=64, nullable=True).text
    return [normal, optional, again]


def completion(content):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 2}})


def fixture_handler(sent, repair=False):
    def handle(request):
        body = json.loads(request.content); sent.append(body)
        batched, calls = llm_calls(body)
        values = [None if call.arguments.get("nullable") else "normal" for _, call in calls]
        if batched:
            if repair:
                values = [[] if value is None else value for value in values]
            return completion(json.dumps({"answers": {f"q{i}": value for i, value in enumerate(values, 1)}}))
        return completion(json.dumps(values[0]) if calls[0][1].arguments.get("nullable") else values[0])
    return handle


class NullableGenerateTests(unittest.IsolatedAsyncioTestCase):
    def call(self, *, identity="internal", **options):
        return normalize_call(identity, "generate", ("标签",),
                              {"min_length": 1, "max_length": 24, "max_num_tokens": 64, **options},
                              InferenceContext("合成正文"))

    async def test_false_is_exact_default_and_true_binds_cache_and_examples(self):
        ctx = InferenceContext("合成正文")
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm")
        try:
            a, b, c = self.call(), self.call(nullable=False), self.call(nullable=True)
            self.assertEqual(a, b)
            self.assertNotIn("nullable", a.arguments)
            for batch in (False, True):
                pa = backend._prepare(ctx, [a], 64, batch=batch)
                pb = backend._prepare(ctx, [b], 64, batch=batch)
                pc = backend._prepare(ctx, [c], 64, batch=batch)
                self.assertEqual((pa.binding, pa.body), (pb.binding, pb.body))
                self.assertNotEqual(pa.binding, pc.binding)
            for invalid in (None, 0, 1, "true", [], {}):
                with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "nullable"):
                    self.call(nullable=invalid)
            for answer in (None, "评价", "null"):
                self.call(nullable=True, examples=[{"state": "示例", "answer": answer}])
            with self.assertRaises(ValueError):
                self.call(examples=[{"state": "示例", "answer": None}])
            validate_response(c, GenerateResponse(None))
            empty_allowed = self.call(nullable=True, min_length=0)
            validate_response(empty_allowed, GenerateResponse(""))
            validate_response(empty_allowed, GenerateResponse(None))
            validate_response(c, GenerateResponse(" \t"))  # Business constrain rejects whitespace separately.
            with self.assertRaisesRegex(InferenceError, "generated text must be a string and satisfy nonempty requirement"):
                validate_response(a, GenerateResponse(None))
        finally:
            await backend.aclose()

    async def test_single_repair_strict_json_scalars_and_null_diagnostics(self):
        ctx = InferenceContext("合成正文")
        for profile in ("legacy", "qwen_vllm"):
            for feedback in (None, "invalid response"):
                for value in (None, "null", "评价", '甲"\\\n😀'):
                    sent = []
                    def handle(request):
                        sent.append(json.loads(request.content)); return completion(json.dumps(value, ensure_ascii=False))
                    backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile,
                                            transport=httpx.MockTransport(handle))
                    try:
                        out = await backend.infer(ctx, self.call(nullable=True), repair_feedback=feedback)
                        self.assertEqual(out.response.text, value)
                        observed = out.metadata["advisory_lengths"]["internal"]
                        self.assertEqual(observed["is_null"], value is None)
                        self.assertEqual(observed["actual_codepoints"], [] if value is None else [len(value)])
                        self.assertEqual(sent[0]["max_tokens"], 64)
                        self.assertIn("JSON null", sent[0]["messages"][0]["content"])
                    finally:
                        await backend.aclose()
            for raw in ('[]', '["评价"]', '{}', '1', 'true', '""', '未加引号', '"截断', ''):
                backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile,
                                        transport=httpx.MockTransport(lambda r: completion(raw)))
                try:
                    with self.subTest(profile=profile, raw=raw), self.assertRaises(InferenceError):
                        await backend.infer(ctx, self.call(nullable=True))
                finally:
                    await backend.aclose()
        # Ordinary raw text "null" retains its literal value.
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                transport=httpx.MockTransport(lambda r: completion("null")))
        try:
            self.assertEqual((await backend.infer(ctx, self.call())).response.text, "null")
        finally:
            await backend.aclose()


    async def test_ir_runtime_reuse_and_repair_in_all_modes(self):
        for optimize in (False, True):
            program = Program.from_json(compile_program(optional_calls, optimize=optimize).to_json())
            plan = ExecutionPlan.from_json(lower_to_plan(program).to_json())
            for mode in ("strict", "eager", "speculative"):
                for repair in (False, True):
                    sent = []
                    backend = FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                                                transport=httpx.MockTransport(fixture_handler(sent, repair)))
                    async with AnnoRuntime(FakeBackend(), backend, trace=True) as runtime:
                        rows = [r async for r in await runtime.submit(plan, [RunInput(0, "正文")], mode=mode)]
                    self.assertEqual(rows[0].status, "success", rows[0])
                    self.assertEqual(rows[0].value, ["normal", None, None])
                    expected = 3 if repair and mode != "strict" else 2
                    self.assertEqual(sum(len(llm_calls(body)[1]) for body in sent), expected)
                    self.assertTrue(all(body["max_tokens"] == (256 if llm_calls(body)[0] else 64) for body in sent))


class DurableNullableTests(unittest.TestCase):
    def test_native_worker_persists_none_and_collect_never_infers(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "jobs.sqlite")
            releases = Releases(configurations(), "test")
            sent = []
            def factory(_):
                return FakeBackend(), FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                                                        transport=httpx.MockTransport(fixture_handler(sent)))
            job = store.submit("alice", "nullable", payload(("甲", "乙"), mode="speculative", program=optional_calls), releases)
            worker = Worker(store, releases, backend_factory=factory)
            async def run():
                while (claim := store.claim("worker", releases)) is not None:
                    await worker.execute(claim)
                await worker.aclose()
            asyncio.run(run()); before = len(sent)
            for _ in range(2):
                rows = store.results("alice", job["job_id"])["results"]
                self.assertEqual([r["value"] for r in rows], [["normal", None, None]] * 2)
                self.assertIsNone(store.claim("worker", releases))
            self.assertEqual(len(sent), before)


if __name__ == "__main__":
    unittest.main()
