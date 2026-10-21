"""Public task names through the real wire, planner, runtime and durable jobs."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from lev._data import thaw
from lev.backends import InferenceContext, LLMAPIBackend, InferenceError
from lev.backends.base import normalize_call
from lev.compiler import compile_program
from lev.ir import Program
from lev.plan import ExecutionPlan
from lev.plan_lowering import lower_to_plan
from lev.runtime import LevRuntime, RunInput
from lev.server.releases import Releases
from lev.server.store import Store
from lev.server.worker import Worker
from lev.test_jobs_server import configurations, payload
from lev.test_runtime import FakeBackend
from lev._test_evaluation_http import FixtureLLMBackend, llm_calls


def named_pair(lev):
    first = lev.generate("same", max_length=10, max_num_tokens=32, field_name="first").text
    second = lev.generate("same", max_length=10, max_num_tokens=32, field_name="second").text
    again = lev.generate("same", max_length=10, max_num_tokens=32, field_name="first").text
    return [first, second, again]


def completion(content):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 2}})


def named_handler(sent, *, repair=False):
    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        batched, calls = llm_calls(body)
        if batched:
            answer = {call.arguments["field_name"]: call.arguments["field_name"] for _, call in calls}
            if repair:
                answer.pop("second")
            return completion(json.dumps({"answers": answer}))
        return completion(calls[0][1].arguments["field_name"])
    return handle


class FieldNameTests(unittest.IsolatedAsyncioTestCase):
    def make_call(self, operation="generate", name=None, identity="run1-q1", context=None, **options):
        context = context or InferenceContext({"text": 'q1 q10 甲"\\\n😀'})
        defaults = {"max_length": 10} if operation == "generate" else {"max_items": 2, "max_length_per_item": 10}
        if operation == "extract":
            defaults["source_field"] = "text"
        if name is not None:
            defaults["field_name"] = name
        return normalize_call(identity, operation, ("保留正文中的 q1 与 q10",), {**defaults, **options}, context)

    async def test_none_and_omitted_preserve_exact_payload_and_binding(self):
        ctx = InferenceContext({"text": "甲"})
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm")
        try:
            for op in ("generate", "generate_list", "extract"):
                absent = self.make_call(op, context=ctx)
                explicit = self.make_call(op, context=ctx, field_name=None)
                self.assertEqual(absent, explicit)
                self.assertNotIn("field_name", absent.arguments)
                a, b = backend._prepare(ctx, [absent], 128), backend._prepare(ctx, [explicit], 128)
                self.assertEqual((a.binding, a.body, a.payload), (b.binding, b.body, b.payload))
                other = self.make_call(op, identity="run1-q2", context=ctx)
                self.assertEqual(backend._batch_payload(ctx, [absent, other], 256),
                                 backend._batch_payload(ctx, [explicit, other], 256))
        finally:
            await backend.aclose()

    async def test_validation_and_collisions_never_send_http(self):
        sent = []
        ctx = InferenceContext({"text": "甲"})
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                transport=httpx.MockTransport(lambda r: sent.append(r)))
        try:
            for op in ("generate", "generate_list", "extract"):
                for value in ("", " \t\n", False, 1, [], {}):
                    with self.subTest(op=op, value=value), self.assertRaisesRegex(ValueError, "field_name"):
                        self.make_call(op, context=ctx, field_name=value)
            for names in (("same", "same"), ("q2", None)):
                calls = [self.make_call(name=n, identity=str(i), context=ctx) for i, n in enumerate(names)]
                with self.assertRaisesRegex(ValueError, "duplicate model-visible"):
                    await backend.infer_batch(ctx, calls, max_tokens=128)
                with self.assertRaisesRegex(ValueError, "duplicate model-visible"):
                    backend.plan_batch(ctx, calls, max_calls=8, max_input_bytes=100000,
                                       max_output_tokens=4096, default_tokens=1024)
            self.assertEqual(sent, [])
        finally:
            await backend.aclose()

    async def test_mixed_names_examples_grammar_and_reordered_answers(self):
        ctx = InferenceContext({"text": 'q1 q10 甲"\\\n😀'})
        strange = ' 地点"\\\n😀 '
        calls = [self.make_call(name="q10", context=ctx),
                 self.make_call("extract", name=strange, identity="run1-q2", context=ctx,
                                examples=[{"state": {"text": "广州"}, "answer": ["广州"]}]),
                 self.make_call("generate_list", identity="run1-q3", context=ctx)]
        body = {"answers": {"q3": [], strange: ["甲"], "q10": "标题"}}
        for profile in ("legacy", "qwen_vllm"):
            sent, events = [], []
            def handle(request):
                sent.append(json.loads(request.content))
                return completion(json.dumps(body, ensure_ascii=False))
            backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile,
                                    transport=httpx.MockTransport(handle), diagnostic_observer=events.append)
            try:
                result = await backend.infer_batch(ctx, calls, max_tokens=256)
                self.assertEqual(result.errors, {})
                self.assertEqual(result.responses[calls[0].id].text, "标题")
                self.assertEqual(result.responses[calls[1].id].items, ["甲"])
                self.assertEqual(result.responses[calls[2].id].items, [])
                prompt = sent[0]["messages"][0]["content"]
                self.assertIn('保留正文中的 q1 与 q10', prompt)
                self.assertIn('任务 ' + json.dumps(strange, ensure_ascii=False), prompt)
                self.assertIn('示例输出：' + json.dumps({"answers": {strange: ["广州"]}}, ensure_ascii=False, separators=(",", ":")), prompt)
                self.assertEqual(json.loads(sent[0]["messages"][1]["content"]), thaw(ctx.state))
                self.assertTrue(any(e.get("wire_call_ids") == {"q10": calls[0].id, strange: calls[1].id, "q3": calls[2].id}
                                    for e in events))
                if profile == "qwen_vllm":
                    root = sent[0]["structured_outputs"]["grammar"].splitlines()[0]
                    self.assertIn("q10", root)
                    self.assertNotIn("run1", root)
            finally:
                await backend.aclose()

    async def test_single_and_repair_only_rename_task_identity(self):
        ctx = InferenceContext({"text": "甲"})
        for op, text in (("generate", "标题"), ("extract", '{"items":["甲"]}'),
                         ("generate_list", '{"items":[]}')):
            sent, events = [], []
            def handle(request):
                sent.append(json.loads(request.content))
                return completion(text)
            backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                                    transport=httpx.MockTransport(handle), diagnostic_observer=events.append)
            try:
                plain = self.make_call(op, context=ctx, max_num_tokens=64)
                named = self.make_call(op, name="summary", context=ctx, max_num_tokens=64)
                for feedback in (None, "invalid answer"):
                    expected = backend._single_payload(ctx, plain, 64, feedback)
                    await backend.infer(ctx, named, repair_feedback=feedback)
                    actual = sent[-1]
                    actual["messages"][0]["content"] = actual["messages"][0]["content"].replace('任务 "summary"', '任务 "run1-q1"')
                    self.assertEqual(actual, expected)
                self.assertTrue(any(e.get("wire_call_ids") == {"summary": named.id} for e in events))
            finally:
                await backend.aclose()

    async def test_names_bind_preparation_and_remain_stable_in_split_subsets(self):
        ctx = InferenceContext({"text": "甲"})
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm")
        try:
            calls = [self.make_call(name=n, identity=str(i), context=ctx) for i, n in enumerate(("first", "second", "third"))]
            first = backend._prepare(ctx, calls[:2], 128, batch=True)
            second = backend._prepare(ctx, calls[1:], 128, batch=True)
            self.assertEqual(dict(first.evidence["wire_call_ids"]), {"first": "0", "second": "1"})
            self.assertEqual(dict(second.evidence["wire_call_ids"]), {"second": "1", "third": "2"})
            changed = replace(calls[0], arguments={**calls[0].arguments, "field_name": "different"})
            self.assertNotEqual(first.binding, backend._prepare(ctx, [changed, calls[1]], 128, batch=True).binding)
        finally:
            await backend.aclose()

    async def test_ir_runtime_reuse_and_repair_in_all_modes(self):
        for optimize in (False, True):
            program = Program.from_json(compile_program(named_pair, optimize=optimize).to_json())
            plan = ExecutionPlan.from_json(lower_to_plan(program).to_json())
            for mode in ("strict", "eager", "speculative"):
                for repair in (False, True):
                    sent = []
                    backend = FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                                                transport=httpx.MockTransport(named_handler(sent, repair=repair)))
                    async with LevRuntime(FakeBackend(), backend, trace=True, max_llm_batch_calls=16,
                                          max_llm_batch_output_tokens=4096) as runtime:
                        results = [r async for r in await runtime.submit(plan, [RunInput(0, "甲")], mode=mode)]
                    self.assertEqual(results[0].status, "success", results[0])
                    self.assertEqual(results[0].value, ["first", "second", "first"])
                    self.assertEqual(sum(len(llm_calls(b)[1]) for b in sent), 3 if repair and mode != "strict" else 2)
                    self.assertTrue(all(b["max_tokens"] == (192 if llm_calls(b)[0] else 32) for b in sent))
                    if repair and mode != "strict":
                        self.assertEqual(len(sent), 2)
                        self.assertIn('任务 "second"', sent[-1]["messages"][0]["content"])


class DurableFieldNameTests(unittest.TestCase):
    def test_names_roundtrip_through_worker_and_collect_does_not_infer(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "jobs.sqlite")
            releases = Releases(configurations(), "test")
            sent = []
            def factory(_):
                return FakeBackend(), FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                                                        transport=httpx.MockTransport(named_handler(sent)))
            request = payload(("甲", "乙"), mode="speculative", program=named_pair)
            job = store.submit("alice", "named", request, releases)
            worker = Worker(store, releases, backend_factory=factory)
            async def run():
                while (claim := store.claim("worker", releases)) is not None:
                    await worker.execute(claim)
            asyncio.run(run())
            before = len(sent)
            for _ in range(2):
                page = store.results("alice", job["job_id"])
                self.assertEqual([r["value"] for r in page["results"]], [["first", "second", "first"]] * 2)
                self.assertEqual(store.submit("alice", "named", request, releases)["job_id"], job["job_id"])
                self.assertIsNone(store.claim("worker", releases))
            self.assertEqual(len(sent), before)


if __name__ == "__main__":
    unittest.main()
