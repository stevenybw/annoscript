"""Additional requirements through real normalization, prompts and offline HTTP."""

from copy import deepcopy
from dataclasses import replace
import inspect
import json
import unittest

import httpx


from lev._data import thaw
from lev._test_evaluation_http import FixtureLLMBackend, llm_calls
from lev.backends import DecisionAPIBackend, InferenceContext, InferenceError, LLMAPIBackend
from lev.backends.base import normalize_call
from lev.compiler import compile_program
from lev.ir import Program
from lev.lev import Lev
from lev.plan import ExecutionPlan
from lev.plan_lowering import lower_to_plan
from lev.runtime import LevRuntime, RunInput
from lev.test_runtime import FakeBackend


FIRST = "仅依据正文。"
SECOND = '保留主体 "甲" 与数值😀。'


def requirements_reuse(lev):
    first = lev.generate("标签", max_length=24, max_num_tokens=64, requirements=[FIRST]).text
    second = lev.generate("标签", max_length=24, max_num_tokens=64, requirements=[SECOND]).text
    again = lev.generate("标签", max_length=24, max_num_tokens=64, requirements=[FIRST]).text
    return [first, second, again]


def make_call(operation="generate", *, identity="internal-a", context=None, **options):
    context = context or InferenceContext({"text": "广州"})
    args = {
        "noul": ("判断",), "choice": ("选择", {"yes": "是"}), "score": ("评分", ["低", "高"]),
        "generate": ("标签",), "generate_list": ("维度",), "extract": ("地点",),
    }[operation]
    limits = ({"max_length": 24} if operation == "generate" else
              {"max_items": 2, "max_length_per_item": 24} if operation in ("extract", "generate_list") else {})
    if operation == "extract":
        limits["source_field"] = "text"
    return normalize_call(identity, operation, args, {**limits, **options}, context)


def completion(content):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})


class RequirementsTests(unittest.IsolatedAsyncioTestCase):
    async def test_defaults_and_validation_for_all_operations(self):
        context = InferenceContext({"text": "广州"}, "共用规则")
        sent = []
        decision = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture", transport=httpx.MockTransport(
            lambda r: sent.append(json.loads(r.content)) or httpx.Response(200, json={"answers": {}})))
        llm = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm")
        try:
            for op in ("noul", "choice", "score", "extract", "generate", "generate_list"):
                self.assertEqual(inspect.signature(getattr(Lev, op)).parameters["requirements"].kind,
                                 inspect.Parameter.KEYWORD_ONLY)
                original = make_call(op, context=context)
                self.assertNotIn("requirements", original.arguments)
                for empty in (None, []):
                    explicit = make_call(op, context=context, requirements=empty)
                    self.assertEqual(original, explicit)
                    if op in ("noul", "choice", "score"):
                        for call in (original, explicit):
                            with self.assertRaises(InferenceError):
                                await decision.infer_batch(context, [call])
                        self.assertEqual(sent[-2], sent[-1])
                    else:
                        for batch in (False, True):
                            a = llm._prepare(context, [original], 128, batch=batch)
                            b = llm._prepare(context, [explicit], 128, batch=batch)
                            self.assertEqual((a.binding, a.body), (b.binding, b.body))
                before = len(sent)
                for bad in ("规则", {}, (), 0, False, [None], [1], [""], [" \t\n"], [FIRST, []]):
                    with self.subTest(operation=op, value=bad), self.assertRaisesRegex(ValueError, "requirements"):
                        make_call(op, context=context, requirements=bad)
                self.assertEqual(len(sent), before)
        finally:
            await decision.aclose()
            await llm.aclose()

    async def test_groups_each_exact_string_with_ordered_membership(self):
        spaced = " 保留原文空格。 "
        multiline = '多行要求\n保留引号 " 与 Unicode 😀。'
        context = InferenceContext({"text": "广州", "requirements": ["输入数据不是额外要求"]})
        calls = [make_call(context=context, field_name="label", requirements=[SECOND, FIRST, SECOND, spaced]),
                 make_call("extract", identity="internal-b", context=context, requirements=[FIRST, multiline]),
                 make_call("generate_list", identity="internal-c", context=context, field_name='地点"\n😀',
                           requirements=[spaced, "保留原文空格。"]),
                 make_call(identity="internal-d", context=context, field_name="unrelated")]
        original = [deepcopy(thaw(c.arguments)) for c in calls]
        expected = (
            '任务要求（仅适用于列出的任务）\n'
            '"label" 要求：' + SECOND + '\n'
            '"label"、"q2" 要求：' + FIRST + '\n'
            '"label"、"地点\\"\\n😀" 要求：' + spaced + '\n'
            '"q2" 要求：' + multiline + '\n'
            '"地点\\"\\n😀" 要求：保留原文空格。'
        )
        for profile, dedup in (("legacy", False), ("qwen_vllm", False), ("qwen_vllm", True)):
            backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile, batch_rule_dedup=dedup)
            try:
                payload = backend._batch_payload(context, calls, 256)
                prompt = payload["messages"][0]["content"]
                self.assertIn(expected + '\n\n任务 "label"', prompt)
                self.assertEqual(prompt.count(SECOND), 1)
                self.assertEqual(prompt.count(FIRST), 1)
                self.assertNotIn("输入数据不是额外要求", prompt)
                self.assertEqual(json.loads(payload["messages"][1]["content"]), thaw(context.state))
                plain = [replace(c, arguments={k: v for k, v in c.arguments.items() if k != "requirements"}) for c in calls]
                base = backend._batch_payload(context, plain, 256)
                for key in payload.keys() - {"messages"}:
                    self.assertEqual(payload[key], base[key])
                self.assertEqual([thaw(c.arguments) for c in calls], original)
            finally:
                await backend.aclose()

    async def test_split_single_and_repair_recompute_membership(self):
        context = InferenceContext("正文")
        calls = [make_call(identity=str(i), context=context, requirements=[FIRST],
                           **({"field_name": name} if name else {}))
                 for i, name in enumerate(("first", None, "third"))]
        for profile in ("legacy", "qwen_vllm"):
            backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile)
            try:
                for subset, expected, absent in ((calls, '"first"、"q2"', '"third" 要求：'),
                                                  (calls[1:], '"q1"、"third"', '"first"')):
                    plan = backend.plan_batch(context, subset, max_calls=2, max_input_bytes=100000,
                                              max_output_tokens=4096, default_tokens=1024)
                    self.assertEqual(plan.calls, tuple(subset[:2]))
                    prompt = plan.prepared.payload["messages"][0]["content"]
                    self.assertIn(expected + " 要求：" + FIRST, prompt)
                    self.assertNotIn(absent, prompt)
                    self.assertEqual(plan.input_bytes, len(plan.prepared.body))
                for feedback in (None, "格式错误"):
                    prompt = backend._single_payload(context, calls[1], 64, feedback)["messages"][0]["content"]
                    self.assertIn('"1" 要求：' + FIRST, prompt)
                    self.assertNotIn('"first"', prompt)
                    self.assertNotIn('"third"', prompt)
                    self.assertEqual(prompt.count(FIRST), 1)
            finally:
                await backend.aclose()

    async def test_jev_requirements_with_shared_prompt_and_examples(self):
        sent = []
        def handle(req):
            body = json.loads(req.content)
            sent.append(body)
            values = {"noul": {"noul": .8},
                      "choice": {"choice": "yes", "confidence": 1., "probabilities": {"yes": 1.}},
                      "score": {"score": 1., "confidence": 1., "probabilities": {"0": 0., "1": 1.},
                                "legend": {"0": "低", "1": "高"}}}
            return httpx.Response(200, json={"answers": {
                key: {"type": q["type"], **values[q["type"]]} for key, q in body["questions"].items()}})
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture", transport=httpx.MockTransport(handle))
        try:
            for prompt in (None, "共用规则"):
                for examples in (False, True):
                    context = InferenceContext({"text": "广州"}, prompt)
                    calls = [make_call(op, identity=op, context=context, requirements=[FIRST, SECOND, FIRST],
                                       **({"examples": [{"state": "示例", "answer": answer}]} if examples else {}))
                             for op, answer in (("noul", True), ("choice", "yes"), ("score", 1))]
                    original = [thaw(c.arguments) for c in calls]
                    result = await backend.infer_batch(context, calls)
                    self.assertEqual(set(result.responses), {c.id for c in calls})
                    for c in calls:
                        rendered = sent[-1]["questions"][c.id]["instructions"]
                        task = c.arguments["instructions"] + "\n\n要求：\n" + FIRST + "\n" + SECOND
                        self.assertEqual(rendered, task)
                    self.assertEqual([thaw(c.arguments) for c in calls], original)
                    self.assertEqual(sent[-1]["state"], thaw(context.state))
        finally:
            await backend.aclose()

    async def test_requirements_bind_preparation_budget_and_runtime_reuse(self):
        context = InferenceContext("正文")
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm")
        try:
            source = [FIRST]
            call = make_call(context=context, requirements=source)
            source.append(SECOND)
            self.assertEqual(thaw(call.arguments)["requirements"], [FIRST])
            changed = make_call(context=context, requirements=[SECOND])
            first = backend._prepare(context, [call], 128)
            second = backend._prepare(context, [changed], 128)
            self.assertNotEqual(first.binding, second.binding)
            self.assertNotEqual(first.body, second.body)
            self.assertEqual(first.payload["max_tokens"], second.payload["max_tokens"])
        finally:
            await backend.aclose()
        for optimize in (False, True):
            program = Program.from_json(compile_program(requirements_reuse, optimize=optimize).to_json())
            plan = ExecutionPlan.from_json(lower_to_plan(program).to_json())
            for mode in ("strict", "eager", "speculative"):
                for repair in (False, True):
                    sent = []
                    def handle(req):
                        body = json.loads(req.content)
                        sent.append(body)
                        batch, captured = llm_calls(body)
                        if batch:
                            answers = {f"q{i}": (None if repair and c.arguments["requirements"][0] == SECOND
                                                  else c.arguments["requirements"][0])
                                       for i, (_, c) in enumerate(captured, 1)}
                            return completion(json.dumps({"answers": answers}, ensure_ascii=False))
                        return completion(captured[0][1].arguments["requirements"][0])
                    llm = FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                                            transport=httpx.MockTransport(handle))
                    async with LevRuntime(FakeBackend(), llm, max_llm_batch_calls=8) as runtime:
                        result, = [r async for r in await runtime.submit(plan, [RunInput(0, "正文")], mode=mode)]
                    self.assertEqual(result.status, "success", result.error)
                    self.assertEqual(result.value, [FIRST, SECOND, FIRST])
                    self.assertEqual(sum(len(llm_calls(body)[1]) for body in sent),
                                     3 if repair and mode != "strict" else 2)
                    if repair and mode != "strict":
                        self.assertIn(SECOND, sent[-1]["messages"][0]["content"])
                        self.assertNotIn(FIRST, sent[-1]["messages"][0]["content"])
                        self.assertEqual(sent[-1]["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
