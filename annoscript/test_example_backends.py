"""Offline examples through real normalization, HTTP adapters and runtime."""

from copy import deepcopy
import json
import unittest

import httpx


from annoscript._data import thaw
from annoscript._test_evaluation_http import http_backends, llm_calls, decision_calls

from annoscript.backends import InferenceContext, InferenceError, DecisionAPIBackend
from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from annoscript.backends.base import normalize_call, validate_response, BatchResult
from annoscript.backends._bounded_grammar import grammar
from annoscript.compiler import compile_program
from annoscript.ir import Program
from annoscript.anno import Example, ExtractResponse, ChoiceResponse
from annoscript.plan import ExecutionPlan
from annoscript.plan_lowering import lower_to_plan
from annoscript.runtime import AnnoRuntime, RunInput
from annoscript.test_examples import REVIEW, LAUNCH, captured, constructed, different_examples
from annoscript.test_generation_policy import policy
from annoscript.test_llm_batch import completion


def call(operation, answer, *, identity="q", examples=True, **options):
    arguments = {"choice": ("分类", {"正面": "积极", "中性": "中立"}),
                 "noul": ("是事件吗",), "score": ("评分", ["评价", "资讯"]),
                 "generate": ("标题",), "extract": ("地点",),
                 "generate_list": ("标签",)}[operation]
    limits = ({"max_length": 80} if operation == "generate" else
              {"max_items": 2, "max_length_per_item": 80} if operation in ("extract", "generate_list") else {})
    if operation == "extract":
        limits["source_field"] = "text"
    if examples:
        limits["examples"] = [Example(LAUNCH, answer)]
    return normalize_call(identity, operation, arguments, {**limits, **options}, InferenceContext(REVIEW))


class ExampleValidationTests(unittest.TestCase):
    def test_all_six_operations_and_advisory_lengths(self):
        for operation, answer in [("choice", "中性"), ("noul", True), ("score", 1),
                                  ("generate", LAUNCH["text"]), ("generate_list", ["资讯"]),
                                  ("extract", ["广州"])]:
            with self.subTest(operation=operation):
                result = call(operation, answer)
                self.assertEqual(thaw(result.arguments)["examples"], [{"state": LAUNCH, "answer": answer}])
        call("generate", LAUNCH["text"], max_length=1)
        call("generate_list", ["资讯"], max_length_per_item=1)
        call("extract", ["广州"], max_length_per_item=1)

    def test_invalid_answers_shapes_and_source_bindings(self):
        cases = [
            ("choice", "不存在", {}), ("choice", {"choice": "中性", "confidence": .95}, {}),
            ("noul", .95, {}), ("score", True, {}), ("score", 2, {}), ("score", -1, {}),
            ("generate", None, {}), ("generate", "", {"min_length": 1}),
            ("generate_list", [None], {}), ("generate_list", [""], {"min_length_per_item": 1}),
            ("generate_list", ["资讯"], {"max_items": 0}),
            ("extract", [], {"min_items": 1}), ("extract", [""], {}),
            ("extract", ["条目"], {}), ("extract", ["广州上市"], {}),
            ("extract", ["广州"], {"source_field": "missing"}),
        ]
        for operation, answer, options in cases:
            with self.subTest(operation=operation, answer=answer), self.assertRaises(ValueError):
                call(operation, answer, **options)
        for examples in ({}, (), [{"state": LAUNCH}], [{"state": LAUNCH, "answer": [], "criteria": {}}],
                         [{"state": None, "answer": []}]):
            with self.subTest(examples=examples), self.assertRaises(ValueError):
                normalize_call("q", "extract", ("地点",),
                               {"examples": examples, "source_field": "text",
                                "max_items": 2, "max_length_per_item": 80}, InferenceContext(REVIEW))
        c = normalize_call("q", "extract", ("地点",),
                           {"examples": [Example(LAUNCH["text"], ["广州"])],
                            "max_items": 2, "max_length_per_item": 80}, InferenceContext(REVIEW["text"]))
        self.assertEqual(c.extraction_source, REVIEW["text"])

    def test_examples_cannot_expand_current_source_or_reuse_prepared_request(self):
        a = call("extract", ["广州"])
        b = call("extract", [])
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                profile="qwen_vllm", extract_mode="substring_chars")
        self.assertEqual(grammar([a], extract_mode="substring_chars"),
                         grammar([b], extract_mode="substring_chars"))
        with self.assertRaises(InferenceError):
            validate_response(a, ExtractResponse(["广州"]))
        validate_response(a, ExtractResponse(["条目"]))
        context = InferenceContext(REVIEW)
        pa = backend._prepare(context, [a], 32)
        pb = backend._prepare(context, [b], 32)
        self.assertEqual(pa.binding, backend._prepare(context, [a], 32).binding)
        self.assertNotEqual(pa.binding, pb.binding)
        self.assertNotEqual(pa.body, pb.body)
        self.assertNotIn("广", pa.payload["structured_outputs"]["grammar"])


class ExampleBackendTests(unittest.IsolatedAsyncioTestCase):


    async def test_missing_and_empty_examples_preserve_wire_for_every_operation(self):
        operations = [("choice", "中性"), ("noul", True), ("score", 1),
                      ("generate", "标题"), ("generate_list", []), ("extract", [])]
        sent = []
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture",
                                     transport=httpx.MockTransport(lambda request: (
                                         sent.append(json.loads(request.content)) or httpx.Response(200, json={"answers": {}}))))
        llm = LLMAPIBackend("http://offline.invalid", "fixture", "fixture")
        context = InferenceContext(REVIEW, "共用规则")
        try:
            for operation, answer in operations:
                original = call(operation, answer, examples=False)
                for empty in (None, []):
                    empty_call = normalize_call("q", operation, (), {**thaw(original.arguments), "examples": empty}, context)
                    self.assertEqual(empty_call, original)
                    if operation in ("generate", "generate_list", "extract"):
                        self.assertEqual(llm._single_payload(context, original, 100),
                                         llm._single_payload(context, empty_call, 100))
                    else:
                        for c in (original, empty_call):
                            with self.assertRaises(InferenceError):
                                await backend.infer_batch(context, [c])
                        self.assertEqual(sent[-2], sent[-1])
        finally:
            await backend.aclose()
            await llm.aclose()

    async def test_jev_criteria_examples_mixed_with_plain_question(self):
        sent = []
        def handle(request):
            body = json.loads(request.content)
            sent.append(body)
            answers = {}
            for identity, q in body["questions"].items():
                if q["type"] == "noul":
                    answer = {"noul": .9}
                elif q["type"] == "score":
                    answer = {"score": 1, "confidence": .9, "legend": {"0": "评价", "1": "资讯"},
                              "probabilities": {"0": .1, "1": .9}}
                else:
                    answer = {"choice": "中性", "confidence": .9, "probabilities": {"正面": .1, "中性": .9}}
                answers[identity] = {"type": q["type"], **answer}
            return httpx.Response(200, json={"answers": answers})
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture", transport=httpx.MockTransport(handle))
        calls = [call("choice", "中性", identity="choice"), call("noul", True, identity="noul"),
                 call("score", 1, identity="score"), call("noul", True, identity="plain", examples=False)]
        try:
            await backend.infer_batch(InferenceContext(REVIEW, "共用规则"), calls)
        finally:
            await backend.aclose()
        body = sent[0]
        self.assertEqual(body["state"], REVIEW)
        self.assertNotIn("examples", body)
        for c in calls:
            self.assertEqual(body["questions"][c.id]["instructions"], c.arguments["instructions"])
        self.assertEqual(body["questions"]["choice"]["criteria"],
                         {"正面": "积极", "中性": {"what": "中立", "examples": [LAUNCH]}})
        self.assertEqual(body["questions"]["noul"]["criteria"], {"true": {"examples": [LAUNCH]}})
        self.assertEqual(body["questions"]["score"]["criteria"],
                         ["评价", {"what": "资讯", "examples": [LAUNCH]}])
        self.assertNotIn("criteria", body["questions"]["plain"])

    async def test_qwen_sharing_single_batch_and_repair_round_trip(self):
        calls = [call("extract", ["广州"], identity="extract", max_num_tokens=32),
                 call("generate", LAUNCH["text"], identity="generate", max_num_tokens=32),
                 call("generate_list", ["资讯"], identity="list", max_num_tokens=32)]
        context = InferenceContext(REVIEW)
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm")
        bodies = [backend._batch_payload(context, calls, 256)]
        for c in calls:
            bodies.extend([backend._single_payload(context, c, 32),
                           backend._single_payload(context, c, 32, "格式错误")])
        for body in bodies:
            self.assertEqual(json.loads(body["messages"][-1]["content"]), REVIEW)
            self.assertEqual(body["messages"][0]["content"].count(
                '输入：' + json.dumps(LAUNCH, ensure_ascii=False, separators=(",", ":"))), 1)
            batched, decoded = llm_calls(body)
            expected = calls if batched else [next(c for c in calls if c.id == decoded[0][1].id)]
            for (_, reconstructed), original in zip(decoded, expected):
                a, b = thaw(reconstructed.arguments), thaw(original.arguments)
                a.pop("max_num_tokens")
                b.pop("max_num_tokens")
                self.assertEqual(a, b)
        self.assertEqual(thaw(calls[0].arguments)["examples"], [{"state": LAUNCH, "answer": ["广州"]}])

    async def test_examples_affect_split_bytes_context_checks_and_preserve_output_budget(self):
        context = InferenceContext(REVIEW)
        calls = [call("generate", LAUNCH["text"], identity=str(i), max_num_tokens=32) for i in range(3)]
        plain = [call("generate", LAUNCH["text"], identity=str(i), examples=False, max_num_tokens=32) for i in range(3)]
        def count(body):
            return 100 * len(llm_calls(body)[1]) if "\n\nExample\n" in body["messages"][0]["content"] else 10
        config = policy()
        config.update(context_window=500, default_max_tokens=32)
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                profile="qwen_vllm", request_policy=config, _token_counter=count)
        options = dict(max_calls=8, max_input_bytes=100000, max_output_tokens=400, default_tokens=32)
        self.assertEqual(len(backend.plan_batch(context, plain, **options).calls), 3)
        plan = backend.plan_batch(context, calls, **options)
        self.assertEqual(len(plan.calls), 2)
        self.assertEqual(plan.max_tokens, 192)
        with self.subTest("byte limit"):
            bounded = backend.plan_batch(context, calls, **{**options, "max_input_bytes": plan.input_bytes - 1})
            self.assertIsNone(bounded)
        sent = []
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                profile="qwen_vllm", request_policy=config,
                                _token_counter=lambda b: 480 if "\n\nExample\n" in b["messages"][0]["content"] else 10,
                                transport=httpx.MockTransport(lambda r: sent.append(r) or completion("标题")))
        try:
            with self.assertRaises(InferenceError) as caught:
                await backend.infer(context, calls[0])
            self.assertEqual(caught.exception.code, "context_length_exceeded")
            self.assertEqual(sent, [])
            await backend.infer(context, plain[0])
            self.assertEqual(json.loads(sent[0].content)["max_tokens"], 32)
        finally:
            await backend.aclose()

    async def test_runtime_snapshots_modes_and_different_examples_never_merge(self):
        class Decision:
            def __init__(self):
                self.calls = []
            async def infer_batch(self, context, calls):
                self.calls.extend(calls)
                return BatchResult({c.id: ChoiceResponse(c.arguments["examples"][0]["answer"], .9,
                                   {"正面": .5, "中性": .5}) for c in calls})
            async def aclose(self):
                pass
        class LLM:
            async def infer(self, context, call):
                from annoscript.backends import InferenceOutput
                self.examples = thaw(call.arguments)["examples"]
                return InferenceOutput(ExtractResponse([]))
            async def aclose(self):
                pass
        for optimize in (False, True):
            for mode in ("strict", "eager"):
                decision, llm = Decision(), LLM()
                async with AnnoRuntime(decision, llm) as runtime:
                    for function in (captured, constructed, different_examples):
                        program = Program.from_json(compile_program(function, optimize=optimize).to_json())
                        plan = lower_to_plan(program)
                        plan = ExecutionPlan.from_json(plan.to_json())
                        result, = [r async for r in await runtime.submit(plan, [RunInput(0, REVIEW)], mode=mode)]
                        self.assertEqual(result.status, "success", result.error)
                        if function is different_examples:
                            self.assertEqual(result.value, ["正面", "中性", "正面"])
                            self.assertEqual(len(decision.calls), 2)
                        else:
                            self.assertEqual(llm.examples[0]["answer"], ["广州"])
                            self.assertEqual(llm.examples[0]["state"]["text"], LAUNCH["text"])
