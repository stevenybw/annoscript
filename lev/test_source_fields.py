"""Field-bound extraction and simplified car requests; no live services."""
import json
import unittest
from dataclasses import replace

import httpx


from lev.backends import InferenceContext, InferenceError
from lev._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from lev.backends.base import normalize_call, validate_response
from lev.backends._bounded_grammar import grammar
from lev.backends._generation_policy import effective_budget
from lev.client import LevClient
from lev.compiler import compile_program, CompileError
from lev.lev import ExtractResponse
from lev.plan_lowering import lower_to_plan
from lev.runtime import LevRuntime, RunInput
from lev._test_evaluation_http import llm_content
from lev.test_generation_policy import policy
from lev.test_llm_batch import completion


def extract(context, **options):
    return normalize_call("e", "extract", ("抽取原文",),
                          {"max_items": 2, "max_length_per_item": 300, **options}, context)


def obsolete_source(lev):
    return lev.extract("原文", source="text", max_items=1, max_length_per_item=20).items


class SourceFieldTests(unittest.TestCase):
    def test_obsolete_api_fails_before_configuration_or_execution(self):
        with self.assertRaisesRegex(TypeError, "source_layout was removed"):
            LevClient.from_env(source_layout="inline")
        for optimize in (False, True):
            with self.assertRaisesRegex(CompileError, "source"):
                compile_program(obsolete_source, optimize=optimize)

    def test_resolution_snapshot_literal_keys_and_invalid_combinations(self):
        data = {"text": "甲乙", "other": "丙丁", "nested": {"text": "错误"}}
        context = InferenceContext(data)
        call = extract(context, source_field="other", max_num_tokens=100)
        data["other"] = "已改变"
        self.assertEqual(call.extraction_source, "丙丁")
        self.assertNotIn("source", call.arguments)
        self.assertEqual(effective_budget(call), 100)
        self.assertEqual(validate_response(call, ExtractResponse(["丙"])).items, ["丙"])
        with self.assertRaises(InferenceError):
            validate_response(call, ExtractResponse(["甲"]))
        self.assertEqual(extract(InferenceContext("甲乙")).extraction_source, "甲乙")
        self.assertEqual(extract(InferenceContext({"a.b": "literal"}), source_field="a.b").extraction_source, "literal")
        for state, options in [
            (data, {}), (data, {"source_field": "missing"}), (data, {"source_field": "nested"}),
            (data, {"source_field": "nested.text"}), ({"text": 1}, {"source_field": "text"}),
            ("甲乙", {"source_field": "text"}), ([], {}), (data, {"source_field": True}),
            (data, {"source": "甲乙"}),
        ]:
            with self.subTest(state=state, options=options), self.assertRaises(ValueError):
                extract(InferenceContext(state), **options)
        for budget in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                extract(context, source_field="text", max_num_tokens=budget)

    def test_source_isolation_in_grammar_and_prepared_cache(self):
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                profile="qwen_vllm", extract_mode="substring_chars")
        a, b = InferenceContext({"text": "甲乙"}), InferenceContext({"text": "丙丁"})
        ca, cb = extract(a, source_field="text"), extract(b, source_field="text")
        self.assertNotEqual(grammar([ca], extract_mode="substring_chars"),
                            grammar([cb], extract_mode="substring_chars"))
        pa, pb = backend._prepare(a, [ca], 100), backend._prepare(b, [cb], 100)
        self.assertNotEqual(pa.binding, pb.binding)
        self.assertNotEqual(pa.body, pb.body)
        with self.assertRaisesRegex(ValueError, "binding differs"):
            backend._prepare(b, [ca], 100)
        with self.assertRaises(ValueError):
            effective_budget(replace(ca, extraction_source=None))


class CarWireTests(unittest.IsolatedAsyncioTestCase):


    async def test_explicit_extract_budget_avoids_9760_but_long_input_fails_locally(self):
        context = InferenceContext({"text": "原文"})
        call = normalize_call("e", "extract", ("抽取",),
                              {"source_field": "text", "max_items": 16, "max_length_per_item": 300,
                               "max_num_tokens": 1440}, context)
        self.assertEqual(effective_budget(call), 1440)
        for input_tokens, expected_count in ((6752, 1), (6753, 0)):
            sent = []

            def handle(request):
                sent.append(request)
                return completion('{"items":["原文"]}')

            backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture",
                                    profile="qwen_vllm", request_policy=policy(),
                                    _token_counter=lambda p: input_tokens,
                                    transport=httpx.MockTransport(handle))
            try:
                if expected_count:
                    await backend.infer(context, call)
                else:
                    with self.assertRaises(InferenceError) as caught:
                        await backend.infer(context, call)
                    self.assertEqual(caught.exception.code, "context_length_exceeded")
                self.assertEqual(len(sent), expected_count)
            finally:
                await backend.aclose()
