"""Protocol and contract checks use HTTPX transports, never real credentials."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from lev.backends import DecisionAPIBackend, LLMAPIBackend, InferenceContext, InferenceError, BackendFatalError
from lev.backends.base import normalize_call, validate_response
from lev.backends.config import backends_from_env
from lev.lev import ChoiceResponse, GenerateResponse, ExtractResponse, GenerateListResponse


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_decision_mixed_payload_and_id_mapping(self):
        context = InferenceContext({"text": "中文"}, "共同规则")
        calls = [normalize_call("q1", "noul", ("是否",), {}, context),
                 normalize_call("q2", "choice", ("选项", {"a": "甲", "b": "乙"}), {}, context),
                 normalize_call("q3", "score", ("程度", ["低", "高"]), {}, context)]
        async def handle(request):
            self.assertEqual(str(request.url), "https://decision.invalid/full/path")
            self.assertEqual(request.headers["Authorization"], "Bearer secret")
            body = json.loads(request.content)
            self.assertEqual(body["state"], {"text": "中文"})
            self.assertEqual(list(body["questions"]), ["q1", "q2", "q3"])
            self.assertNotIn("criteria", body["questions"]["q1"])
            self.assertTrue(all("共同规则" in q["instructions"] for q in body["questions"].values()))
            return httpx.Response(200, json={"answers": {
                "q3": {"type": "score", "score": 0.7, "confidence": 0.8, "probabilities": {"0": 0.3, "1": 0.7}, "legend": {"0": "低", "1": "高"}},
                "q1": {"type": "noul", "noul": 0.8},
                "q2": {"type": "choice", "choice": "b", "confidence": 0.9, "probabilities": {"a": 0.1, "b": 0.9}},
            }, "usage": {"input_tokens": 20, "output_tokens": 4}})
        backend = DecisionAPIBackend("https://decision.invalid/full/path", "secret", "model", transport=httpx.MockTransport(handle))
        try:
            result = await backend.infer_batch(context, calls)
            self.assertEqual(result.responses["q2"].choice, "b")
            self.assertEqual(result.responses["q3"].legend, {0: "低", 1: "高"})
            self.assertEqual(result.metadata["usage"], {"input_tokens": 20, "output_tokens": 4})
            client = backend._client
            await backend.infer_batch(context, calls)
            self.assertIs(backend._client, client)
        finally:
            await backend.aclose()

    async def test_llm_mapping_limits_extraction_source_and_thinking(self):
        context = InferenceContext({"text": "原文甲", "other": "独立来源广州"}, "共享政策")
        for operation, args, options, answer in (
            ("generate", ("简述",), {"min_length": 1, "max_length": 2, "max_num_tokens": 16}, "中文"),
            ("generate_list", ("列出",), {"min_items": 1, "max_items": 8, "max_length_per_item": 2}, '{"items":["中文"]}'),
            ("extract", ("抽取",), {"source_field": "other", "min_items": 1, "max_items": 8, "max_length_per_item": 58}, '{"items":["广州"]}'),
        ):
            with self.subTest(operation=operation):
                async def handle(request):
                    body = json.loads(request.content)
                    self.assertEqual(str(request.url), "https://llm.invalid/v1/chat/completions")
                    self.assertEqual(body["thinking"], {"type": "disabled"})
                    self.assertTrue(body["messages"][0]["content"].startswith("共享政策\n"))
                    payload = json.loads(body["messages"][-1]["content"])
                    self.assertEqual(payload, {"text": "原文甲", "other": "独立来源广州"})
                    if operation == "extract":
                        self.assertIn('输入的顶层字段 "other"', body["messages"][0]["content"])
                    if operation == "generate":
                        self.assertNotIn("response_format", body)
                        self.assertEqual(body["max_tokens"], 16)
                    else:
                        self.assertEqual(body["response_format"], {"type": "json_object"})
                        self.assertGreater(body["max_tokens"], 0)
                    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer, "reasoning_content": "never record this"}}], "usage": {"completion_tokens": 4}})
                backend = LLMAPIBackend("https://llm.invalid/v1", "secret", "model", transport=httpx.MockTransport(handle))
                try:
                    result = await backend.infer(context, normalize_call("q", operation, args, options, context))
                    self.assertNotIn("reasoning", json.dumps(result.metadata))
                finally:
                    await backend.aclose()

    async def test_http_failures_are_sanitized_and_not_retried(self):
        context = InferenceContext("x")
        call = normalize_call("q", "generate", ("task",), {'max_length': 512}, context)
        for status, error in ((401, BackendFatalError), (403, BackendFatalError), (429, InferenceError), (500, InferenceError)):
            count = 0
            async def handle(request):
                nonlocal count
                count += 1
                return httpx.Response(status, text="credential secret, body must not escape")
            backend = LLMAPIBackend("https://llm.invalid", "secret", "model", transport=httpx.MockTransport(handle))
            try:
                with self.assertRaises(error) as caught:
                    await backend.infer(context, call)
                self.assertNotIn("secret", str(caught.exception))
                self.assertEqual(count, 1)
            finally:
                await backend.aclose()

    async def test_bad_llm_results_never_become_success(self):
        context = InferenceContext("广州")
        call = normalize_call("q", "extract", ("地点",), {'max_items': 8, 'max_length_per_item': 58}, context)
        for message, finish in (("{}", "stop"), ('{"items":["北京"]}', "stop"), ('{"items":[""]}', "stop"), ('{"items":[null]}', "stop"), ('{"items":[]}', "length"), ("```json\n{}\n```", "stop"), (None, "stop")):
            backend = LLMAPIBackend("https://llm.invalid", "secret", "model", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {"content": message}}]})))
            try:
                with self.assertRaises(InferenceError):
                    await backend.infer(context, call)
            finally:
                await backend.aclose()

    async def test_timeout_cancellation_and_model_default_thinking(self):
        context = InferenceContext("x")
        call = normalize_call("q", "generate", ("x",), {'max_length': 512}, context)
        cancelled = asyncio.Event()
        async def hang(request):
            self.assertNotIn("thinking", json.loads(request.content))
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        backend = LLMAPIBackend("https://llm.invalid", "secret", "model", disable_thinking=False, timeout=0.01, transport=httpx.MockTransport(hang))
        try:
            with self.assertRaises(InferenceError) as caught:
                await backend.infer(context, call)
            self.assertEqual(caught.exception.code, "model_timeout")
            self.assertTrue(cancelled.is_set())
        finally:
            await backend.aclose()

    async def test_missing_decision_answers_and_wrong_types(self):
        context = InferenceContext("x")
        call = normalize_call("q", "choice", ("x", {"a": "A"}), {}, context)
        for answers in ({}, {"q": {"type": "choice", "choice": "b", "confidence": 0.8, "probabilities": {"a": 1}}}, {"q": {"type": "noul", "noul": 0.5}}):
            backend = DecisionAPIBackend("https://decision.invalid", "secret", "model", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": answers})))
            try:
                with self.assertRaises(InferenceError):
                    await backend.infer_batch(context, [call])
            finally:
                await backend.aclose()


class ContractTests(unittest.TestCase):
    def test_input_validation_and_payload_immutability(self):
        context = InferenceContext({"a": [1]})
        with self.assertRaises(TypeError):
            context.state["b"] = 2
        for operation, args, options in (
            ("generate", (1,), {}), ("generate", ("x",), {"max_num_tokens": 0}),
            ("generate", ("x",), {"min_length": 3, "max_length": 2}),
            ("extract", ("x",), {}), ("extract", ("x",), {"source_field": "text", "max_length_per_item": 0}),
            ("choice", ("x", {}), {}), ("score", ("x", []), {}),
        ):
            with self.subTest(operation=operation, options=options), self.assertRaises((ValueError, TypeError)):
                normalize_call("q", operation, args, options, context)

    def test_response_contracts(self):
        context = InferenceContext("原文")
        cases = [
            ("choice", ("x", {"a": "A"}), {}, ChoiceResponse("a", float("nan"), {"a": 1})),
            ("generate", ("x",), {"min_length":1,"max_length": 1}, GenerateResponse("")),
            ("generate_list", ("x",), {"min_items": 1,"max_items":2,"max_length_per_item":4}, GenerateListResponse([])),
            ("extract", ("x",), {"max_items":2,"max_length_per_item":4}, ExtractResponse(["原 文"])),
        ]
        for op, args, kwargs, response in cases:
            with self.assertRaises(InferenceError):
                validate_response(normalize_call("q", op, args, kwargs, context), response)

    def test_explicit_environment_precedence_and_no_shell_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env"
            path.write_text('\n'.join(f'export LEV_{prefix}_{suffix}="{value}"' for prefix in ("DECISION", "GENERATE") for suffix, value in (("BASE_URL", "https://example.invalid/v1"), ("AUTH_TOKEN", "$(never-execute)"), ("MODEL", "file-model"))))
            with patch.dict(os.environ, {"LEV_GENERATE_MODEL": "env-model"}, clear=True):
                decision, llm = backends_from_env(env_file=path)
                self.assertEqual(decision.model, "file-model")
                self.assertEqual(llm.model, "env-model")
                self.assertEqual(llm._token, "$(never-execute)")
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "LEV_DECISION_BASE_URL"):
                backends_from_env()


if __name__ == "__main__":
    unittest.main()
