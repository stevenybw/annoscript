"""Qwen wire/configuration contracts and real Runtime with offline HTTP responses."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from annoscript.backends import InferenceContext, InferenceError
from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from annoscript.backends._grammar import ITEMS_GRAMMAR, batch_grammar
from annoscript.backends._bounded_grammar import grammar
from annoscript._test_evaluation_http import llm_content, llm_calls
from annoscript.backends.base import normalize_call
from annoscript.backends.config import backends_from_env
from annoscript.client import AnnoClient


def completion(content, finish="stop", **message):
    return httpx.Response(200, json={"choices": [{"finish_reason": finish,
        "message": {"content": content, **message}}]})


class QwenBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_profiles_thinking_operations_and_token_budgets(self):
        context = InferenceContext({"text": "广州"}, "共同规则")
        for profile in ("legacy", "qwen_vllm"):
            for disabled in (True, False):
                for operation in ("generate", "generate_list", "extract"):
                    for budget in ((None, 7) if operation != "extract" else (None,)):
                        with self.subTest(profile=profile, disabled=disabled, operation=operation, budget=budget):
                            requests = []

                            def handle(request):
                                body = json.loads(request.content)
                                requests.append(body)
                                self.assertEqual(str(request.url), "https://llm.invalid/v1/chat/completions")
                                self.assertEqual(body["model"], "qwen-model-name-is-not-detection")
                                if disabled and profile == "legacy":
                                    self.assertEqual(body["thinking"], {"type": "disabled"})
                                else:
                                    self.assertNotIn("thinking", body)
                                if disabled and profile == "qwen_vllm":
                                    self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                                else:
                                    self.assertNotIn("chat_template_kwargs", body)
                                if operation == "generate":
                                    self.assertNotIn("response_format", body)
                                    self.assertEqual("structured_outputs" in body, profile=="qwen_vllm")
                                elif profile == "qwen_vllm":
                                    self.assertEqual(body["structured_outputs"], {"grammar": grammar([normalize_call("q", operation, ("正文",), options, context)])})
                                    self.assertNotIn("response_format", body)
                                else:
                                    self.assertEqual(body["response_format"], {"type": "json_object"})
                                    self.assertNotIn("structured_outputs", body)
                                if budget is None:
                                    self.assertEqual(body["max_tokens"], 1024)
                                else:
                                    self.assertEqual(body["max_tokens"], budget)
                                return completion("广州" if operation == "generate" else '{"items":["广州"]}')

                            options = {"source_field": "text"} if operation == "extract" else {"max_num_tokens": budget}
                            options.update({"max_length":512} if operation=="generate" else {"max_items":8,"max_length_per_item":58})
                            backend = LLMAPIBackend("https://llm.invalid/v1", "fixture", "qwen-model-name-is-not-detection",
                                profile=profile, disable_thinking=disabled, transport=httpx.MockTransport(handle))
                            try:
                                await backend.infer(context, normalize_call("q", operation, ("正文",), options, context))
                                self.assertEqual(len(requests), 1)
                            finally:
                                await backend.aclose()

    async def test_http_rejection_is_not_downgraded_or_retried(self):
        context = InferenceContext("正文")
        call = normalize_call("q", "generate_list", ("列举",), {'max_items': 8, 'max_length_per_item': 58}, context)
        for status in (400, 401, 403, 408, 422, 429, 500, 503):
            with self.subTest(status=status):
                requests = []

                def handle(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(status, text="private model response", headers={"Retry-After": "9"})

                backend = LLMAPIBackend("https://llm.invalid", "fixture", "model", profile="qwen_vllm",
                                        transport=httpx.MockTransport(handle))
                try:
                    with self.assertRaises(InferenceError) as caught:
                        await backend.infer(context, call)
                    self.assertEqual(caught.exception.metadata, {"http_status": status, "retry_after": "9"})
                    self.assertNotIn("private", str(caught.exception))
                    self.assertEqual(len(requests), 1)
                    self.assertIn("grammar", requests[0]["structured_outputs"])
                    self.assertNotIn("response_format", requests[0])
                finally:
                    await backend.aclose()

    async def test_invalid_output_and_truncation_do_not_trigger_repair(self):
        context = InferenceContext("广州")
        call = normalize_call("q", "extract", ("地点",), {"min_items": 1, 'max_items': 8, 'max_length_per_item': 58}, context)
        cases = [
            ("{}", "stop", {}), ('{"items":["北京"]}', "stop", {}),
            ('{"items":[]}', "stop", {}), ('{"items":[null]}', "stop", {}),
            ('{"items":[""]}', "stop", {}), ('{"items":["广州"],"extra":0}', "stop", {}),
            ('{"items":[', "stop", {}), ('{"items":["广州"]}', "length", {}),
            (None, "stop", {}), ('{"items":["广州"]}', "stop", {"refusal": "refused"}),
        ]
        for content, finish, message in cases:
            requests = []

            def handle(request):
                requests.append(request)
                return completion(content, finish, **message)

            backend = LLMAPIBackend("https://llm.invalid", "fixture", "model", profile="qwen_vllm",
                                    transport=httpx.MockTransport(handle))
            try:
                with self.subTest(content=content, finish=finish), self.assertRaises(InferenceError) as caught:
                    await backend.infer(context, call)
                self.assertEqual(caught.exception.code, "incomplete_generation" if finish == "length" else "invalid_response")
                self.assertEqual(len(requests), 1)
            finally:
                await backend.aclose()

    async def test_decoded_strings_are_preserved_and_count_unicode_characters(self):
        source = ' 中文😀 "引号" \\路径\n换行\t '
        context = InferenceContext(source)
        call = normalize_call("q", "extract", ("原文",), {"max_length_per_item": len(source), 'max_items': 8}, context)
        for ascii_only in (True, False):
            backend = LLMAPIBackend("https://llm.invalid", "fixture", "model", profile="qwen_vllm",
                transport=httpx.MockTransport(lambda _: completion(json.dumps({"items": [source]}, ensure_ascii=ascii_only))))
            try:
                response = await backend.infer(context, call)
                self.assertEqual(response.response.items, [source])
            finally:
                await backend.aclose()


class ConfigurationTests(unittest.TestCase):
    def test_invalid_configuration_fails_before_any_http(self):
        with patch("httpx.AsyncClient", side_effect=AssertionError("HTTP must not be constructed")):
            for profile in (None, "", "Qwen", 1, True, [], {}):
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    LLMAPIBackend("https://llm.invalid", "fixture", "model", profile=profile)
            for disabled in (None, "false", 0, 1):
                with self.subTest(disabled=disabled), self.assertRaises(ValueError):
                    LLMAPIBackend("https://llm.invalid", "fixture", "model", disable_thinking=disabled)

    def test_environment_precedence_and_public_client_forwarding(self):
        credentials = {f"ANNO_{prefix}_{suffix}": value for prefix in ("DECISION", "GENERATE")
                       for suffix, value in (("BASE_URL", "https://offline.invalid"),
                                             ("AUTH_TOKEN", "fixture"), ("MODEL", "model"))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "backend.env")
            path.write_text("ANNO_GENERATE_PROFILE=qwen_vllm\n")
            with patch.dict(os.environ, credentials, clear=True):
                self.assertEqual(backends_from_env()[1].profile, "legacy")
                self.assertEqual(backends_from_env(env_file=path)[1].profile, "qwen_vllm")
                with AnnoClient.from_env(env_file=path, generation_profile="legacy", disable_thinking=False) as client:
                    self.assertEqual(client.runtime.llm_backend.profile, "legacy")
                    self.assertFalse(client.runtime.llm_backend.disable_thinking)
                with patch.dict(os.environ, {"ANNO_GENERATE_PROFILE": "legacy"}):
                    self.assertEqual(backends_from_env(env_file=path)[1].profile, "legacy")
                    self.assertEqual(backends_from_env(env_file=path, generation_profile="qwen_vllm")[1].profile, "qwen_vllm")
                with patch.dict(os.environ, {"ANNO_GENERATE_PROFILE": "invalid"}):
                    with self.assertRaises(ValueError):
                        backends_from_env()
                    self.assertEqual(backends_from_env(generation_profile="legacy")[1].profile, "legacy")
                for profile in ("", 0, False):
                    with self.assertRaises(ValueError):
                        backends_from_env(generation_profile=profile)


class CarRuntimeTests(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
