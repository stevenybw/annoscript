"""Compact wire aliases and strict values, independent of fixture serialization."""

import json
import unittest

import httpx

from lev.backends import InferenceContext, LLMAPIBackend, InferenceError
from lev.backends.base import normalize_call


class CompactBatchTests(unittest.IsolatedAsyncioTestCase):
    def setup_calls(self):
        context = InferenceContext({"text": '甲"\\\n😀', "other": "乙"})
        calls = [
            normalize_call("q2", "generate", ("标题",), {"min_length": 1, "max_length": 1}, context),
            normalize_call('run9-"\n😀', "extract", ("抽取",),
                           {"source_field": "text", "min_items": 1, "max_items": 1, "max_length_per_item": 1}, context),
            normalize_call("run9-q2", "generate_list", ("空列表",),
                           {"max_items": 0, "max_length_per_item": 0}, context),
        ]
        return context, calls

    async def invoke(self, profile, content, context, calls):
        requests, events = [], []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})
        backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile,
                                transport=httpx.MockTransport(handle), diagnostic_observer=events.append)
        try:
            result = await backend.infer_batch(context, calls, max_tokens=512)
            return result, requests, events
        finally:
            await backend.aclose()

    async def test_aliases_restore_original_ids_and_accept_reordered_answers(self):
        context, calls = self.setup_calls()
        for profile in ("legacy", "qwen_vllm"):
            content = json.dumps({"answers": {"q3": [], "q2": [context.state["text"]], "q1": "超出建议长度"}}, ensure_ascii=False)
            result, requests, events = await self.invoke(profile, content, context, calls)
            self.assertEqual(result.errors, {})
            self.assertEqual(set(result.responses), {call.id for call in calls})
            self.assertEqual(result.responses["q2"].text, "超出建议长度")
            self.assertEqual(result.responses[calls[1].id].items, [context.state["text"]])
            system = requests[0]["messages"][0]["content"]
            for index in (1, 2, 3):
                self.assertIn(f'任务 "q{index}"', system)
            self.assertNotIn("run9", system)
            self.assertEqual(json.loads(requests[0]["messages"][1]["content"]), dict(context.state))
            mapping = {"q1": "q2", "q2": calls[1].id, "q3": "run9-q2"}
            self.assertTrue(any(event.get("wire_call_ids") == mapping for event in events))
            self.assertTrue(any(event.get("batch_response_format") == "short_ids_values_v1" for event in events))

    async def test_old_wrappers_and_wrong_value_types_are_local_errors(self):
        context, calls = self.setup_calls()
        for profile in ("legacy", "qwen_vllm"):
            for value in ({"type": "generate", "text": "甲"}, {"text": "甲"}, [], None, 1, True):
                with self.subTest(profile=profile, value=value):
                    content = json.dumps({"answers": {"q1": value, "q2": ["甲"], "q3": []}})
                    result, _, _ = await self.invoke(profile, content, context, calls)
                    self.assertEqual(set(result.errors), {calls[0].id})
                    self.assertEqual(result.errors[calls[0].id].call_id, calls[0].id)
                    self.assertEqual(set(result.responses), {calls[1].id, calls[2].id})
            for value in ("甲", {"type": "extract", "items": ["甲"]}, [None], ["乙"], [""], ["甲", "甲"]):
                content = json.dumps({"answers": {"q1": "标题", "q2": value, "q3": []}})
                result, _, _ = await self.invoke(profile, content, context, calls)
                self.assertEqual(set(result.errors), {calls[1].id})

    async def test_invalid_envelope_is_not_guessed_or_partially_adopted(self):
        context, calls = self.setup_calls()
        bad = ['{"answers":{"q1":"a","q1":"b"}}', '{"answers":{},"answers":{}}',
               '{"answers":{"run9-q2":[]}}', '{"answers":{"q0":[]}}',
               '{"answers":{"q4":[]}}', '{"answers":[],"extra":0}', '[]', '{}']
        for profile in ("legacy", "qwen_vllm"):
            for content in bad:
                with self.subTest(profile=profile, content=content), self.assertRaises(InferenceError) as caught:
                    await self.invoke(profile, content, context, calls)
                self.assertEqual(caught.exception.code, "invalid_batch_output")

    async def test_subsets_restart_aliases_without_reusing_a_different_binding(self):
        context, calls = self.setup_calls()
        for profile in ("legacy", "qwen_vllm"):
            backend = LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile)
            try:
                first = backend._prepare(context, calls[:2], 512, batch=True)
                second = backend._prepare(context, calls[1:], 512, batch=True)
                self.assertNotEqual(first.binding, second.binding)
                self.assertEqual(dict(first.evidence["wire_call_ids"]), {"q1": calls[0].id, "q2": calls[1].id})
                self.assertEqual(dict(second.evidence["wire_call_ids"]), {"q1": calls[1].id, "q2": calls[2].id})
                result, _, _ = await self.invoke(profile, '{"answers":{"q1":["甲"],"q2":[]}}', context, calls[1:])
                self.assertEqual(set(result.responses), {calls[1].id, calls[2].id})
                self.assertEqual(result.errors, {})
            finally:
                await backend.aclose()
