"""Assert actual model messages independently of the fixed-response fixtures."""

import json
import unittest
from copy import deepcopy

from annoscript._data import thaw
from annoscript.backends import InferenceContext, LLMAPIBackend
from annoscript.backends.base import normalize_call
from annoscript.backends._bounded_grammar import grammar
from annoscript.anno import Example


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class PromptLayoutTests(unittest.TestCase):
    def backend(self, profile):
        return LLMAPIBackend("http://offline.invalid", "fixture", "fixture", profile=profile)

    def test_user_is_exact_input_for_all_profiles_and_request_modes(self):
        states = ['正文"\\\n😀é', '{"text":"this is still a string"}', "",
                  {"text": "广州", "tasks": "输入字段，不是任务", "state": ["保留"]},
                  ["广州", {"examples": ["输入数据"]}]]
        for profile in ("legacy", "qwen_vllm"):
            for state in states:
                with self.subTest(profile=profile, state=state):
                    context = InferenceContext(state, "共用规则")
                    a = normalize_call("a", "generate", ("保持主体与时间。",),
                                       {"min_length": 1, "max_length": 80}, context)
                    b = normalize_call("b", "generate_list", ("产品维度。",),
                                       {"min_items": 0, "max_items": 8, "min_length_per_item": 1,
                                        "max_length_per_item": 80}, context)
                    backend = self.backend(profile)
                    bodies = [backend._single_payload(context, a, 160),
                              backend._single_payload(context, b, 160),
                              backend._single_payload(context, a, 160, "格式错误"),
                              backend._batch_payload(context, [a, b], 400)]
                    for body in bodies:
                        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
                        self.assertEqual(body["messages"][1]["content"],
                                         state if isinstance(state, str) else encoded(state))
                        prompt = body["messages"][0]["content"]
                        self.assertTrue(prompt.startswith("共用规则\n\n输出规则\n"))
                        self.assertIn("字符长度为建议", prompt)
                        for key in ('"instructions":', '"min_items":', '"max_length":',
                                    "state_ref", "example_states", "max_num_tokens"):
                            self.assertNotIn(key, prompt)
                        self.assertNotIn("\nExample\n", prompt)
                    self.assertIn("只返回所需文本，不加 JSON 包装或引号", bodies[0]["messages"][0]["content"])
                    self.assertIn("唯一字段 items 为字符串数组", bodies[1]["messages"][0]["content"])
                    self.assertIn("唯一顶层字段为 answers", bodies[-1]["messages"][0]["content"])
                    self.assertIn('任务 "q2"（generate_list）：\n产品维度。', bodies[-1]["messages"][0]["content"])

    def test_examples_group_inputs_without_overwriting_answers_or_changing_calls(self):
        first = {"text": "示例广州", "meta": ["甲"]}
        reordered = {"meta": ["甲"], "text": "示例广州"}
        second = {"text": "示例南京"}
        context = InferenceContext({"text": "当前北京"}, "共享规则")
        a = normalize_call("a", "generate", ("生成标题。",),
                           {"max_length": 80, "examples": [
                               Example(first, '标题"甲"\n😀'), Example(second, "标题乙"),
                               Example(reordered, "另一答案")]}, context)
        b = normalize_call("b", "extract", ("地点。",),
                           {"source_field": "text", "max_items": 2, "max_length_per_item": 80,
                            "examples": [Example(reordered, ["广州"])]}, context)
        c = normalize_call("c", "generate_list", ("维度。",),
                           {"max_items": 1, "max_length_per_item": 80}, context)
        original = [deepcopy(thaw(call.arguments)) for call in (a, b, c)]
        for profile in ("legacy", "qwen_vllm"):
            backend = self.backend(profile)
            body = backend._batch_payload(context, [a, b, c], 400)
            prompt = body["messages"][0]["content"]
            rules, examples = prompt.split("\n\nExample\n")
            self.assertNotIn("当前北京", prompt)
            self.assertNotIn("示例广州", rules)
            self.assertEqual(examples.count("输入：" + encoded(first)), 1)
            self.assertNotIn("输入：" + encoded(reordered), examples)
            self.assertLess(examples.index("示例广州"), examples.index("示例南京"))
            outputs = [json.loads(line.removeprefix("示例输出：")) for line in examples.splitlines()
                       if line.startswith("示例输出：")]
            self.assertEqual(outputs, [{"answers": {"q1": '标题"甲"\n😀'}},
                                       {"answers": {"q1": "另一答案", "q2": ["广州"]}},
                                       {"answers": {"q1": "标题乙"}}])
            self.assertNotIn('"q3"', examples)
            self.assertIn("当前正式回答必须包含全部任务", examples)
            self.assertNotIn("state_ref", prompt)
            repair = backend._single_payload(context, b, 160, "格式错误")
            repaired_prompt = repair["messages"][0]["content"]
            self.assertLess(repaired_prompt.index("格式错误"), repaired_prompt.index("\n\nExample\n"))
            self.assertNotIn('任务 "a"', repaired_prompt)
            self.assertNotIn("示例南京", repaired_prompt)
            self.assertEqual(repair["messages"][1], body["messages"][1])
            if profile == "qwen_vllm":
                self.assertEqual(body["structured_outputs"]["grammar"], grammar([a, b, c], batch=True))
                self.assertEqual(repair["structured_outputs"]["grammar"], grammar([b]))
            self.assertEqual([thaw(call.arguments) for call in (a, b, c)], original)

    def test_natural_limits_preserve_zero_nonempty_and_source_semantics(self):
        for profile in ("legacy", "qwen_vllm"):
            backend = self.backend(profile)
            for source in (None, '正文"\\\n😀'):
                context = InferenceContext("北京" if source is None else {source: "北京"})
                call = normalize_call("e", "extract", ("逐字抽取。",),
                                      {"source_field": source, "min_items": 1, "max_items": 8,
                                       "max_length_per_item": 80}, context)
                prompt = backend._single_payload(context, call, 160)["messages"][0]["content"]
                self.assertIn("返回 1 至 8 项", prompt)
                self.assertIn("非空连续原文，不得改写或拼接", prompt)
                self.assertIn("每项建议不超过 80 字符", prompt)
                self.assertIn("输入字符串" if source is None else "输入的顶层字段 " + encoded(source), prompt)
                self.assertNotIn("北京", prompt)
            context = InferenceContext([])
            cases = [
                ("generate", {"min_length": 0, "max_length": 0}, "允许空文本。建议长度为 0 至 0 字符"),
                ("generate", {"min_length": 3, "max_length": 80}, "必须非空。建议长度为 3 至 80 字符"),
                ("generate_list", {"max_items": 0, "max_length_per_item": 0}, "返回 0 至 0 项"),
                ("generate_list", {"max_items": 8, "min_length_per_item": 3, "max_length_per_item": 80},
                 "每项为非空字符串。每项建议长度为 3 至 80 字符"),
            ]
            for operation, options, expected in cases:
                call = normalize_call("q", operation, ("说明",), options, context)
                self.assertIn(expected, backend._single_payload(context, call, 160)["messages"][0]["content"])

    def test_profiles_share_messages_and_no_fixture_data_is_sent(self):
        context = InferenceContext({"text": "当前输入"})
        calls = [normalize_call("a", "generate", ("标题",),
                                {"max_length": 80, "examples": [Example(["示例输入"], "示例答案")]}, context),
                 normalize_call("b", "generate_list", ("维度",),
                                {"max_items": 8, "max_length_per_item": 80}, context)]
        a, b = self.backend("legacy"), self.backend("qwen_vllm")
        for feedback in (None, "错误"):
            left = a._single_payload(context, calls[0], 160, feedback)
            right = b._single_payload(context, calls[0], 160, feedback)
            self.assertEqual(left["messages"], right["messages"])
            self.assertEqual(set(left), {"model", "messages", "thinking", "max_tokens"})
            self.assertEqual(set(right), {"model", "messages", "chat_template_kwargs", "max_tokens", "structured_outputs"})
        self.assertEqual(a._batch_payload(context, calls, 400)["messages"],
                         b._batch_payload(context, calls, 400)["messages"])
