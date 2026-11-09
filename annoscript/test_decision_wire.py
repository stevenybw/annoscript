"""Decision request layout and public names across real HTTP and runtime paths."""

from copy import deepcopy
import json
import unittest

import httpx

from annoscript._data import thaw
from annoscript.backends import DecisionAPIBackend, InferenceContext, InferenceError
from annoscript.backends.base import normalize_call
from annoscript.compiler import compile_program
from annoscript.ir import Program
from annoscript.plan import ExecutionPlan
from annoscript.plan_lowering import lower_to_plan
from annoscript.runtime import AnnoRuntime, RunInput
from annoscript.test_runtime import FakeBackend


def named_decisions(anno):
    first = anno.choice("选择", {"a": "甲", "b": "乙"}, field_name="category").choice
    second = anno.noul("判断", field_name="is_event").noul
    third = anno.score("程度", ["低", "高"], field_name="severity").score
    again = anno.choice("选择", {"a": "甲", "b": "乙"}, field_name="category").choice
    return [first, second, third, again]


def answers_for(body):
    answers = {}
    for name, q in reversed(list(body["questions"].items())):
        if q["type"] == "choice":
            selected = next(iter(q["criteria"]))
            answer = {"choice": selected, "confidence": 1.,
                      "probabilities": {key: float(key == selected) for key in q["criteria"]}}
        elif q["type"] == "noul":
            answer = {"noul": .8}
        else:
            answer = {"score": 1., "confidence": 1., "probabilities": {"0": 0., "1": 1.},
                      "legend": {str(i): value for i, value in enumerate(q["criteria"])}}
        answers[name] = {"type": q["type"], **answer}
    return answers


class DecisionWireTests(unittest.IsolatedAsyncioTestCase):
    def call(self, operation, identity="internal", *, context=None, **options):
        args = {"choice": ("选择", {"a": "甲", "b": "乙"}),
                "noul": ("判断",), "score": ("程度", ["低", "高"])}[operation]
        return normalize_call(identity, operation, args, options, context or InferenceContext("当前输入"))

    async def test_grouped_examples_preserve_states_order_and_original_arguments(self):
        context = InferenceContext({"text": "当前事实"}, "共享规则仅供 LLM 使用")
        states = [{"text": "示例甲", "nested": [1, {"answer": "数据"}]}, "示例乙", ["示例丙"]]
        calls = [self.call(op, identity=op, field_name=name, requirements=["额外规则", "额外规则"],
                           examples=[{"state": state, "answer": answer} for state, answer in zip(states, answers)])
                 for op, name, answers in (("choice", "category", ["b", "a", "b"]),
                                           ("noul", "is_event", [False, True, False]),
                                           ("score", "severity", [1, 0, 1]))]
        original = [thaw(c.arguments) for c in calls]
        sent, events = [], []
        def handle(request):
            body = json.loads(request.content)
            sent.append(body)
            return httpx.Response(200, json={"answers": answers_for(body)})
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture",
                                     transport=httpx.MockTransport(handle), diagnostic_observer=events.append)
        try:
            result = await backend.infer_batch(context, calls)
        finally:
            await backend.aclose()
        body = sent[0]
        self.assertEqual(body["state"], thaw(context.state))
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(list(body["questions"]), ["category", "is_event", "severity"])
        for call, q in zip(calls, body["questions"].values()):
            self.assertEqual(q["instructions"], call.arguments["instructions"] + "\n\n要求：\n额外规则")
            self.assertEqual(set(q), {"type", "instructions", "criteria"})
        self.assertEqual(body["questions"]["category"]["criteria"], {
            "a": {"what": "甲", "examples": [states[1]]},
            "b": {"what": "乙", "examples": [states[0], states[2]]}})
        self.assertEqual(body["questions"]["is_event"]["criteria"], {
            "false": {"examples": [states[0], states[2]]}, "true": {"examples": [states[1]]}})
        self.assertEqual(body["questions"]["severity"]["criteria"], [
            {"what": "低", "examples": [states[1]]}, {"what": "高", "examples": [states[0], states[2]]}])
        self.assertEqual([thaw(c.arguments) for c in calls], original)
        self.assertEqual(set(result.responses), {c.id for c in calls})
        self.assertEqual(result.responses["score"].legend, {0: "低", 1: "高"})
        self.assertTrue(any(e.get("wire_call_ids") == {"category": "choice", "is_event": "noul", "severity": "score"}
                            for e in events))

    async def test_empty_examples_and_noul_descriptions(self):
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture")
        context = InferenceContext("当前输入", "不应拼接")
        try:
            for op in ("choice", "noul", "score"):
                plain = self.call(op)
                expected = backend._batch_payload(context, [plain])
                for examples in (None, []):
                    explicit = self.call(op, field_name=None, examples=examples)
                    self.assertEqual(explicit, plain)
                    self.assertEqual(backend._batch_payload(context, [explicit]), expected)
                self.assertEqual(list(expected["questions"]), [plain.id])
                self.assertEqual(expected["questions"][plain.id]["instructions"], plain.arguments["instructions"])
            call = self.call("noul", criteria_false="不成立", criteria_true="成立",
                             examples=[{"state": "肯定示例", "answer": True}])
            self.assertEqual(backend._batch_payload(context, [call])["questions"][call.id]["criteria"],
                             {"false": "不成立", "true": {"what": "成立", "examples": ["肯定示例"]}})
        finally:
            await backend.aclose()

    async def test_invalid_names_and_collisions_fail_before_http(self):
        sent = []
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture",
                                     transport=httpx.MockTransport(lambda r: sent.append(r)))
        try:
            for op in ("choice", "noul", "score"):
                for name in ("", " \t", 1, False, {}, []):
                    with self.subTest(op=op, name=name), self.assertRaisesRegex(ValueError, "field_name"):
                        self.call(op, field_name=name)
            for calls in ([self.call("choice", "a", field_name="same"), self.call("score", "b", field_name="same")],
                          [self.call("choice", "a", field_name="b"), self.call("noul", "b")]):
                with self.assertRaisesRegex(ValueError, "duplicate model-visible"):
                    await backend.infer_batch(InferenceContext("输入"), calls)
            self.assertEqual(sent, [])
        finally:
            await backend.aclose()

    async def test_partial_answers_and_errors_keep_internal_ids(self):
        name = ' 字段"\\\n😀 '
        calls = [self.call("noul", "run-q0", field_name=name), self.call("choice", "run-q1")]
        answers = {name: {"type": "noul", "noul": .8}}
        backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture",
                                     transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": answers})))
        try:
            result = await backend.infer_batch(InferenceContext("输入"), calls, allow_partial=True)
            self.assertEqual(set(result.responses), {"run-q0"})
            self.assertEqual(result.errors["run-q1"].call_id, "run-q1")
            with self.assertRaises(InferenceError):
                await backend.infer_batch(InferenceContext("输入"), calls)
            answers[name]["noul"] = "invalid"
            result = await backend.infer_batch(InferenceContext("输入"), calls, allow_partial=True)
            self.assertEqual(set(result.errors), {"run-q0", "run-q1"})
            answers["unexpected"] = {"type": "noul", "noul": .5}
            with self.assertRaises(InferenceError):
                await backend.infer_batch(InferenceContext("输入"), calls, allow_partial=True)
        finally:
            await backend.aclose()

    async def test_score_legend_rejects_wrong_description_or_examples(self):
        call = self.call("score", field_name="severity", examples=[{"state": "示例", "answer": 1}])
        for legend in ({"0": "低", "1": "高"},
                       {"0": "低", "1": {"what": "高", "examples": ["示例"]}},
                       {"0": "低", "1": {"what": "高", "examples": ["错误示例"]}},
                       {"0": "低", "1": "错误描述"}):
            def handle(request):
                answers = answers_for(json.loads(request.content))
                answers["severity"]["legend"] = legend
                return httpx.Response(200, json={"answers": answers})
            backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture", transport=httpx.MockTransport(handle))
            try:
                if "错误" in json.dumps(legend, ensure_ascii=False):
                    with self.assertRaises(InferenceError):
                        await backend.infer_batch(InferenceContext("输入"), [call])
                else:
                    result = await backend.infer_batch(InferenceContext("输入"), [call])
                    self.assertEqual(result.responses[call.id].legend, {0: "低", 1: "高"})
            finally:
                await backend.aclose()

    async def test_named_decisions_roundtrip_and_reuse_in_all_runtime_modes(self):
        for optimize in (False, True):
            program = Program.from_json(compile_program(named_decisions, optimize=optimize).to_json())
            plan = ExecutionPlan.from_json(lower_to_plan(program).to_json())
            for mode in ("strict", "eager", "speculative"):
                for limit in (1, 32):
                    sent = []
                    def handle(request):
                        body = json.loads(request.content)
                        sent.append(deepcopy(body))
                        return httpx.Response(200, json={"answers": answers_for(body)})
                    backend = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture", transport=httpx.MockTransport(handle))
                    async with AnnoRuntime(backend, FakeBackend(), max_batch_questions=limit) as runtime:
                        result, = [r async for r in await runtime.submit(plan, [RunInput(0, "输入")], mode=mode)]
                    self.assertEqual(result.status, "success", result.error)
                    self.assertEqual(result.value, ["a", .8, 1., "a"])
                    names = [name for body in sent for name in body["questions"]]
                    self.assertCountEqual(names, ["category", "is_event", "severity"])


if __name__ == "__main__":
    unittest.main()
