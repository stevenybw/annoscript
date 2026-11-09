"""Deterministic response tables, independent of instruction parsing."""
from copy import deepcopy
import json

from annoscript._data import thaw
from annoscript.backends import BatchResult, InferenceError, InferenceOutput
from annoscript.anno import (ChoiceResponse, ExtractResponse, GenerateListResponse,
                     GenerateResponse, NoulResponse, ScoreResponse)

REVIEW_INPUT = {"text": "示例条目"}
LAUNCH_INPUT = {"text": "示例在广州发布"}


def key(value):
    return json.dumps(thaw(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CannedBackend:
    def __init__(self, scenarios=(), failures=None, *, speculative=False):
        self.tables = {key(state): deepcopy(value) for state, value in scenarios}
        self.failures, self.calls = failures or {}, []

    def answer(self, context, call):
        state_key = key(context.state)
        field = call.arguments.get("field_name", call.id)
        self.calls.append((state_key, call.operation, field))
        if (state_key, field) in self.failures:
            raise InferenceError("injected model failure", code=self.failures[state_key, field])
        value = self.tables[state_key][field]
        if call.operation == "noul":
            return NoulResponse(0.9 if value else 0.1)
        if call.operation == "choice":
            return ChoiceResponse(value, 1.0, {k: float(k == value) for k in call.arguments["criteria"]})
        if call.operation == "score":
            return ScoreResponse({0: 0.0, 1: 1.0}, value, {0: "low", 1: "high"}, 1.0)
        if call.operation == "generate":
            return GenerateResponse(value)
        if call.operation == "generate_list":
            return GenerateListResponse(deepcopy(value))
        if call.operation == "extract":
            return ExtractResponse(deepcopy(value))
        raise AssertionError(call.operation)

    async def infer(self, context, call):
        return InferenceOutput(self.answer(context, call))

    async def infer_batch(self, context, calls):
        return BatchResult({call.id: self.answer(context, call) for call in calls})

    async def aclose(self):
        pass


def cases():
    scenarios = []
    for index, text in enumerate(["alpha beta", "gamma delta", "示例甲 示例乙", "one two",
                                  "three four", "five six", "seven eight", "nine ten"]):
        state = {"text": text, "kind": "note"}
        value = {"category": "note", "highlight": index != 1, "clarity": 1.0,
                 "title": None if index == 1 else "Synthetic title", "tags": ["sample"],
                 "spans": [text.split()[0]]}
        scenarios.append((state, value))
    inputs = [{"text": ""}] + [s for s, _ in scenarios]
    expected = [{"status": "error", "code": "invalid_input"}]
    expected += [{"status": "success", "value": value} for _, value in scenarios[:-1]]
    expected += [{"status": "error", "code": "model_timeout"}]
    return scenarios, inputs, expected, {(key(scenarios[-1][0]), "title"): "model_timeout"}
