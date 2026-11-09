"""Deterministic model HTTP responses; never connects to a model service."""

from dataclasses import asdict
from collections import OrderedDict
import hashlib
import json

import httpx

from lev._test_synthetic import CannedBackend
from lev.backends import DecisionAPIBackend, LLMAPIBackend as _LLMAPIBackend
from lev.lev import GenerateResponse


_CAPTURES = OrderedDict()


def _key(body):
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).digest()


class FixtureLLMBackend(_LLMAPIBackend):
    """Test-only capture before transport; never add fixture metadata to the wire.

    Fixed answers use the original immutable calls, not a parser for prose. Tests
    separately assert messages, grammars and budgets on the actual HTTP requests.
    The bounded registry keeps captures available after a runtime closes its backend.
    """

    @staticmethod
    def _capture(payload, context, calls, batch):
        key = _key(payload)
        _CAPTURES[key] = (batch, tuple((context, call) for call in calls))
        _CAPTURES.move_to_end(key)
        if len(_CAPTURES) > 4096:
            _CAPTURES.popitem(last=False)
        return payload

    def _single_payload(self, context, call, max_tokens, repair_feedback=None):
        payload = super()._single_payload(context, call, max_tokens, repair_feedback)
        return self._capture(payload, context, [call], False)

    def _batch_payload(self, context, calls, max_tokens):
        payload = super()._batch_payload(context, calls, max_tokens)
        return self._capture(payload, context, calls, True)


class FixtureDecisionBackend(DecisionAPIBackend):
    """Keep original call identity and example order outside the model request."""

    def _batch_payload(self, context, calls):
        payload = super()._batch_payload(context, calls)
        return FixtureLLMBackend._capture(payload, context, calls, True)


def decision_calls(body):
    return list(_CAPTURES[_key(body)][1])


def llm_calls(body):
    """Look up test-side calls by the exact request, without parsing model prose."""
    batch, calls = _CAPTURES[_key(body)]
    return batch, list(calls)


def llm_content(body, answer):
    batched, calls = llm_calls(body)
    values = [asdict(answer(context, call))["text" if call.operation == "generate" else "items"]
              for context, call in calls]
    if batched:
        return json.dumps({"answers": {call.arguments.get("field_name", f"q{index}"): value
                                      for index, ((_, call), value) in enumerate(zip(calls, values), 1)}},
                          ensure_ascii=False, separators=(",", ":"))
    call = calls[0][1]
    if call.operation == "generate":
        return json.dumps(values[0], ensure_ascii=False) if call.arguments.get("nullable", False) else values[0]
    return json.dumps({"items": values[0]}, ensure_ascii=False)


def llm_wire_id(body, call_id):
    """Test-side alias lookup, independent of the production mapping helper."""
    batched, calls = llm_calls(body)
    assert batched
    return next(call.arguments.get("field_name", f"q{index}")
                for index, (_, call) in enumerate(calls, 1) if call.id == call_id)


def http_backends(scenarios=(), *, requests=None, request_bytes=None, override=None, profile="qwen_vllm", speculative=False):
    canned = CannedBackend(scenarios, speculative=speculative)
    requests = requests if requests is not None else []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        if request_bytes is not None:
            request_bytes.append(len(request.content))
        if override is not None:
            response = override(body)
            if response is not None:
                return response
        if "questions" in body:
            answers = {}
            for context, call in decision_calls(body):
                answer = canned.answer(context, call)
                identity = call.arguments.get("field_name", call.id)
                answers[identity] = {"type": call.operation, **asdict(answer)}
            return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 11, "output_tokens": 3}})
        content = llm_content(body, lambda context, call: GenerateResponse(context.state)
                              if type(context.state) is str else canned.answer(context, call))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                      "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9,
                                                "completion_tokens_details": {"reasoning_tokens": 1}}})

    transport = httpx.MockTransport(handle)
    return (FixtureDecisionBackend("http://offline.invalid", "fixture", "fixture", transport=transport),
            FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile=profile, transport=transport))
