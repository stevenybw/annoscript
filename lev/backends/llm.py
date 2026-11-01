"""Chat completions adapter for generation and validated verbatim extraction."""

import json
import hashlib
import time

from lev._data import thaw, freeze
from lev._backend_profiles import validate_generation_profile, validate_extract_mode
from lev.lev import ExtractResponse, GenerateResponse, GenerateListResponse
from .base import (HTTPBackend, InferenceOutput, InferenceError, validate_response,
                   LLM_OPERATIONS, BatchResult, LLMBatchPlan, PreparedLLMRequest, encode_payload, normalize_call)
from ._generation_policy import validate_policy, PinnedTokenCounter, effective_budget, validate_token_ratio
from ._bounded_grammar import grammar as bounded_grammar, ensure_satisfiable
from ._shared_rules import factor_rules, validate_rule_dedup
from lev._request_adapters import DEFAULT_GENERATION_ADAPTER
from .adapters import generation_adapter
from ._preview import request_preview
from ._batch_wire import FORMAT, batch_calls, wire_name


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


class LLMAPIBackend(HTTPBackend):
    supports_llm_batch = True
    supports_prepared_batch = True

    def __init__(self, *args, profile="legacy", request_adapter=DEFAULT_GENERATION_ADAPTER, extract_mode="validated", disable_thinking=True,
                 request_policy=None, assumed_avg_num_tokens_per_char=2, batch_rule_dedup=False, _token_counter=None, **kwargs):
        if "source_layout" in kwargs:
            raise TypeError("source_layout was removed; extract now uses source_field")
        self._request_adapter = generation_adapter(request_adapter)
        self.request_adapter = request_adapter
        self.profile = validate_generation_profile(profile)
        self.batch_rule_dedup = validate_rule_dedup(batch_rule_dedup, self.profile)
        self.extract_mode = validate_extract_mode(extract_mode, self.profile)
        if type(disable_thinking) is not bool:
            raise ValueError("disable_thinking must be a boolean")
        self.assumed_avg_num_tokens_per_char = validate_token_ratio(assumed_avg_num_tokens_per_char)
        self.request_policy = validate_policy(request_policy)
        if self.request_policy is not None and self.profile != "qwen_vllm":
            raise ValueError("request_policy requires qwen_vllm")
        self._token_counter = ((_token_counter or PinnedTokenCounter(self.request_policy["tokenizer"]))
                               if self.request_policy is not None else None)
        super().__init__(*args, **kwargs)
        self.disable_thinking = disable_thinking
        self._last_batch = None
        self.preparation_stats = dict(payload_builds=0, token_counts=0, encodes=0,
                                      batch_plan_hits=0, grammar_seconds=0.0,
                                      token_seconds=0.0, encoding_seconds=0.0)

    def _sources_evidence(self, context, calls):
        bindings = []
        for call in calls:
            if call.operation != "extract":
                continue
            resolved = normalize_call(call.id, call.operation, (), thaw(call.arguments), context)
            if resolved.extraction_source != call.extraction_source:
                raise ValueError("extract source_field binding differs from request state")
            bindings.append({"call_id": call.id, "source_field": call.arguments["source_field"],
                             "source_sha256": hashlib.sha256(call.extraction_source.encode()).hexdigest()})
        return {"source_layout": "source_field_v1", "source_bindings": bindings}

    def _grammar(self, calls, *, batch=False):
        started = time.perf_counter()
        try:
            return bounded_grammar(calls, batch=batch, extract_mode=self.extract_mode)
        finally:
            self.preparation_stats["grammar_seconds"] += time.perf_counter() - started

    def _binding(self, context, calls, max_tokens, repair_feedback=None):
        value = {"context_identity": id(context), "backend_identity": id(self),
                 "state": thaw(context.state), "system_prompt": context.system_prompt,
                 "calls": [{"id": c.id, "operation": c.operation, "arguments": thaw(c.arguments),
                            "extraction_source": c.extraction_source} for c in calls],
                 "model": self.model, "base_url": self.base_url, "profile": self.profile,
                 "request_adapter": self.request_adapter,
                 "extract_mode": self.extract_mode, "source_layout": "source_field_v1",
                 "batch_rule_dedup": self.batch_rule_dedup,
                 "batch_response_format": FORMAT,
                 "disable_thinking": self.disable_thinking, "request_policy": self.request_policy,
                 "token_ratio": self.assumed_avg_num_tokens_per_char,
                 "max_tokens": max_tokens, "repair_feedback": repair_feedback}
        return hashlib.sha256(encode_payload(value)).hexdigest()

    def _prepare(self, context, calls, max_tokens, *, batch=False, repair_feedback=None):
        started = time.perf_counter()
        before = dict(self.preparation_stats)
        self.preparation_stats["payload_builds"] += 1
        payload = (self._batch_payload(context, calls, max_tokens) if batch else
                   self._single_payload(context, calls[0], max_tokens, repair_feedback))
        evidence = {"request_adapter": self.request_adapter, **self._context_evidence(payload), **self._sources_evidence(context, calls)}
        if batch:
            evidence.update(batch_response_format=FORMAT,
                            wire_call_ids={key: call.id for key, call in batch_calls(calls).items()})
        elif calls[0].arguments.get("field_name") is not None:
            evidence["wire_call_ids"] = {wire_name(calls[0], calls[0].id): calls[0].id}
        if batch and self.batch_rule_dedup:
            tasks = {c.id: {"task": {"instructions": c.arguments["instructions"]}} for c in calls}
            _, rule_evidence = factor_rules(tasks)
            evidence["shared_rules"] = rule_evidence
        encoded_started = time.perf_counter()
        body = encode_payload(payload)
        self.preparation_stats["encodes"] += 1
        self.preparation_stats["encoding_seconds"] += time.perf_counter() - encoded_started
        binding = self._binding(context, calls, max_tokens, repair_feedback)
        evidence["request_body_sha256"] = hashlib.sha256(body).hexdigest()
        evidence["preparation"] = {k: v - before[k] for k, v in self.preparation_stats.items()}
        evidence["preparation"]["seconds"] = time.perf_counter() - started
        return PreparedLLMRequest(binding, freeze(payload), body, freeze(evidence))

    def resolve_token_budget(self, call, fallback):
        return effective_budget(call, self.assumed_avg_num_tokens_per_char)

    def _context_evidence(self, payload):
        if self.request_policy is None:
            return {"assumed_avg_num_tokens_per_char": self.assumed_avg_num_tokens_per_char}
        started = time.perf_counter()
        try:
            tokens = self._token_counter(payload)
        finally:
            self.preparation_stats["token_counts"] += 1
            self.preparation_stats["token_seconds"] += time.perf_counter() - started
        return {"assumed_avg_num_tokens_per_char": self.assumed_avg_num_tokens_per_char, "input_tokens": tokens, "output_budget": payload["max_tokens"],
                "context_window": self.request_policy["context_window"],
                "effective_sampling": self.request_policy["sampling"],
                "tokenizer_revision": self.request_policy["tokenizer"]["revision"],
                "grammar_sha256": hashlib.sha256(payload["structured_outputs"]["grammar"].encode()).hexdigest()}

    def _check_context(self, payload, calls, *, repair=False, prepared=None):
        evidence = thaw(prepared.evidence) if prepared is not None else self._context_evidence(payload)
        if "input_tokens" in evidence and evidence["input_tokens"] + evidence["output_budget"] > evidence["context_window"]:
            self._diagnose({"event": "local_rejection", "code": "context_length_exceeded",
                                  "dispatch_status": "not_sent", "calls": [c.id for c in calls],
                                  "repair": repair, **evidence})
            raise InferenceError("input tokens plus output budget exceed context window",
                                 code="context_length_exceeded", metadata={"dispatch_status": "not_sent", **evidence})
        return evidence

    async def _preflight(self, payload, calls, *, repair=False, prepared=None):
        return self._check_context(payload, calls, repair=repair, prepared=prepared)

    async def _check_constraints(self, calls, *, repair=False):
        self._validate_constraints(calls, repair=repair)

    def _validate_constraints(self, calls, *, repair=False):
        for call in calls:
            effective_budget(call, self.assumed_avg_num_tokens_per_char)
            try:
                ensure_satisfiable(call)
            except InferenceError as exc:
                self._diagnose({"event": "local_rejection", "code": exc.code,
                                      "dispatch_status": "not_sent", "calls": [call.id], "repair": repair})
                raise

    def _single_payload(self, context, call, max_tokens, repair_feedback=None):
        return self._request_adapter.single_payload(
            context, call, model=self.model, profile=self.profile, disable_thinking=self.disable_thinking,
            max_tokens=max_tokens, grammar=self._grammar, repair_feedback=repair_feedback,
            sampling=self.request_policy["sampling"] if self.request_policy is not None else None)

    def preview(self, context, call, *, repair_feedback=None, max_tokens_override=None):
        """Preview one resolved call, with the same constraints and preflight as infer."""
        if call.operation not in LLM_OPERATIONS:
            raise ValueError("unsupported LLM operation")
        self._validate_constraints([call], repair=repair_feedback is not None)
        budget = effective_budget(call, self.assumed_avg_num_tokens_per_char, max_tokens_override)
        return self._preview(context, [call], budget, repair_feedback=repair_feedback)

    def preview_batch(self, context, calls, *, max_tokens):
        """Preview an already selected batch; does not speculate or choose a path."""
        batch_calls(calls)
        self._validate_constraints(calls)
        return self._preview(context, calls, max_tokens, batch=True)

    def _preview(self, context, calls, max_tokens, *, batch=False, repair_feedback=None):
        prepared = self._prepare(context, calls, max_tokens, batch=batch, repair_feedback=repair_feedback)
        payload = thaw(prepared.payload)
        evidence = self._check_context(payload, calls, repair=repair_feedback is not None, prepared=prepared)
        evidence.pop("preparation", None)
        return request_preview(payload, stage="generation", adapter=self.request_adapter,
                               url=self.base_url + "/chat/completions", evidence=evidence)

    async def infer(self, context, call, *, repair_feedback=None, max_tokens_override=None):
        if call.operation not in LLM_OPERATIONS:
            raise ValueError("unsupported LLM operation")
        await self._check_constraints([call], repair=repair_feedback is not None)
        max_tokens = effective_budget(call, self.assumed_avg_num_tokens_per_char, max_tokens_override)
        prepared = self._prepare(context, [call], max_tokens, repair_feedback=repair_feedback)
        payload = thaw(prepared.payload)
        evidence = await self._preflight(payload, [call], repair=repair_feedback is not None, prepared=prepared)
        data, metadata = await self.post(self.base_url + "/chat/completions", payload,
            diagnostic_context={"calls": [{"id": call.id, "operation": call.operation, "arguments": thaw(call.arguments)}],
                                "repair": repair_feedback is not None, **evidence}, _encoded_body=prepared.body)
        try:
            choice = data["choices"][0]
            if choice["finish_reason"] != "stop":
                raise InferenceError("generation did not complete normally", code="incomplete_generation", metadata=metadata)
            message = choice["message"]
            if message.get("refusal") or type(message["content"]) is not str:
                raise ValueError()
            content = message["content"]
            if call.operation == "generate":
                response = GenerateResponse(json.loads(content) if call.arguments.get("nullable", False) else content)
            else:
                value = json.loads(content)
                if type(value) is not dict or set(value) != {"items"}:
                    raise ValueError()
                cls = ExtractResponse if call.operation == "extract" else GenerateListResponse
                response = cls(value["items"])
            validate_response(call, response)
        except InferenceError as exc:
            exc.metadata = metadata
            raise
        except (KeyError, IndexError, TypeError, ValueError):
            raise InferenceError("invalid LLM output", code="invalid_response", metadata=metadata) from None
        metadata["advisory_lengths"] = {call.id: self._length_observation(call, response)}
        self._diagnose({"event": "advisory_lengths", "request_id": metadata.get("request_id"),
                        "observations": metadata["advisory_lengths"]})
        return InferenceOutput(response, metadata)

    @staticmethod
    def _length_observation(call, response):
        a = call.arguments
        single = call.operation == "generate"
        lengths = ([] if response.text is None else [len(response.text)]) if single else [len(item) for item in response.items]
        low = a.get("min_length" if single else "min_length_per_item", 0)
        high = a["max_length" if single else "max_length_per_item"]
        return {**({"is_null": response.text is None} if single and a.get("nullable", False) else {}),
                "actual_codepoints": lengths, "suggested_min": low, "suggested_max": high,
                "outside_suggestion": sum(n < low or n > high for n in lengths),
                "hard_failure": False}

    def _batch_payload(self, context, calls, max_tokens):
        if (not calls or len({c.id for c in calls}) != len(calls)
                or type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("a batch requires distinct calls and a positive token budget")
        for call in calls:
            if call.operation not in LLM_OPERATIONS:
                raise ValueError("unsupported LLM operation")
            effective_budget(call, self.assumed_avg_num_tokens_per_char)
        return self._request_adapter.batch_payload(
            context, calls, model=self.model, profile=self.profile, disable_thinking=self.disable_thinking,
            max_tokens=max_tokens, grammar=self._grammar, rule_dedup=self.batch_rule_dedup,
            sampling=self.request_policy["sampling"] if self.request_policy is not None else None)

    def plan_batch(self, context, calls, *, max_calls, max_input_bytes, max_output_tokens, default_tokens):
        """Choose a bounded prefix without network work; None means use single inference."""
        if any(type(v) is not int or v < 1 for v in (max_calls, max_input_bytes, max_output_tokens, default_tokens)):
            raise ValueError("invalid batch planning limits")
        if len(calls) < 2 or max_calls < 2:
            return None
        key = (self._binding(context, calls, max_output_tokens), max_calls, max_input_bytes, default_tokens)
        if self._last_batch is not None and self._last_batch[0] is context and self._last_batch[1] == key:
            self.preparation_stats["batch_plan_hits"] += 1
            return self._last_batch[2]
        selected, budget, plan = [], 64, None
        if self.request_policy is not None:
            default_tokens = self.request_policy["default_max_tokens"]
        for call in calls[:max_calls]:
            budget += effective_budget(call, self.assumed_avg_num_tokens_per_char) + 32
            if budget > max_output_tokens:
                break
            candidate = selected + [call]
            try:
                prepared = self._prepare(context, candidate, budget, batch=True)
            except InferenceError:
                break  # The offending single call reports its own local error when dispatched.
            size = len(prepared.body)
            if size > max_input_bytes:
                break
            evidence = prepared.evidence
            if "input_tokens" in evidence and evidence["input_tokens"] + budget > evidence["context_window"]:
                break
            selected = candidate
            if len(selected) >= 2:
                plan = LLMBatchPlan(tuple(selected), budget, size, prepared)
        self._last_batch = (context, key, plan)
        return plan

    async def infer_batch(self, context, calls, *, max_tokens, prepared=None):
        identified = batch_calls(calls)
        await self._check_constraints(calls)
        if prepared is None or prepared.binding != self._binding(context, calls, max_tokens):
            prepared = self._prepare(context, calls, max_tokens, batch=True)
        payload = thaw(prepared.payload)
        evidence = await self._preflight(payload, calls, prepared=prepared)
        data, metadata = await self.post(self.base_url + "/chat/completions", payload, logical_calls=len(calls),
            diagnostic_context={"calls": [{"id": c.id, "operation": c.operation, "arguments": thaw(c.arguments)} for c in calls],
                                "repair": False, **evidence}, _encoded_body=prepared.body)
        try:
            choice = data["choices"][0]
            if type(choice) is not dict:
                raise ValueError()
            if choice.get("finish_reason") == "content_filter":
                raise InferenceError("model refused batch generation", code="model_refusal", metadata=metadata)
            message = choice["message"]
            if type(message) is not dict:
                raise ValueError()
            if message.get("refusal"):
                raise InferenceError("model refused batch generation", code="model_refusal", metadata=metadata)
            if choice["finish_reason"] == "length":
                raise InferenceError("batch generation was truncated", code="incomplete_batch_generation", metadata=metadata)
            if choice["finish_reason"] != "stop":
                raise InferenceError("generation did not complete normally", code="incomplete_generation", metadata=metadata)
            value = json.loads(message["content"], object_pairs_hook=_unique_object)
            if type(value) is not dict or set(value) != {"answers"} or type(value["answers"]) is not dict:
                raise ValueError()
            answers = value["answers"]
            if set(answers) - set(identified):
                raise ValueError()
        except InferenceError:
            raise
        except (KeyError, IndexError, TypeError, ValueError):
            raise InferenceError("unreliable batch output mapping", code="invalid_batch_output", metadata=metadata) from None
        responses, errors = {}, {}
        for identity, call in identified.items():
            try:
                answer = answers[identity]
                cls = {"generate": GenerateResponse, "generate_list": GenerateListResponse, "extract": ExtractResponse}[call.operation]
                responses[call.id] = validate_response(call, cls(answer))
            except InferenceError as exc:
                exc.metadata = metadata
                errors[call.id] = exc
            except (KeyError, TypeError, ValueError):
                errors[call.id] = InferenceError("missing or invalid batch answer", code="invalid_response",
                                                 metadata=metadata, call_id=call.id)
        metadata["advisory_lengths"] = {c.id: self._length_observation(c, responses[c.id])
                                        for c in calls if c.id in responses}
        self._diagnose({"event": "advisory_lengths", "request_id": metadata.get("request_id"),
                        "observations": metadata["advisory_lengths"]})
        return BatchResult(responses, metadata, errors)
