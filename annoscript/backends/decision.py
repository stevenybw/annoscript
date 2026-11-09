"""Decision service: one physical request per compatible, identified batch."""

import hashlib

from annoscript._data import thaw, freeze
from annoscript.anno import NoulResponse, ChoiceResponse, ScoreResponse
from .base import (HTTPBackend, BatchResult, InferenceError, validate_response, encode_payload,
                   DecisionBatchPlan, PreparedDecisionRequest)
from annoscript._request_adapters import DEFAULT_DECISION_ADAPTER
from .adapters import decision_adapter
from .adapters.decision import identified_calls as _identified_calls
from ._preview import request_preview


class DecisionAPIBackend(HTTPBackend):
    supports_partial_answers = True
    supports_decision_batch_planning = True
    supports_prepared_batch = True

    def __init__(self, *args, request_adapter=DEFAULT_DECISION_ADAPTER, request_adapter_options=None, **kwargs):
        self._request_adapter = decision_adapter(request_adapter, request_adapter_options)
        super().__init__(*args, **kwargs)

    @property
    def request_adapter(self):
        return self._request_adapter.name

    @property
    def request_adapter_options(self):
        return self._request_adapter.options

    def _batch_payload(self, context, calls):
        return self._request_adapter.batch_payload(context, calls, model=self.model)

    def _binding(self, context, calls):
        value = {"context_identity": id(context), "backend_identity": id(self), "model": self.model,
                 "base_url": self.base_url, "state": thaw(context.state),
                 "request_adapter": self.request_adapter, "request_adapter_options": thaw(self.request_adapter_options),
                 "calls": [{"id": c.id, "operation": c.operation, "arguments": thaw(c.arguments)} for c in calls]}
        return hashlib.sha256(encode_payload(value)).hexdigest()

    def _prepare(self, context, calls):
        payload = self._batch_payload(context, calls)
        body = encode_payload(payload)
        evidence = {"request_adapter": self.request_adapter,
                    "request_adapter_options": thaw(self.request_adapter_options),
                    "wire_call_ids": {name: c.id for name, c in _identified_calls(calls).items()},
                    "task_recipes": {c.id: thaw(self._request_adapter.recipe(c)) for c in calls},
                    "state_sha256": hashlib.sha256(encode_payload(payload["state"])).hexdigest(),
                    "request_body_sha256": hashlib.sha256(body).hexdigest()}
        return PreparedDecisionRequest(self._binding(context, calls), freeze(payload), body, freeze(evidence))

    def plan_batch(self, context, calls, *, max_calls):
        """Select an ordered compatible prefix; local call errors are deferred to infer."""
        if type(max_calls) is not int or max_calls < 1 or not calls:
            raise ValueError("Decision planning needs calls and a positive max_calls")
        selected, key = [], None
        for call in calls[:max_calls]:
            try:
                state, _ = self._request_adapter.render(context, call)
                current = encode_payload(state)
            except (InferenceError, ValueError, TypeError):
                # Keep a malformed speculative call isolated. Its error is only
                # observed if adopted, through the regular infer/receive path.
                if not selected:
                    selected.append(call)
                break
            if selected and key != current:
                break
            selected.append(call)
            key = current
        try:
            prepared = self._prepare(context, selected)
        except (InferenceError, ValueError, TypeError):
            # E.g. duplicate wire names retain the old batch failure instead of
            # being silently split, renamed or raised by the scheduler itself.
            prepared = None
        return DecisionBatchPlan(tuple(selected), len(prepared.body) if prepared else None, prepared)

    def _preview(self, context, calls, prepared=None):
        if prepared is None or prepared.binding != self._binding(context, calls):
            prepared = self._prepare(context, calls)
        return request_preview(thaw(prepared.payload), stage="decision", adapter=self.request_adapter,
                               url=self.base_url, evidence=thaw(prepared.evidence))

    def preview_batch(self, context, calls):
        """Preview one physical batch; incompatible views fail without HTTP."""
        return self._preview(context, calls)

    def preview_batches(self, context, calls, *, max_calls=32):
        """Preview all physical prefixes through the same planner as Runtime."""
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        if not calls or len({c.id for c in calls}) != len(calls):
            raise ValueError("a preview plan requires distinct calls")
        remaining, requests = list(calls), []
        while remaining:
            plan = self.plan_batch(context, remaining, max_calls=max_calls)
            requests.append(self._preview(context, plan.calls, plan.prepared))
            remaining = remaining[len(plan.calls):]
        return {"format": "lev-request-preview-plan", "version": 1, "stage": "decision",
                "request_adapter": self.request_adapter, "request_adapter_options": thaw(self.request_adapter_options),
                "logical_calls": [c.id for c in calls], "physical_requests": len(requests), "requests": requests}

    async def infer_batch(self, context, calls, *, allow_partial=False, prepared=None):
        identified = _identified_calls(calls)
        if prepared is None or prepared.binding != self._binding(context, calls):
            prepared = self._prepare(context, calls)
        payload = thaw(prepared.payload)
        questions = payload["questions"]
        data, metadata = await self.post(self.base_url, payload,
            diagnostic_context={"calls": [{"id": c.id, "operation": c.operation, "arguments": thaw(c.arguments)} for c in calls],
                                **thaw(prepared.evidence)}, _encoded_body=prepared.body)
        current_call = None
        try:
            answers = data["answers"]
            if (type(answers) is not dict or set(answers) - set(questions)
                    or not allow_partial and set(answers) != set(questions)):
                raise ValueError()
            results, errors = {}, {}
            for name, call in identified.items():
                current_call = call.id
                try:
                    answer = answers[name]
                    if answer["type"] != call.operation:
                        raise ValueError()
                    if call.operation == "noul":
                        response = NoulResponse(answer["noul"])
                    elif call.operation == "choice":
                        response = ChoiceResponse(answer["choice"], answer["confidence"], answer["probabilities"])
                    else:
                        def indexed(mapping):
                            if type(mapping) is not dict or any(type(k) is not str or str(int(k)) != k for k in mapping):
                                raise ValueError()
                            return {int(k): v for k, v in mapping.items()}
                        legend = indexed(answer["legend"])
                        # Jev can echo the structured levels; expose the original
                        # string legend while still rejecting mismatched levels.
                        if legend == dict(enumerate(questions[name]["criteria"])):
                            legend = dict(enumerate(call.arguments["criteria"]))
                        response = ScoreResponse(indexed(answer["probabilities"]), answer["score"], legend, answer["confidence"])
                    results[call.id] = validate_response(call, response)
                except (KeyError, TypeError, ValueError, InferenceError):
                    if not allow_partial:
                        raise
                    errors[call.id] = InferenceError("invalid Decision answer", code="invalid_response",
                                                     metadata=metadata, call_id=call.id)
        except (KeyError, TypeError, ValueError, InferenceError):
            raise InferenceError("invalid Decision answers", code="invalid_response", metadata=metadata, call_id=current_call) from None
        return BatchResult(results, metadata, errors)
