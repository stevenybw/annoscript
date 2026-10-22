"""Decision service: one state and multiple independently identified questions."""

from lev._data import thaw
from lev.lev import NoulResponse, ChoiceResponse, ScoreResponse
from .base import HTTPBackend, BatchResult, InferenceError, validate_response, normalize_arguments, DECISION_OPERATIONS
from ._batch_wire import wire_name


def _identified_calls(calls):
    if not calls or len({c.id for c in calls}) != len(calls):
        raise ValueError("a batch requires distinct calls")
    identified = {}
    for call in calls:
        name = wire_name(call, call.id)
        if name in identified:
            raise ValueError(f"duplicate model-visible field name: {name!r}")
        identified[name] = call
    return identified


def _question(call):
    if call.operation not in DECISION_OPERATIONS:
        raise ValueError("unsupported Decision operation")
    a = normalize_arguments(call.operation, (), thaw(call.arguments))
    instructions = a["instructions"]
    if a.get("requirements"):
        instructions += "\n\n要求：\n" + "\n".join(dict.fromkeys(a["requirements"]))
    q = {"type": call.operation, "instructions": instructions}
    if call.operation == "noul":
        criteria = {key: a[name] for key, name in (("false", "criteria_false"), ("true", "criteria_true"))
                    if a[name] is not None}
    else:
        criteria = a["criteria"]
    for example in a.get("examples", []):
        key = example["answer"]
        if call.operation == "noul":
            key = "true" if key else "false"
            # A boolean example needs no invented description when none was given.
            criteria.setdefault(key, {})
        if type(criteria[key]) is str:
            criteria[key] = {"what": criteria[key]}
        criteria[key].setdefault("examples", []).append(example["state"])
    if criteria:
        q["criteria"] = criteria
    return q


class DecisionAPIBackend(HTTPBackend):
    supports_partial_answers = True

    def _batch_payload(self, context, calls):
        return {"model": self.model, "state": thaw(context.state),
                "questions": {name: _question(call) for name, call in _identified_calls(calls).items()}}

    async def infer_batch(self, context, calls, *, allow_partial=False):
        identified = _identified_calls(calls)
        payload = self._batch_payload(context, calls)
        questions = payload["questions"]
        data, metadata = await self.post(self.base_url, payload,
            diagnostic_context={"calls": [{"id": c.id, "operation": c.operation, "arguments": thaw(c.arguments)} for c in calls],
                                "wire_call_ids": {name: call.id for name, call in identified.items()}})
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
