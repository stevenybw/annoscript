"""Decision service: one state and multiple independently identified questions."""

from lev._data import thaw
from lev.lev import NoulResponse, ChoiceResponse, ScoreResponse
from .base import HTTPBackend, BatchResult, InferenceError, validate_response, normalize_arguments, DECISION_OPERATIONS
from ._example_layout import EXAMPLE_CONTRACT


class DecisionAPIBackend(HTTPBackend):
    supports_partial_answers = True

    async def infer_batch(self, context, calls, *, allow_partial=False):
        if not calls or len({c.id for c in calls}) != len(calls):
            raise ValueError("a batch requires distinct calls")
        questions = {}
        for call in calls:
            if call.operation not in DECISION_OPERATIONS:
                raise ValueError("unsupported Decision operation")
            a = normalize_arguments(call.operation, (), thaw(call.arguments))
            instructions = a["instructions"]
            if a.get("requirements"):
                instructions += "\n\n要求：\n" + "\n".join(dict.fromkeys(a["requirements"]))
            if a.get("examples"):
                instructions = {
                    "task_instructions": instructions,
                    "example_policy": EXAMPLE_CONTRACT,
                    "examples": a["examples"],
                }
                if context.system_prompt is not None:
                    instructions["shared_instructions"] = context.system_prompt
            elif context.system_prompt is not None:
                instructions = "Shared instructions:\n" + context.system_prompt + "\n\nTask instructions:\n" + instructions
            q = {"type": call.operation, "instructions": instructions}
            if call.operation == "noul":
                criteria = {key: a[name] for key, name in (("false", "criteria_false"), ("true", "criteria_true")) if a[name] is not None}
                if criteria:
                    q["criteria"] = criteria
            else:
                q["criteria"] = a["criteria"]
            questions[call.id] = q
        data, metadata = await self.post(self.base_url, {"model": self.model, "state": thaw(context.state), "questions": questions},
            diagnostic_context={"calls": [{"id": c.id, "operation": c.operation, "arguments": thaw(c.arguments)} for c in calls]})
        current_call = None
        try:
            answers = data["answers"]
            if (type(answers) is not dict or set(answers) - set(questions)
                    or not allow_partial and set(answers) != set(questions)):
                raise ValueError()
            results, errors = {}, {}
            for call in calls:
                current_call = call.id
                try:
                    answer = answers[call.id]
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
                        response = ScoreResponse(indexed(answer["probabilities"]), answer["score"], indexed(answer["legend"]), answer["confidence"])
                    results[call.id] = validate_response(call, response)
                except (KeyError, TypeError, ValueError, InferenceError):
                    if not allow_partial:
                        raise
                    errors[call.id] = InferenceError("invalid Decision answer", code="invalid_response",
                                                     metadata=metadata, call_id=call.id)
        except (KeyError, TypeError, ValueError, InferenceError):
            raise InferenceError("invalid Decision answers", code="invalid_response", metadata=metadata, call_id=current_call) from None
        return BatchResult(results, metadata, errors)
