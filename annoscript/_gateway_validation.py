"""Pure operation validation shared by gateway and base-only clients."""
import json


def validate_operation(data):
    if set(data) - {"state", "system_prompt", "calls", "max_tokens", "repair_feedback"}:
        raise ValueError("unknown fields")
    from annoscript.backends.base import InferenceContext, normalize_call, DECISION_OPERATIONS
    context = InferenceContext(data["state"], data.get("system_prompt"))
    if context.system_prompt and len(context.system_prompt) > 4000:
        raise ValueError("system prompt too large")
    if len(json.dumps(data["state"], ensure_ascii=False).encode()) > 16384:
        raise ValueError("state too large")
    rows = data["calls"]
    if type(rows) is not list or not 1 <= len(rows) <= 32:
        raise ValueError("invalid calls")
    calls = []
    for row in rows:
        if type(row) is not dict or set(row) != {"id", "operation", "arguments"}:
            raise ValueError("invalid call")
        if type(row["id"]) is not str or not 1 <= len(row["id"]) <= 100:
            raise ValueError("invalid id")
        call = normalize_call(row["id"], row["operation"], (), row["arguments"], context)
        args = call.arguments
        if len(args["instructions"]) > 4000:
            raise ValueError("instructions too large")
        criteria = args.get("criteria")
        if criteria is not None and len(criteria) > 32:
            raise ValueError("too many options")
        for key in ("max_items", "max_length", "max_length_per_item"):
            if args.get(key) is not None and args[key] > (16 if key == "max_items" else 2000):
                raise ValueError("operation bound exceeded")
        calls.append(call)
    if len({c.id for c in calls}) != len(calls):
        raise ValueError("duplicate id")
    decision = all(c.operation in DECISION_OPERATIONS for c in calls)
    if not decision and any(c.operation in DECISION_OPERATIONS for c in calls):
        raise ValueError("mixed backends")
    budget = data.get("max_tokens", 1024)
    if type(budget) is not int or not 1 <= budget <= 4096:
        raise ValueError("invalid token budget")
    feedback = data.get("repair_feedback")
    if feedback is not None and (type(feedback) is not str or len(feedback) > 1000 or len(calls) != 1):
        raise ValueError("invalid repair")
    return context, calls, decision, budget, feedback
