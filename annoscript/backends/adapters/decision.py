"""decision_native_v1: shared state and examples grouped under each criterion."""

from annoscript._data import thaw, freeze
from annoscript._request_adapters import DEFAULT_DECISION_ADAPTER, CONFIGURABLE_DECISION_ADAPTER, native_decision_recipe
from ..base import normalize_arguments, DECISION_OPERATIONS, InferenceError, encode_payload
from .._batch_wire import wire_name


def identified_calls(calls):
    if not calls or len({c.id for c in calls}) != len(calls):
        raise ValueError("a batch requires distinct calls")
    identified = {}
    for call in calls:
        name = wire_name(call, call.id)
        if name in identified:
            raise ValueError(f"duplicate model-visible field name: {name!r}")
        identified[name] = call
    return identified


def question(call):
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


class NativeDecisionAdapter:
    name = DEFAULT_DECISION_ADAPTER
    options = None

    def recipe(self, call):
        return native_decision_recipe()

    def render(self, context, call):
        return thaw(context.state), question(call)

    def batch_payload(self, context, calls, *, model):
        questions, shared, key = {}, None, None
        for name, call in identified_calls(calls).items():
            state, q = self.render(context, call)
            state_key = encode_payload(state)
            if key is not None and key != state_key:
                raise InferenceError("incompatible Decision state views; use plan_batch or preview_batches",
                                     code="incompatible_decision_batch", call_id=call.id,
                                     metadata={"dispatch_status": "not_sent"})
            shared, key = state, state_key
            questions[name] = q
        return {"model": model, "state": shared, "questions": questions}


def project_state(value, projection, call_id):
    value = thaw(value)
    mode = projection["mode"]
    if mode == "full":
        return value
    fields = projection["fields"] if mode == "fields_object" else [projection["field"]]
    if (type(value) is not dict or any(f not in value for f in fields)
            or mode == "field_string" and type(value[fields[0]]) is not str):
        raise InferenceError("state or example does not satisfy its configured projection",
                             code="invalid_state_projection", call_id=call_id,
                             metadata={"dispatch_status": "not_sent", "projection": thaw(projection)})
    return {f: value[f] for f in fields} if mode == "fields_object" else value[fields[0]]


class ConfigurableDecisionAdapter(NativeDecisionAdapter):
    name = CONFIGURABLE_DECISION_ADAPTER

    def __init__(self, options):
        self.options = freeze(options)

    def recipe(self, call):
        return self.options["tasks"].get(call.arguments.get("field_name"), self.options["default"])

    def render(self, context, call):
        recipe = self.recipe(call)
        state = project_state(context.state, recipe["state"], call.id)
        q = question(call)
        criteria = q.get("criteria")
        if criteria is None:
            return state, q

        def criterion(value):
            if type(value) is not dict:
                return value
            if recipe["examples"] == "omit":
                value.pop("examples", None)
            elif "examples" in value:
                value["examples"] = [project_state(v, recipe["state"], call.id) for v in value["examples"]]
            # A noul example may have no description. Do not invent an empty one.
            return value.get("what") if recipe["criteria"] == "text" else value

        if type(criteria) is list:
            q["criteria"] = [criterion(v) for v in criteria]
        else:
            q["criteria"] = {k: result for k, v in criteria.items() if (result := criterion(v)) is not None}
            if not q["criteria"]:
                q.pop("criteria")
        return state, q
