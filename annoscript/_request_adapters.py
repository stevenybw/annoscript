"""Versioned request recipes; independent of models, credentials and transport."""

DEFAULT_DECISION_ADAPTER = "decision_native_v1"
DEFAULT_GENERATION_ADAPTER = "generation_chat_v1"
CONFIGURABLE_DECISION_ADAPTER = "decision_configurable_v1"
REQUEST_ADAPTERS = {
    "decision": (DEFAULT_DECISION_ADAPTER, CONFIGURABLE_DECISION_ADAPTER),
    "generation": (DEFAULT_GENERATION_ADAPTER,),
}


def validate_request_adapter(value, stage):
    if stage not in REQUEST_ADAPTERS:
        raise ValueError("request adapter stage must be decision or generation")
    if type(value) is not str or value not in REQUEST_ADAPTERS[stage]:
        raise ValueError(f"unsupported {stage} request_adapter; expected " +
                         ", ".join(REQUEST_ADAPTERS[stage]))
    return value


def native_decision_recipe():
    return {"criteria": "preserve", "examples": "preserve", "state": {"mode": "full"}}


def normalize_decision_options(adapter, options=None):
    """Detach and validate JSON configuration before any transport is initialized."""
    from annoscript._data import snapshot
    validate_request_adapter(adapter, "decision")
    if options is not None and type(options) is not dict:
        raise ValueError("request_adapter_options must be an object")
    if adapter == DEFAULT_DECISION_ADAPTER:
        if options:
            raise ValueError("decision_native_v1 does not accept request_adapter_options")
        return None
    options = snapshot(options) if options is not None else {}
    if set(options) - {"default", "tasks"}:
        raise ValueError("request_adapter_options only accepts default and tasks")

    def recipe(value):
        if type(value) is not dict or set(value) != {"criteria", "examples", "state"}:
            raise ValueError("a task recipe requires criteria, examples and state")
        if value["criteria"] not in ("preserve", "text") or type(value["criteria"]) is not str:
            raise ValueError("criteria must be preserve or text")
        if value["examples"] not in ("preserve", "omit") or type(value["examples"]) is not str:
            raise ValueError("examples must be preserve or omit")
        if value["criteria"] == "text" and value["examples"] == "preserve":
            raise ValueError("text criteria cannot preserve nested examples")
        state = value["state"]
        if type(state) is not dict:
            raise ValueError("state projection must be an object")
        mode = state.get("mode")
        if mode == "full" and set(state) == {"mode"}:
            pass
        elif mode == "fields_object" and set(state) == {"mode", "fields"}:
            fields = state["fields"]
            if (type(fields) is not list or not fields
                    or any(type(f) is not str or not f.strip() for f in fields)
                    or len(set(fields)) != len(fields)):
                raise ValueError("fields_object needs distinct nonblank fields in order")
        elif mode == "field_string" and set(state) == {"mode", "field"}:
            if type(state["field"]) is not str or not state["field"].strip():
                raise ValueError("field_string needs a nonblank field")
        else:
            raise ValueError("invalid state projection")
        return {"criteria": value["criteria"], "examples": value["examples"], "state": state}

    tasks = options.get("tasks", {})
    if type(tasks) is not dict or any(type(k) is not str or not k.strip() for k in tasks):
        raise ValueError("tasks must map exact nonblank field_name values to recipes")
    return {"default": recipe(options.get("default", native_decision_recipe())),
            "tasks": {key: recipe(tasks[key]) for key in sorted(tasks)}}
