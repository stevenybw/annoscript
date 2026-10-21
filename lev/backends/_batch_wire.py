"""Request-local aliases for the compact LLM batch response protocol."""

FORMAT = "short_ids_values_v1"


def wire_name(call, fallback):
    """Resolve an optional public name without replacing runtime identity."""
    name = call.arguments.get("field_name")
    if name is None:
        return fallback
    if type(name) is not str or not name.strip():
        raise ValueError("field_name must be a nonblank string or None")
    return name


def batch_calls(calls):
    """Map wire IDs to unchanged calls; aliases never enter runtime identity."""
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("a batch requires distinct calls")
    identified = {}
    for index, call in enumerate(calls, 1):
        name = wire_name(call, f"q{index}")
        if name in identified:
            raise ValueError(f"duplicate model-visible field name: {name!r}")
        identified[name] = call
    return identified
