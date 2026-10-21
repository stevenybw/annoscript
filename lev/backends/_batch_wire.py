"""Request-local aliases for the compact LLM batch response protocol."""

FORMAT = "short_ids_values_v1"


def batch_calls(calls):
    """Map wire IDs to unchanged calls; aliases never enter runtime identity."""
    if len({call.id for call in calls}) != len(calls):
        raise ValueError("a batch requires distinct calls")
    return {f"q{index}": call for index, call in enumerate(calls, 1)}
