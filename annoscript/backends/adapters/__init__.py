"""Pure request construction. Transport, response checks and scheduling stay outside."""

from annoscript._request_adapters import validate_request_adapter


def decision_adapter(name, options=None):
    validate_request_adapter(name, "decision")
    from annoscript._request_adapters import DEFAULT_DECISION_ADAPTER, normalize_decision_options
    from .decision import NativeDecisionAdapter, ConfigurableDecisionAdapter
    normalized = normalize_decision_options(name, options)
    return NativeDecisionAdapter() if name == DEFAULT_DECISION_ADAPTER else ConfigurableDecisionAdapter(normalized)


def generation_adapter(name):
    validate_request_adapter(name, "generation")
    from .chat import ChatGenerationAdapter
    return ChatGenerationAdapter()
