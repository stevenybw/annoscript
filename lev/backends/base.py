"""Inference contracts, validation and shared asynchronous HTTP transport."""

from dataclasses import dataclass, field
from collections.abc import Mapping
import inspect
import json
import math
import time
import uuid

from lev._data import freeze, thaw, snapshot
from lev.lev import (Example, Lev, NoulResponse, ChoiceResponse, ScoreResponse,
                     ExtractResponse, GenerateResponse, GenerateListResponse)


DECISION_OPERATIONS = frozenset(("noul", "choice", "score"))
LLM_OPERATIONS = frozenset(("extract", "generate", "generate_list"))
LLM_BATCH_DEFAULTS = {
    "max_llm_batch_calls": 8,
    "max_llm_batch_input_bytes": 262144,
    "max_llm_batch_output_tokens": 16384,
    "llm_batch_default_tokens": 1024,
    "llm_batch_repair_attempts": 1,
}
RESPONSE_TYPES = dict(zip(
    ("noul", "choice", "score", "extract", "generate", "generate_list"),
    (NoulResponse, ChoiceResponse, ScoreResponse, ExtractResponse, GenerateResponse, GenerateListResponse),
))


class InferenceError(Exception):
    """A request-local model or response failure, safe for user diagnostics."""
    def __init__(self, message, *, code="inference_failed", metadata=None, call_id=None):
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}
        self.call_id = call_id


class BackendFatalError(InferenceError):
    """Configuration/authentication/connectivity failure of a submission."""


@dataclass(frozen=True)
class InferenceContext:
    state: object
    system_prompt: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "state", freeze(snapshot(thaw(self.state), state=True)))
        if self.system_prompt is not None and type(self.system_prompt) is not str:
            raise ValueError("system_prompt must be a string")


@dataclass(frozen=True)
class InferenceCall:
    id: str
    operation: str
    arguments: Mapping
    # Resolved at normalization, never included in public arguments or messages.
    extraction_source: str | None = field(default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "arguments", freeze(snapshot(dict(self.arguments))))


@dataclass(frozen=True)
class InferenceOutput:
    response: object
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    responses: Mapping[str, object]
    metadata: dict = field(default_factory=dict)
    errors: Mapping[str, InferenceError] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedLLMRequest:
    """Internal, attempt-local immutable transport and preflight snapshot."""
    binding: str
    payload: object
    body: bytes
    evidence: object


@dataclass(frozen=True)
class LLMBatchPlan:
    calls: tuple
    max_tokens: int
    input_bytes: int
    prepared: PreparedLLMRequest | None = field(default=None, repr=False, compare=False)


def encode_payload(payload):
    """The exact JSON encoding used at the HTTP boundary and by batch planning."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def validate_batch_options(options):
    for name, value in options.items():
        if type(value) is not int or (value not in (0, 1) if name == "llm_batch_repair_attempts" else value < 1):
            raise ValueError(f"invalid {name}")


def normalize_call(call_id, operation, args, kwargs, context):
    values = normalize_arguments(operation, args, kwargs)
    source = None
    if operation == "extract":
        source = extraction_source(context.state, values["source_field"])
    return InferenceCall(call_id, operation, values, source)


def extraction_source(state, name):
    if type(state) is str:
        if name is not None:
            raise ValueError("string state requires source_field=None")
        return state
    if not isinstance(state, Mapping) or name is None:
        raise ValueError("object state requires an explicit extract source_field")
    if name not in state or type(state[name]) is not str:
        raise ValueError("extract source_field must name an existing string field")
    return state[name]


def normalize_examples(operation, values):
    examples = values.pop("examples")
    if examples is None:
        return
    if type(examples) is not list:
        raise ValueError("examples must be a list")
    normalized = []
    for index, example in enumerate(examples):
        if type(example) is Example:
            example = example.to_dict()
        if type(example) is not dict or set(example) != {"state", "answer"}:
            raise ValueError(f"examples[{index}] requires exactly state and answer")
        example = snapshot(example)
        snapshot(example["state"], state=True)
        answer = example["answer"]
        if operation == "choice":
            valid = type(answer) is str and answer in values["criteria"]
        elif operation == "noul":
            valid = type(answer) is bool
        elif operation == "score":
            valid = type(answer) is int and 0 <= answer < len(values["criteria"])
        else:
            source = (extraction_source(example["state"], values["source_field"])
                      if operation == "extract" else None)
            call = InferenceCall(f"example-{index}", operation, values, source)
            try:
                validate_response(call, RESPONSE_TYPES[operation](answer))
                valid = True
            except InferenceError:
                valid = False
        if not valid:
            raise ValueError(f"examples[{index}].answer violates {operation} contract")
        normalized.append(example)
    if normalized:
        values["examples"] = normalized


def normalize_arguments(operation, args, kwargs):
    """Validate public arguments without manufacturing a context for budget checks."""
    if operation not in RESPONSE_TYPES:
        raise ValueError("unsupported inference operation")
    if operation == "extract" and "source" in kwargs:
        raise ValueError("extract source was removed; use source_field with state")
    bound = inspect.signature(getattr(Lev, operation)).bind(None, *args, **kwargs)
    bound.apply_defaults()
    values = {k: v for k, v in bound.arguments.items() if k != "self"}
    if operation == "generate":
        if type(values["nullable"]) is not bool:
            raise ValueError("nullable must be a boolean")
        if not values["nullable"]:
            # Keep default calls and their cache bindings unchanged.
            values.pop("nullable")
    if "field_name" in values:
        name = values["field_name"]
        if name is None:
            # Preserve the normalized arguments of existing, unnamed programs.
            values.pop("field_name")
        elif type(name) is not str or not name.strip():
            raise ValueError("field_name must be a nonblank string or None")
    if type(values["instructions"]) is not str:
        raise ValueError("instructions must be a string")
    requirements = values.pop("requirements")
    if requirements is not None:
        if type(requirements) is not list or any(
            type(value) is not str or not value.strip() for value in requirements
        ):
            raise ValueError("requirements must be a list of nonblank strings or None")
        if requirements:
            values["requirements"] = requirements
    if operation == "noul":
        for key in ("criteria_false", "criteria_true"):
            if values[key] is not None and type(values[key]) is not str:
                raise ValueError(f"{key} must be a string or None")
    if operation == "choice":
        criteria = values["criteria"]
        if type(criteria) is not dict or not criteria or any(type(k) is not str or type(v) is not str for k, v in criteria.items()):
            raise ValueError("choice criteria must be a nonempty string mapping")
    if operation == "score":
        if type(values["criteria"]) is not list or not values["criteria"] or any(type(v) is not str for v in values["criteria"]):
            raise ValueError("score criteria must be a nonempty list of strings")
    for key, value in values.items():
        if key.startswith(("min_", "max_")):
            optional = key == "max_num_tokens"
            positive = key == "max_num_tokens" or (key == "max_length_per_item" and operation == "extract")
            if value is None and optional:
                continue
            if type(value) is not int or value < (1 if positive else 0):
                raise ValueError(f"invalid {key}")
    for low, high in (("min_items", "max_items"), ("min_length", "max_length"), ("min_length_per_item", "max_length_per_item")):
        if low in values and values.get(high) is not None and values[low] > values[high]:
            raise ValueError(f"{low} exceeds {high}")
    if operation == "extract":
        if values["source_field"] is not None and type(values["source_field"]) is not str:
            raise ValueError("extract source_field must be a string or None")
    normalize_examples(operation, values)
    return values


def validate_response(call, response):
    a, op = call.arguments, call.operation
    def require(condition, message):
        if not condition:
            raise InferenceError(message, code="invalid_response", call_id=call.id)
    def number(x, low=None, high=None):
        return type(x) in (int, float) and (type(x) is int or math.isfinite(x)) and (low is None or x >= low) and (high is None or x <= high)
    def length(value, minimum, maximum):
        return len(value) >= minimum and (maximum is None or len(value) <= maximum)
    require(type(response) is RESPONSE_TYPES[op], "incorrect response type")
    if op == "noul":
        require(number(response.noul, 0, 1), "invalid noul probability")
    elif op in ("choice", "score"):
        keys = set(a["criteria"]) if op == "choice" else set(range(len(a["criteria"])))
        require(type(response.probabilities) is dict and set(response.probabilities) == keys and
                all(type(k) is (str if op == "choice" else int) for k in response.probabilities) and
                all(number(v, 0, 1) for v in response.probabilities.values()), "invalid probabilities")
        require(number(response.confidence, 0, 1), "invalid confidence")
        if op == "choice":
            require(type(response.choice) is str and response.choice in keys, "choice is outside criteria")
        else:
            require(number(response.score, 0, len(keys)-1), "invalid score")
            require(response.legend == dict(enumerate(a["criteria"])), "incorrect score legend")
    elif op == "generate":
        require((response.text is None and a.get("nullable", False)) or
                (type(response.text) is str and (not a["min_length"] or bool(response.text))),
                "generated text must satisfy its string/null and nonempty contract" if a.get("nullable", False) else
                "generated text must be a string and satisfy nonempty requirement")
    else:
        require(type(response.items) is list and all(type(x) is str for x in response.items), "items must be a list of strings")
        require(length(response.items, a["min_items"], a["max_items"]), "items violate count limits")
        for item in response.items:
            if op == "extract":
                require(type(call.extraction_source) is str and bool(item) and item in call.extraction_source,
                        "extraction is not a nonempty verbatim source substring")
            else:
                require(not a["min_length_per_item"] or bool(item), "generated item violates nonempty requirement")
    return response


class HTTPBackend:
    def __init__(self, base_url, auth_token, model, *, timeout=60.0, transport=None,
                 diagnostic_observer=None):
        import httpx
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError("backend URL must be an HTTP(S) address without credentials, query or fragment")
        if type(model) is not str or not model or type(auth_token) is not str or not auth_token:
            raise ValueError("model and authentication token are required")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url, self.model = base_url.rstrip("/"), model
        self._token, self.timeout, self._transport = auth_token, timeout, transport
        self._client = None
        self.diagnostic_observer = diagnostic_observer
        self._diagnostic_calls = {}

    def _diagnose(self, event):
        if self.diagnostic_observer is not None:
            try:
                self.diagnostic_observer({"model": self.model, **event})
            except Exception:
                raise BackendFatalError("diagnostic persistence failed", code="diagnostics_failed") from None

    async def post(self, url, payload, *, logical_calls=None, diagnostic_context=None, _encoded_body=None):
        guard = getattr(self, "_dispatch_guard", None)
        if guard is not None:
            guard()
        body_args = {"_encoded_body": _encoded_body} if _encoded_body is not None else {}
        meter = getattr(self, "_request_meter", None)
        observing = self.diagnostic_observer is not None
        if meter is None and not observing:
            return await self._post(url, payload, **body_args)
        questions = logical_calls if logical_calls is not None else len(payload["questions"]) if "questions" in payload else 1
        identity = await meter.start(questions) if meter is not None else uuid.uuid4().hex
        timing, metadata = {}, {}
        diagnostic = {"request_id": identity} if observing else None
        try:
            if observing:
                context = dict(diagnostic_context or {})
                calls = context.get("calls", [])
                parents = {self._diagnostic_calls[c["id"]] for c in calls
                           if context.get("repair") and c["id"] in self._diagnostic_calls}
                self._diagnose({"event": "request", "request_id": identity, "started_unix": time.time(),
                                "body": _encoded_body if _encoded_body is not None else encode_payload(payload), "questions": questions,
                                "repair_parent_request_ids": sorted(parents), **context})
                for call in calls:
                    self._diagnostic_calls[call["id"]] = identity
            data, metadata = await self._post(url, payload, _timing=timing, _diagnostic=diagnostic, **body_args)
            if observing:
                metadata["request_id"] = identity
            return data, metadata
        except InferenceError as exc:
            metadata = exc.metadata
            raise
        finally:
            # Timing covers the HTTP await, not lazy client construction, JSON
            # parsing or metric persistence. No finished metric is invented if
            # execution never reached the transport boundary.
            if "seconds" in timing:
                try:
                    if meter is not None:
                        await meter.finish(identity, timing["seconds"], metadata.get("http_status", timing.get("http_status")), metadata.get("usage"))
                finally:
                    if observing:
                        self._diagnose({"event": "outcome", "request_id": identity,
                                        "seconds": timing["seconds"], "finished_unix": time.time(),
                                        "http_status": metadata.get("http_status", timing.get("http_status")),
                                        "usage": metadata.get("usage"),
                                        "response_received": diagnostic.get("response_received", False)})
            if observing and diagnostic.get("write_error"):
                raise BackendFatalError("diagnostic persistence failed", code="diagnostics_failed") from None

    async def _post(self, url, payload, *, _timing=None, _diagnostic=None, _encoded_body=None):
        import asyncio
        import httpx
        if self._client is None:
            auth = self._token if self._token.lower().startswith("bearer ") else "Bearer " + self._token
            self._client = httpx.AsyncClient(headers={"Authorization": auth},
                timeout=self.timeout, transport=self._transport or httpx.AsyncHTTPTransport(retries=0))
        guard = getattr(self, "_dispatch_guard", None)
        if guard is not None:
            guard()
        try:
            clock = getattr(getattr(self, "_request_meter", None), "clock", time.monotonic)
            started = clock() if _timing is not None else None
            try:
                async with asyncio.timeout(self.timeout):
                    response = await self._client.post(url, content=_encoded_body if _encoded_body is not None else encode_payload(payload),
                                                       headers={"Content-Type": "application/json"})
                if _timing is not None:
                    _timing["http_status"] = response.status_code
            finally:
                if _timing is not None:
                    _timing["seconds"] = max(0.0, clock() - started)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise BackendFatalError("backend connection failed", code="backend_unavailable") from None
        except (TimeoutError, httpx.TimeoutException):
            raise InferenceError("model request timed out", code="model_timeout") from None
        except httpx.RequestError:
            raise InferenceError("model transport failed", code="transport_error") from None
        if _diagnostic is not None:
            _diagnostic["response_received"] = True
            try:
                self._diagnose({"event": "response", "request_id": _diagnostic["request_id"],
                                "finished_unix": time.time(), "http_status": response.status_code,
                                "body": response.content, "complete": True})
            except BackendFatalError:
                # Preserve numeric usage and HTTP status even if the raw sink fails.
                _diagnostic["write_error"] = True
        metadata = {"http_status": response.status_code}
        if "Retry-After" in response.headers:
            metadata["retry_after"] = response.headers["Retry-After"]
        try:
            data = json.loads(response.content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if type(data) is not dict:
                data = None
        except (ValueError, UnicodeError):
            data = None
        # Numerical usage only: no provider body, prompts or reasoning text.
        def usage(value):
            if type(value) is dict:
                return {k: usage(v) for k, v in value.items() if type(v) in (dict, int, float)}
            return value if type(value) is int or math.isfinite(value) else None
        if data is not None and (response.status_code < 400 or type(data.get("usage")) is dict):
            metadata["usage"] = usage(data.get("usage", {})) if type(data.get("usage")) is dict else {}
        if response.status_code in (401, 403):
            raise BackendFatalError("backend authentication failed", code="authentication_failed", metadata=metadata)
        if response.status_code >= 400:
            error = data.get("error") if isinstance(data, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            if response.status_code == 400 and isinstance(message, str) and (
                    "System message must be at the beginning." in message):
                # Recognize a deployment protocol error without logging the
                # provider body. Context/input errors remain request-local.
                raise BackendFatalError("backend chat template rejected system message layout",
                                        code="backend_protocol_error", metadata=metadata)
            raise InferenceError(f"model HTTP error {response.status_code}", code="model_http_error", metadata=metadata)
        if data is None:
            raise InferenceError("backend returned invalid JSON", code="invalid_response", metadata=metadata)
        return data, metadata

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
