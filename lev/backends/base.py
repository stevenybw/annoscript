"""Inference contracts, validation and shared asynchronous HTTP transport."""

from dataclasses import dataclass, field
from collections.abc import Mapping
import inspect
import json
import math

from lev._data import freeze, thaw, snapshot
from lev.lev import (Lev, NoulResponse, ChoiceResponse, ScoreResponse,
                     ExtractResponse, GenerateResponse, GenerateListResponse)


DECISION_OPERATIONS = frozenset(("noul", "choice", "score"))
LLM_OPERATIONS = frozenset(("extract", "generate", "generate_list"))
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


def normalize_call(call_id, operation, args, kwargs, context):
    if operation not in RESPONSE_TYPES:
        raise ValueError("unsupported inference operation")
    bound = inspect.signature(getattr(Lev, operation)).bind(None, *args, **kwargs)
    bound.apply_defaults()
    values = {k: v for k, v in bound.arguments.items() if k != "self"}
    if type(values["instructions"]) is not str:
        raise ValueError("instructions must be a string")
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
            optional = key.startswith("max_")
            positive = key == "max_num_tokens" or (key == "max_length_per_item" and operation == "extract")
            if value is None and optional:
                continue
            if type(value) is not int or value < (1 if positive else 0):
                raise ValueError(f"invalid {key}")
    for low, high in (("min_items", "max_items"), ("min_length", "max_length"), ("min_length_per_item", "max_length_per_item")):
        if low in values and values.get(high) is not None and values[low] > values[high]:
            raise ValueError(f"{low} exceeds {high}")
    if operation == "extract":
        if values["source"] is None:
            if type(context.state) is not str:
                raise ValueError("extract requires source for structured state")
            values["source"] = context.state
        if type(values["source"]) is not str:
            raise ValueError("extract source must be a string")
    return InferenceCall(call_id, operation, values)


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
        require(type(response.text) is str and length(response.text, a["min_length"], a["max_length"]), "generated text violates length limits")
    else:
        require(type(response.items) is list and all(type(x) is str for x in response.items), "items must be a list of strings")
        require(length(response.items, a["min_items"], a["max_items"]), "items violate count limits")
        for item in response.items:
            if op == "extract":
                require(bool(item) and item in a["source"], "extraction is not a nonempty verbatim source substring")
                require(a["max_length_per_item"] is None or len(item) <= a["max_length_per_item"], "extraction violates length limit")
            else:
                require(length(item, a["min_length_per_item"], a["max_length_per_item"]), "generated item violates length limits")
    return response


class HTTPBackend:
    def __init__(self, base_url, auth_token, model, *, timeout=60.0, transport=None):
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

    async def post(self, url, payload):
        meter = getattr(self, "_request_meter", None)
        if meter is None:
            return await self._post(url, payload)
        identity = await meter.start(len(payload["questions"]) if "questions" in payload else 1)
        timing, metadata = {}, {}
        try:
            data, metadata = await self._post(url, payload, _timing=timing)
            return data, metadata
        except InferenceError as exc:
            metadata = exc.metadata
            raise
        finally:
            # Timing covers the HTTP await, not lazy client construction, JSON
            # parsing or metric persistence. No finished metric is invented if
            # execution never reached the transport boundary.
            if "seconds" in timing:
                await meter.finish(identity, timing["seconds"], metadata.get("http_status", timing.get("http_status")), metadata.get("usage"))

    async def _post(self, url, payload, *, _timing=None):
        import asyncio
        import httpx
        if self._client is None:
            auth = self._token if self._token.lower().startswith("bearer ") else "Bearer " + self._token
            self._client = httpx.AsyncClient(headers={"Authorization": auth},
                timeout=self.timeout, transport=self._transport or httpx.AsyncHTTPTransport(retries=0))
        try:
            started = self._request_meter.clock() if _timing is not None else None
            try:
                async with asyncio.timeout(self.timeout):
                    response = await self._client.post(url, json=payload)
                if _timing is not None:
                    _timing["http_status"] = response.status_code
            finally:
                if _timing is not None:
                    _timing["seconds"] = max(0.0, self._request_meter.clock() - started)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise BackendFatalError("backend connection failed", code="backend_unavailable") from None
        except (TimeoutError, httpx.TimeoutException):
            raise InferenceError("model request timed out", code="model_timeout") from None
        except httpx.RequestError:
            raise InferenceError("model transport failed", code="transport_error") from None
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
            raise InferenceError(f"model HTTP error {response.status_code}", code="model_http_error", metadata=metadata)
        if data is None:
            raise InferenceError("backend returned invalid JSON", code="invalid_response", metadata=metadata)
        return data, metadata

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
