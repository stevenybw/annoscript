"""Opt-in, release-bound Qwen request limits. No network or credential access."""

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

from .base import BackendFatalError


SAMPLING = {"temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0,
            "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0}


def validate_policy(value):
    if value is None:
        return None
    keys = {"sampling", "default_max_tokens", "context_window", "tokenizer", "bounded_outputs"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError("invalid generation request_policy")
    sampling = value["sampling"]
    if (type(sampling) is not dict or set(sampling) != set(SAMPLING)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in sampling.values())
            or sampling != SAMPLING):
        raise ValueError("P0 requires explicit greedy sampling without additional penalties")
    for name in ("default_max_tokens", "context_window"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError("generation limits must be positive integers")
    if value["default_max_tokens"] > value["context_window"] or value["bounded_outputs"] is not True:
        raise ValueError("invalid bounded generation policy")
    binding = value["tokenizer"]
    if (type(binding) is not dict or set(binding) != {"revision", "files"}
            or type(binding["revision"]) is not str or not binding["revision"]
            or type(binding["files"]) is not dict
            or not {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= set(binding["files"])):
        raise ValueError("tokenizer needs an immutable revision and file digests")
    for name, digest in binding["files"].items():
        if (type(name) is not str or Path(name).name != name or name in (".", "..")
                or type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid tokenizer file binding")
    return json.loads(json.dumps(value))


@lru_cache(maxsize=4)
def _load_tokenizer(directory, serialized_binding):
    binding = json.loads(serialized_binding)
    root = Path(directory)
    for name, digest in binding["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("tokenizer artifact digest mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if tokenizer.chat_template != (root / "chat_template.jinja").read_text():
        raise ValueError("loaded chat template differs from bound artifact")
    # Validate the template engine at startup, before any task can send HTTP.
    tokenizer.apply_chat_template([{"role": "system", "content": ""},
                                   {"role": "user", "content": "startup check"}],
                                  tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer


class PinnedTokenCounter:
    def __init__(self, binding):
        try:
            directory = os.environ["LEV_GENERATE_TOKENIZER_DIR"]
            self.tokenizer = _load_tokenizer(directory, json.dumps(binding, sort_keys=True))
        except Exception:
            raise BackendFatalError("pinned tokenizer is unavailable or mismatched",
                                    code="configuration_failed") from None

    def __call__(self, payload):
        try:
            ids = self.tokenizer.apply_chat_template(
                payload["messages"], tokenize=True, return_dict=False, add_generation_prompt=True,
                **payload.get("chat_template_kwargs", {}))
            if type(ids) is not list or any(type(token) is not int for token in ids):
                raise ValueError("template must return one token ID sequence")
            return len(ids)
        except Exception:
            raise BackendFatalError("bound chat template could not tokenize request",
                                    code="configuration_failed") from None


def validate_token_ratio(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("assumed_avg_num_tokens_per_char must be finite and positive")
    return value


def effective_budget(call, assumed_avg_num_tokens_per_char=2, override=None):
    """One rule for direct, runtime, batch and repair budgets; no hidden cap."""
    ratio = validate_token_ratio(assumed_avg_num_tokens_per_char)
    a = call.arguments
    # Direct backend callers also pass through the public argument validator.
    from .base import normalize_call, InferenceContext
    normalize_call(call.id, call.operation, (), dict(a), InferenceContext(a.get("source", "")))
    explicit = a.get("max_num_tokens")
    if explicit is not None:
        budget = explicit
    elif call.operation == "generate":
        budget = max(1, math.ceil(ratio * a["max_length"]))
    else:
        count = a["max_items"]
        budget = math.ceil(ratio * count * a["max_length_per_item"]) + 32 + 8 * count
    if override is not None:
        if type(override) is not int or override < 1:
            raise ValueError("invalid token override")
        budget = min(budget, override)
    return budget
