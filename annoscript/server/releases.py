"""Immutable execution manifests; credentials stay outside manifests and SQLite."""

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import sys
from annoscript._env import env_value
from urllib.parse import urlsplit

from annoscript.job_protocol import decode, encode
from annoscript._backend_profiles import validate_generation_profile, validate_extract_mode
from annoscript._request_adapters import (DEFAULT_DECISION_ADAPTER, DEFAULT_GENERATION_ADAPTER,
                                   validate_request_adapter, normalize_decision_options)
from annoscript.backends.base import LLM_BATCH_DEFAULTS, validate_batch_options
from annoscript.backends._generation_policy import validate_policy, validate_token_ratio
from annoscript.server._plan_cache import MAX_ENTRIES, MAX_BYTES
from annoscript.backends._shared_rules import validate_rule_dedup

RUNTIME = {"max_active_runs": 1, "max_decision_inflight": 4, "max_llm_inflight": 4,
           "max_batch_questions": 32, "loop_window": 32, "local_task_budget": 1024, **LLM_BATCH_DEFAULTS}
POLICY = {"max_attempts": 3, "input_timeout": 300.0, "model_timeout": 60.0,
          "max_inputs": 500, "max_request_bytes": 32 * 1024 * 1024,
          "retry_unknown_outcomes": True}


def build_sha256():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith(("test_", "_test_")):
            continue
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def manifest(config):
    allowed = {"decision", "generation", "server_optimize", "runtime", "policy", "disable_thinking"}
    if type(config) is not dict or set(config) - allowed:
        raise ValueError("invalid release configuration")
    models = {}
    for name in ("decision", "generation"):
        model = config.get(name)
        if name == "generation" and isinstance(model, dict) and "source_layout" in model:
            raise ValueError("source_layout was removed; extract now uses source_field")
        required = {"base_url", "model", "revision"}
        allowed_model = required | {"request_adapter"} | ({"profile", "extract_mode", "batch_rule_dedup", "structured_output_revision", "request_policy", "assumed_avg_num_tokens_per_char"} if name == "generation" else {"request_adapter_options"})
        if (type(model) is not dict or not required <= set(model) or set(model) - allowed_model
                or any(type(v) is not str or not v.strip() for k, v in model.items() if k not in ("request_policy", "assumed_avg_num_tokens_per_char", "batch_rule_dedup", "request_adapter_options"))):
            raise ValueError(f"{name} needs base_url, model and immutable deployment revision")
        url = urlsplit(model["base_url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("model URL must be HTTP(S), without credentials, query or fragment")
        models[name] = model.copy()
        models[name]["request_adapter"] = validate_request_adapter(
            model.get("request_adapter", DEFAULT_DECISION_ADAPTER if name == "decision" else DEFAULT_GENERATION_ADAPTER), name)
        if name == "decision":
            if "request_adapter_options" in model and type(model["request_adapter_options"]) is not dict:
                raise ValueError("request_adapter_options must be an object")
            options = normalize_decision_options(models[name]["request_adapter"], model.get("request_adapter_options"))
            if options is not None:
                models[name]["request_adapter_options"] = options
            else:
                models[name].pop("request_adapter_options", None)
        if name == "generation":
            models[name]["assumed_avg_num_tokens_per_char"] = validate_token_ratio(model.get("assumed_avg_num_tokens_per_char", 2))
            models[name]["profile"] = validate_generation_profile(model.get("profile", "legacy"))
            if "batch_rule_dedup" in model:
                models[name]["batch_rule_dedup"] = validate_rule_dedup(model["batch_rule_dedup"], models[name]["profile"])
            # Omitted mode keeps existing release hashes stable for old manifests.
            mode = validate_extract_mode(model.get("extract_mode", "validated"), models[name]["profile"])
            if "extract_mode" in model:
                models[name]["extract_mode"] = mode
            if mode == "substring_chars" and not model.get("structured_output_revision"):
                raise ValueError("substring_chars requires immutable structured_output_revision")
            if "request_policy" in model:
                request_policy = validate_policy(model["request_policy"])
                if request_policy is None or models[name]["profile"] != "qwen_vllm":
                    raise ValueError("request_policy requires qwen_vllm")
                if request_policy["tokenizer"]["revision"] != model["revision"]:
                    raise ValueError("tokenizer revision differs from generation revision")
                if not model.get("structured_output_revision"):
                    raise ValueError("bounded outputs require immutable decoder build identity")
                models[name]["request_policy"] = request_policy
    runtime, policy = {**RUNTIME, **config.get("runtime", {})}, {**POLICY, **config.get("policy", {})}
    validate_batch_options({k: runtime[k] for k in LLM_BATCH_DEFAULTS})
    if (set(runtime) != set(RUNTIME) or any(type(v) is not int or v < 1 for k, v in runtime.items() if k != "llm_batch_repair_attempts")
            or runtime["max_active_runs"] != 1 or set(policy) != set(POLICY)
            or type(policy["retry_unknown_outcomes"]) is not bool
            or any(type(v) not in (int, float) or not 0 < v < float("inf")
                   for k, v in policy.items() if k != "retry_unknown_outcomes")
            or any(type(policy[k]) is not int for k in ("max_attempts", "max_inputs", "max_request_bytes"))):
        raise ValueError("invalid release runtime or resource policy")
    generation_policy = models["generation"].get("request_policy")
    optimize, thinking = config.get("server_optimize", True), config.get("disable_thinking", True)
    if type(optimize) is not bool or type(thinking) is not bool:
        raise ValueError("release switches must be booleans")
    return {"format": "lev-release", "version": 1, "build_sha256": build_sha256(),
            "engineering": {"plan_cache": {"max_entries": MAX_ENTRIES, "max_serialized_bytes": MAX_BYTES,
                                            "compiler_threads": 1, "key": "release_id+canonical_ir_sha256"},
                            "qwen_source_layout": "source_field_v1", "request_preparation": "immutable_v1",
                            "worker_refill": "completion_event_v1", "shared_model_slots": "ready_at_admission_fifo_v1",
                            "shared_slot_scope": "single_worker_process; limits bound by execution manifest"},
            "python": f"{sys.version_info.major}.{sys.version_info.minor}", "ir_version": 1,
            "dependencies": {name: version(name) for name in ("httpx", "httpcore", "fastapi", "starlette", "uvicorn")},
            **models, "server_optimize": optimize, "disable_thinking": thinking,
            "runtime": runtime, "policy": policy}


class Releases:
    def __init__(self, configs, default):
        self.manifests, self.aliases = {}, {}
        for alias, config in configs.items():
            value = manifest(config)
            release_id = hashlib.sha256(encode(value, canonical=True)).hexdigest()
            self.manifests[release_id] = value
            self.aliases[alias] = release_id
        if default not in self.aliases:
            raise ValueError("default_release must name a configured release")
        self.default = self.aliases[default]

    def resolve(self, release_id=None):
        release_id = release_id or self.default
        if release_id not in self.manifests:
            raise KeyError(release_id)
        return release_id, self.manifests[release_id]


def load_config(path):
    if sys.version_info < (3, 12):
        raise RuntimeError("the AnnoScript job service requires Python 3.12+")
    config = decode(Path(path).read_bytes())
    if type(config) is not dict or set(config) != {"releases", "default_release", "auth"}:
        raise ValueError("config requires releases, default_release and auth")
    if type(config["auth"]) is not dict or not config["auth"]:
        raise ValueError("auth must map principals to token environment variable names")
    tokens = {}
    for principal, variable in config["auth"].items():
        token = env_value(variable) if variable.startswith("ANNO_") else os.environ.get(variable)
        if not principal or not token or token in tokens:
            raise ValueError("service authentication tokens must be present and unique")
        tokens[token] = principal
    return Releases(config["releases"], config["default_release"]), tokens
