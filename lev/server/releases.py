"""Immutable execution manifests; credentials stay outside manifests and SQLite."""

import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from lev.job_protocol import decode, encode
from lev._backend_profiles import validate_generation_profile
from lev.backends.base import LLM_BATCH_DEFAULTS, validate_batch_options

RUNTIME = {"max_active_runs": 1, "max_decision_inflight": 4, "max_llm_inflight": 4,
           "max_batch_questions": 32, "loop_window": 32, "local_task_budget": 1024, **LLM_BATCH_DEFAULTS}
POLICY = {"max_attempts": 3, "input_timeout": 300.0, "model_timeout": 60.0,
          "max_inputs": 500, "max_request_bytes": 32 * 1024 * 1024}


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
        required = {"base_url", "model", "revision"}
        allowed_model = required | ({"profile"} if name == "generation" else set())
        if (type(model) is not dict or not required <= set(model) or set(model) - allowed_model
                or any(type(v) is not str or not v.strip() for v in model.values())):
            raise ValueError(f"{name} needs base_url, model and immutable deployment revision")
        url = urlsplit(model["base_url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("model URL must be HTTP(S), without credentials, query or fragment")
        models[name] = model.copy()
        if name == "generation":
            models[name]["profile"] = validate_generation_profile(model.get("profile", "legacy"))
    runtime, policy = {**RUNTIME, **config.get("runtime", {})}, {**POLICY, **config.get("policy", {})}
    validate_batch_options({k: runtime[k] for k in LLM_BATCH_DEFAULTS})
    if (set(runtime) != set(RUNTIME) or any(type(v) is not int or v < 1 for k, v in runtime.items() if k != "llm_batch_repair_attempts")
            or runtime["max_active_runs"] != 1 or set(policy) != set(POLICY)
            or any(type(v) not in (int, float) or not 0 < v < float("inf") for v in policy.values())
            or any(type(policy[k]) is not int for k in ("max_attempts", "max_inputs", "max_request_bytes"))):
        raise ValueError("invalid release runtime or resource policy")
    optimize, thinking = config.get("server_optimize", True), config.get("disable_thinking", True)
    if type(optimize) is not bool or type(thinking) is not bool:
        raise ValueError("release switches must be booleans")
    return {"format": "lev-release", "version": 1, "build_sha256": build_sha256(),
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
        raise RuntimeError("the Lev job service requires Python 3.12+")
    config = decode(Path(path).read_bytes())
    if type(config) is not dict or set(config) != {"releases", "default_release", "auth"}:
        raise ValueError("config requires releases, default_release and auth")
    if type(config["auth"]) is not dict or not config["auth"]:
        raise ValueError("auth must map principals to token environment variable names")
    tokens = {}
    for principal, variable in config["auth"].items():
        token = os.environ.get(variable)
        if not principal or not token or token in tokens:
            raise ValueError("service authentication tokens must be present and unique")
        tokens[token] = principal
    return Releases(config["releases"], config["default_release"]), tokens
