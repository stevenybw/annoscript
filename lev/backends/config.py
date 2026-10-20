"""Explicit environment configuration; credential files are data, not shell code."""

import os
import re
import shlex
from pathlib import Path

from lev._backend_profiles import validate_generation_profile, validate_extract_mode
from .decision import DecisionAPIBackend
from .llm import LLMAPIBackend


def backends_from_env(*, env_file=None, disable_thinking=True, generation_profile=None, extract_mode=None, batch_rule_dedup=False, request_policy=None, assumed_avg_num_tokens_per_char=2, timeout=60.0):
    values = {}
    if env_file is not None:
        for number, line in enumerate(Path(env_file).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
            if not match:
                raise ValueError(f"invalid environment assignment at line {number}")
            try:
                parts = shlex.split(match[2], comments=True)
            except ValueError:
                raise ValueError(f"invalid environment quoting at line {number}") from None
            if len(parts) > 1:
                raise ValueError(f"invalid environment value at line {number}")
            values[match[1]] = parts[0] if parts else ""
    values.update(os.environ)
    profile = validate_generation_profile(
        generation_profile if generation_profile is not None else values.get("LEV_GENERATE_PROFILE", "legacy")
    )
    mode = validate_extract_mode(extract_mode if extract_mode is not None
                                else values.get("LEV_GENERATE_EXTRACT_MODE", "validated"), profile)
    def config(prefix):
        names = [f"LEV_{prefix}_{key}" for key in ("BASE_URL", "AUTH_TOKEN", "MODEL")]
        for name in names:
            if not values.get(name):
                raise ValueError(f"missing environment variable {name}")
        return [values[name] for name in names]
    return (DecisionAPIBackend(*config("DECISION"), timeout=timeout),
            LLMAPIBackend(*config("GENERATE"), timeout=timeout, profile=profile, extract_mode=mode, batch_rule_dedup=batch_rule_dedup, disable_thinking=disable_thinking, request_policy=request_policy, assumed_avg_num_tokens_per_char=assumed_avg_num_tokens_per_char))
