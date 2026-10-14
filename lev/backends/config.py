"""Explicit environment configuration; credential files are data, not shell code."""

import os
import re
import shlex
from pathlib import Path

from .decision import DecisionAPIBackend
from .llm import LLMAPIBackend


def backends_from_env(*, env_file=None, disable_thinking=True, timeout=60.0):
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
    def config(prefix):
        names = [f"LEV_{prefix}_{key}" for key in ("BASE_URL", "AUTH_TOKEN", "MODEL")]
        for name in names:
            if not values.get(name):
                raise ValueError(f"missing environment variable {name}")
        return [values[name] for name in names]
    return (DecisionAPIBackend(*config("DECISION"), timeout=timeout),
            LLMAPIBackend(*config("GENERATE"), timeout=timeout, disable_thinking=disable_thinking))
