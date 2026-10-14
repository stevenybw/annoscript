"""JSON snapshots and immutable inference payloads; no application code runs here."""

import math
from types import MappingProxyType
from collections.abc import Mapping


def snapshot(value, path="$", *, state=False):
    if state and type(value) not in (str, list, dict):
        raise ValueError(f"{path}: input must be a string, list or object")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [snapshot(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError(f"{path}: object keys must be strings")
        return {k: snapshot(v, f"{path}.{k}") for k, v in value.items()}
    raise ValueError(f"{path}: expected finite JSON data")


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value
