"""Canonical ANNO configuration with legacy LEV input compatibility."""

import os


def normalize(values):
    """Normalize one configuration layer before merging higher-priority layers."""
    result = dict(values)
    for name, value in values.items():
        if name.startswith("LEV_"):
            result.setdefault("ANNO_" + name[4:], value)
    return result


def env_value(name, default=None, *, environ=None):
    values = os.environ if environ is None else environ
    if name in values:
        return values[name]
    return values.get("LEV_" + name[5:], default)
