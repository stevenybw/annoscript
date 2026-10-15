"""Artifact assertions: Python 3.12 is the canonical generation interpreter."""

from dataclasses import fields, is_dataclass
import json
import sys

from lev.ir import SourceSpan


def without_coordinates(value):
    """Ignore only typed source coordinates, never user fields named 'source'."""
    if isinstance(value, SourceSpan):
        return {"module": value.module, "function": value.function}
    if is_dataclass(value):
        result = {f.name: without_coordinates(getattr(value, f.name)) for f in fields(value)}
        if hasattr(value, "kind"):
            result["kind"] = value.kind
        return result
    if isinstance(value, tuple):
        return [without_coordinates(item) for item in value]
    return value


def assert_saved_artifact(test, path, generated):
    text = path.read_text(encoding="utf-8")
    restored = type(generated).from_json(text)
    if sys.version_info[:2] == (3, 12):
        test.assertEqual(text, generated.to_json(indent=2) + "\n")
    else:
        # Serialize to keep bool/int/float distinctions, as in the IR optimizer.
        encode = lambda v: json.dumps(without_coordinates(v), ensure_ascii=False, allow_nan=False)
        test.assertEqual(encode(restored), encode(generated))
    test.assertEqual(generated.to_json(), type(generated).from_json(generated.to_json()).to_json())
