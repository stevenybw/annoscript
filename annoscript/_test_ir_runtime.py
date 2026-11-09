"""Test-only interpreter for serialized IR; validates, never repairs constraints.

Deliberately consumes the JSON representation, independently of compiler classes.
It is not a supported execution API or a production solver.
"""

import json
import operator
from copy import deepcopy


class _Returned(Exception):
    def __init__(self, value):
        self.value = value


_COMPARE = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
    "gt": operator.gt, "ge": operator.ge, "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b, "is": operator.is_, "is_not": operator.is_not,
}


def evaluate(program, anno):
    """Evaluate JSON IR against a supplied deterministic fake AnnoScript."""
    env = {}

    def expr(node):
        kind = node["kind"]
        if kind == "literal":
            return node["value"]
        if kind == "var":
            return env[node["name"]]
        if kind == "state":
            return anno.state
        if kind == "get_item":
            return expr(node["value"])[expr(node["index"])]
        if kind == "field":
            return getattr(expr(node["value"]), node["name"])
        if kind in ("list", "tuple"):
            items = [expr(item) for item in node["items"]]
            return items if kind == "list" else tuple(items)
        if kind == "dict":
            return {expr(key): expr(value) for key, value in node["entries"]}
        if kind == "unary":
            return {"not": operator.not_, "positive": operator.pos, "negative": operator.neg}[node["op"]](expr(node["value"]))
        if kind == "binary":
            assert node["op"] == "add"
            return expr(node["left"]) + expr(node["right"])
        if kind == "compare":
            left = expr(node["left"])
            for op, right_node in zip(node["ops"], node["comparators"]):
                right = expr(right_node)
                if not _COMPARE[op](left, right):
                    return False
                left = right
            return True
        if kind == "bool_op":
            value = expr(node["values"][0])
            for item in node["values"][1:]:
                if (node["op"] == "and" and not value) or (node["op"] == "or" and value):
                    break
                value = expr(item)
            return value
        if kind == "intrinsic":
            functions = {"len": len, "int": int, "range": range, "enumerate": enumerate,
                         "str.strip": str.strip, "json.dumps": json.dumps,
                         "example": lambda state, answer: deepcopy({"state": state, "answer": answer})}
            args = [expr(arg) for arg in node["args"]]
            kwargs = {key: expr(value) for key, value in node["kwargs"]}
            return functions[node["name"]](*args, **kwargs)
        if kind == "format_value":
            value = expr(node["value"])
            spec = expr(node["spec"]) if node["spec"] is not None else ""
            conversion = node["conversion"]
            if conversion != "none":
                value = {"s": str, "r": repr, "a": ascii}[conversion](value)
            return format(value, spec)
        if kind == "format_string":
            return "".join(expr(part) for part in node["parts"])
        raise AssertionError(f"Unknown expression {kind}")

    def target(node, value):
        if node["kind"] == "name_target":
            env[node["name"]] = value
        else:
            values = list(value)
            if len(values) != len(node["items"]):
                raise ValueError("Unpack length mismatch")
            for item, v in zip(node["items"], values):
                target(item, v)

    def block(body):
        for node in body:
            kind = node["kind"]
            if kind == "assign":
                env[node["target"]] = expr(node["value"])
            elif kind == "unpack":
                target(node["target"], expr(node["value"]))
            elif kind == "set_item":
                value, index, item = expr(node["value"]), expr(node["index"]), expr(node["item"])
                value[index] = item
            elif kind == "if":
                block(node["then_body"] if expr(node["condition"]) else node["else_body"])
            elif kind == "for":
                for value in expr(node["iterable"]):
                    target(node["target"], value)
                    block(node["body"])
            elif kind == "infer":
                args = [expr(arg) for arg in node["args"]]
                kwargs = {key: expr(value) for key, value in node["kwargs"]}
                env[node["target"]] = getattr(anno, node["operation"])(*args, **kwargs)
            elif kind == "constrain":
                valid = expr(node["condition"])
                message = expr(node["message"]) if node["message"] is not None else None
                if not valid:
                    raise ValueError(message or "Constraint failed")
            elif kind == "return":
                raise _Returned(expr(node["value"]))
            else:
                raise AssertionError(f"Unknown statement {kind}")

    try:
        block(program["body"])
    except _Returned as result:
        return result.value
    raise AssertionError("Program ended without return")


def walk_json(value):
    """Traverse a decoded tree for structural acceptance checks."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_json(item)
