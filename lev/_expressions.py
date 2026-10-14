"""Closed expression interpreter for the runtime's value bindings."""

import json
import operator

from lev import ir
from lev.backends.base import RESPONSE_TYPES


_COMPARE = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
    "gt": operator.gt, "ge": operator.ge, "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b, "is": operator.is_, "is_not": operator.is_not,
}


def expression(node, env, state, allocations=None):
    ev = lambda e: expression(e, env, state, allocations)
    def allocated(value):
        if allocations is not None and type(value) in (list, dict):
            allocations[id(value)] = value
        return value
    if isinstance(node, ir.Literal):
        return node.value
    if isinstance(node, ir.Var):
        return env[node.name]
    if isinstance(node, ir.State):
        return state
    if isinstance(node, ir.GetItem):
        return ev(node.value)[ev(node.index)]
    if isinstance(node, ir.Field):
        value = ev(node.value)
        if type(value) not in RESPONSE_TYPES.values() or node.name not in value.__dataclass_fields__:
            raise ValueError("field access requires a response field")
        return getattr(value, node.name)
    if isinstance(node, ir.ListExpr):
        return allocated([ev(x) for x in node.items])
    if isinstance(node, ir.TupleExpr):
        return tuple(ev(x) for x in node.items)
    if isinstance(node, ir.DictExpr):
        return allocated({ev(k): ev(v) for k, v in node.entries})
    if isinstance(node, ir.Unary):
        return {"not": operator.not_, "negative": operator.neg, "positive": operator.pos}[node.op](ev(node.value))
    if isinstance(node, ir.Binary):
        return allocated(ev(node.left) + ev(node.right))
    if isinstance(node, ir.Compare):
        left = ev(node.left)
        for op, other in zip(node.ops, node.comparators):
            right = ev(other)
            if not _COMPARE[op](left, right):
                return False
            left = right
        return True
    if isinstance(node, ir.BoolOp):
        value = ev(node.values[0])
        for other in node.values[1:]:
            if (node.op == "and" and not value) or (node.op == "or" and value):
                break
            value = ev(other)
        return value
    if isinstance(node, ir.Intrinsic):
        fn = {"len": len, "int": int, "range": range, "enumerate": enumerate,
              "str.strip": str.strip, "json.dumps": json.dumps}[node.name]
        return fn(*[ev(x) for x in node.args], **{k: ev(v) for k, v in node.kwargs})
    if isinstance(node, ir.FormatValue):
        value = ev(node.value)
        spec = ev(node.spec) if node.spec is not None else ""
        if node.conversion != "none":
            value = {"s": str, "r": repr, "a": ascii}[node.conversion](value)
        return format(value, spec)
    if isinstance(node, ir.FormatString):
        return "".join(ev(x) for x in node.parts)
    raise ValueError("unsupported expression")


def unpack(target, value):
    if isinstance(target, ir.NameTarget):
        return [value]
    # Consume at most one extra item, just like exact Python unpacking.
    it = iter(value)
    values = []
    for _ in target.items:
        try:
            values.append(next(it))
        except StopIteration:
            raise ValueError("not enough values to unpack") from None
    try:
        next(it)
    except StopIteration:
        return [v for t, item in zip(target.items, values) for v in unpack(t, item)]
    raise ValueError("too many values to unpack")
