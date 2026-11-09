"""Bounded folding of a closed set of scalar operations, never Python source."""

from dataclasses import replace
import math
import operator

from annoscript import ir
from ._analysis import map_expressions, rewrite_expr


_COMPARE = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
    "gt": operator.gt, "ge": operator.ge, "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b, "is": operator.is_, "is_not": operator.is_not,
}
_MAX_STRING = 65536
_MAX_BITS = 4096


def _bounded(value):
    if type(value) is str:
        return len(value) <= _MAX_STRING
    if type(value) is int:
        return value.bit_length() <= _MAX_BITS
    if type(value) is float:
        return math.isfinite(value)
    return value is None or type(value) is bool


def _fold(expr):
    if isinstance(expr, ir.BoolOp):
        values = list(expr.values)
        while len(values) > 1 and isinstance(values[0], ir.Literal):
            first = values.pop(0)
            if (expr.op == "and" and not first.value) or (expr.op == "or" and first.value):
                return first
        return values[0] if len(values) == 1 else replace(expr, values=tuple(values))

    def literal(node):
        if not isinstance(node, ir.Literal) or not _bounded(node.value):
            raise _NotConstant
        return node.value

    try:
        if isinstance(expr, ir.Unary):
            value = {"not": operator.not_, "positive": operator.pos, "negative": operator.neg}[expr.op](literal(expr.value))
        elif isinstance(expr, ir.Binary):
            a, b = literal(expr.left), literal(expr.right)
            if isinstance(a, str) and isinstance(b, str) and len(a) + len(b) > _MAX_STRING:
                return expr
            value = operator.add(a, b)
        elif isinstance(expr, ir.Compare):
            left = literal(expr.left)
            value = True
            for op, item in zip(expr.ops, expr.comparators):
                right = literal(item)
                # Identity is meaningful only for the IR's singleton values.
                if op in ("is", "is_not") and not any(v is None or type(v) is bool for v in (left, right)):
                    return expr
                if not _COMPARE[op](left, right):
                    value = False
                    break
                left = right
        elif isinstance(expr, ir.FormatValue):
            if expr.spec is not None and literal(expr.spec) != "":
                return expr
            value = literal(expr.value)
            if expr.conversion != "none":
                value = {"s": str, "r": repr, "a": ascii}[expr.conversion](value)
            value = format(value, "")
        elif isinstance(expr, ir.FormatString):
            parts = [literal(part) for part in expr.parts]
            if not all(type(part) is str for part in parts) or sum(map(len, parts)) > _MAX_STRING:
                return expr
            value = "".join(parts)
        elif isinstance(expr, ir.Intrinsic) and expr.name in ("len", "int", "str.strip"):
            args = [literal(arg) for arg in expr.args]
            kwargs = {name: literal(arg) for name, arg in expr.kwargs}
            value = {"len": len, "int": int, "str.strip": str.strip}[expr.name](*args, **kwargs)
        else:
            return expr
        return ir.Literal(value) if _bounded(value) else expr
    except (_NotConstant, TypeError, ValueError, OverflowError):
        # A failing expression must still fail at its original execution point.
        return expr


class _NotConstant(Exception):
    pass


def fold_constants(program: ir.Program) -> ir.Program:
    def block(body):
        output = []
        for statement in body:
            statement = map_expressions(statement, lambda e: rewrite_expr(e, _fold))
            if isinstance(statement, ir.If):
                statement = replace(statement, then_body=block(statement.then_body), else_body=block(statement.else_body))
            elif isinstance(statement, ir.For):
                statement = replace(statement, body=block(statement.body))
            output.append(statement)
        return tuple(output)

    return replace(program, body=block(program.body))
