"""Compile a Python function object into Lev IR without executing its body.

The public entry point is compile_program(function). Source must be available to
inspect. Only reachable same-module helpers are compiled; all calls are inlined.
Captured JSON constants are snapshots, while preconditions remain client-local.
Annotations and decorators are never evaluated by the compiler.

This is a deliberately restricted Python language. It supports structured finite
loops, conditional control flow, local containers and the documented intrinsics.
It rejects dynamic calls, recursion, closures, comprehensions, exceptions,
async, loop exits and writes to inputs/captured values. Definite assignment and
container ownership checks are conservative, particularly across loops. Runtime
type errors on input-dependent expressions remain runtime errors.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field, replace
import inspect
import json
import math
import textwrap
from types import FunctionType
from typing import Callable

from lev import ir
from lev._client_common import JSONValue, get_program_config, precondition, system_prompt
from lev.lev import Example, Lev, constrain


class CompileError(ValueError):
    """A rejected program, with a source location and inline call chain."""

    def __init__(self, message: str, source: ir.Source | None = None):
        self.message = message
        self.source = source
        location = ""
        if source is not None:
            s = source.span
            location = f"{s.module}:{s.line}:{s.column} ({s.function}): "
            if source.calls:
                message += "\n  inlined from " + " -> ".join(
                    f"{s.module}:{s.line}:{s.column} ({s.function})" for s in source.calls
                )
        super().__init__(location + message)


@dataclass(frozen=True)
class _Info:
    origin: str = "unknown"
    container: str | None = None
    items: tuple[_Info, ...] | None = None
    element: _Info | None = None
    response: str | None = None


_UNKNOWN = _Info()
_SCALAR = _Info("scalar")
_STRING = _Info("scalar", "str")
_READONLY = _Info("readonly")
_CONTEXT = _Info("context")


def _merge(a: _Info, b: _Info) -> _Info:
    if a == b:
        return a
    origin = a.origin if a.origin == b.origin else "unknown"
    container = a.container if a.container == b.container else None
    items = None
    if a.items is not None and b.items is not None and len(a.items) == len(b.items):
        items = tuple(_merge(x, y) for x, y in zip(a.items, b.items))
    element = _merge(a.element, b.element) if a.element and b.element else None
    response = a.response if a.response == b.response else None
    return _Info(origin, container, items, element, response)


def _merge_env(a: dict[str, _Info], b: dict[str, _Info]) -> dict[str, _Info]:
    return {name: _merge(info, b[name]) for name, info in a.items() if name in b}


@dataclass(frozen=True)
class _Value:
    expr: ir.Expr | None  # context is a compile-time capability, never a value
    info: _Info = _UNKNOWN


@dataclass
class _Parsed:
    function: FunctionType
    node: ast.FunctionDef
    signature: inspect.Signature
    lines: list[str]
    start: int
    indentation: int


@dataclass
class _Frame:
    parsed: _Parsed
    names: dict[str, str]
    env: dict[str, _Info]
    calls: tuple[ir.SourceSpan, ...] = ()
    result: str | None = None
    done: str | None = None
    return_infos: list[_Info] = field(default_factory=list)
    loop_depth: int = 0

    def fork(self):
        return replace(self, env=dict(self.env))


_OPERATIONS = ("noul", "choice", "score", "extract", "generate", "generate_list")
_RESPONSES = {
    "noul": {"noul": _SCALAR},
    "choice": {"choice": _STRING, "confidence": _SCALAR, "probabilities": _READONLY},
    "score": {"score": _SCALAR, "confidence": _SCALAR, "probabilities": _READONLY, "legend": _READONLY},
    "extract": {"items": _Info("readonly", "list", element=_STRING)},
    "generate": {"text": _STRING},
    "generate_list": {"items": _Info("readonly", "list", element=_STRING)},
}
_COMPARISONS = {
    ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le",
    ast.Gt: "gt", ast.GtE: "ge", ast.In: "in", ast.NotIn: "not_in",
    ast.Is: "is", ast.IsNot: "is_not",
}
_INTRINSICS = ((builtins.len, "len"), (builtins.int, "int"),
               (builtins.range, "range"), (builtins.enumerate, "enumerate"),
               (json.dumps, "json.dumps"))


class _Compiler:
    def __init__(self, entry: FunctionType):
        self.entry = entry
        self.counter = 0
        self.parsed: dict[FunctionType, _Parsed] = {}
        self.active: list[FunctionType] = []

    def fresh(self, hint: str) -> str:
        self.counter += 1
        if self.counter > 100_000:
            raise CompileError("Expanded program exceeds the 100000-node/name compilation budget")
        return f"v{self.counter}_{hint}"

    def span(self, f: _Frame, node: ast.AST) -> ir.SourceSpan:
        p = f.parsed
        line = getattr(node, "lineno", p.node.lineno)
        end = getattr(node, "end_lineno", line) or line

        def column(lineno, offset):
            # Python AST columns are UTF-8 byte offsets, not character offsets.
            content = p.lines[lineno - 1] if lineno <= len(p.lines) else ""
            return len(content.encode("utf-8")[:offset].decode("utf-8")) + p.indentation + 1

        return ir.SourceSpan(
            p.function.__module__, p.function.__qualname__, p.start + line - 1,
            column(line, getattr(node, "col_offset", 0)), p.start + end - 1,
            column(end, getattr(node, "end_col_offset", 0) or 0),
        )

    def source(self, f, node):
        return ir.Source(self.span(f, node), f.calls)

    def fail(self, f, node, message):
        raise CompileError(message, self.source(f, node))

    def stmt(self, cls, f, node, *args, **kwargs):
        return cls(*args, **kwargs, id=self.fresh("stmt"), source=self.source(f, node))

    def parse(self, function, caller=None, call=None):
        def error(message):
            if caller is not None:
                self.fail(caller, call, message)
            raise CompileError(message)

        if not isinstance(function, FunctionType):
            error("Expected an ordinary Python function")
        if function.__closure__:
            error("Closures are not supported")
        if hasattr(function, "__wrapped__") or hasattr(function, "__signature__"):
            error("Wrapped functions and overridden signatures are not supported")
        if function in self.parsed:
            return self.parsed[function]
        try:
            lines, start = inspect.getsourcelines(function)
            raw = "".join(lines)
            source = textwrap.dedent(raw)
            tree = ast.parse(source)
            signature = inspect.signature(function, follow_wrapped=False, eval_str=False)
        except (OSError, IOError, TypeError, SyntaxError, ValueError) as exc:
            error(f"Function source is unavailable or invalid: {exc}")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            error("Only synchronous function definitions are supported")
        node = tree.body[0]
        if node.name != function.__code__.co_name:
            error("Source does not match the function's code name")
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()):
            error("Variadic function parameters are not supported")
        indentation = len(lines[0]) - len(lines[0].lstrip())
        parsed = _Parsed(function, node, signature, source.splitlines(), start, indentation)
        self.parsed[function] = parsed
        return parsed

    def frame(self, parsed, calls=(), helper=False):
        names = dict.fromkeys(parsed.signature.parameters)
        for statement in parsed.node.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names[node.id] = None
        return _Frame(
            parsed, {name: self.fresh(name) for name in names}, {}, calls,
            self.fresh("result") if helper else None,
            self.fresh("done") if helper else None,
        )

    def raw_name(self, f, node):
        if node.id in f.names:
            self.fail(f, node, "Dynamic calls and function-valued local variables are not supported")
        namespace = f.parsed.function.__globals__
        if node.id in namespace:
            return namespace[node.id]
        namespace = f.parsed.function.__builtins__
        if node.id in namespace:
            return namespace[node.id]
        self.fail(f, node, f"Unknown name: {node.id}")

    def decorators(self, f, entry=False):
        for decorator in f.parsed.node.decorator_list:
            if not entry:
                self.fail(f, decorator, "Decorated helpers are not supported")
            if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name):
                self.fail(f, decorator, "Only system_prompt and precondition decorators are supported")
            value = self.raw_name(f, decorator.func)
            if value is not system_prompt and value is not precondition:
                self.fail(f, decorator, "Only system_prompt and precondition decorators are supported")

    def capture(self, value, f, node, seen=None):
        seen = set() if seen is None else seen
        if type(value) is Example:
            return self.capture(value.to_dict(), f, node, seen)
        if value is None or type(value) in (str, bool, int, float):
            if type(value) is float and not math.isfinite(value):
                self.fail(f, node, "Captured numbers must be finite")
            return _Value(ir.Literal(value), _STRING if type(value) is str else _SCALAR)
        if type(value) not in (list, dict):
            self.fail(f, node, f"Cannot capture {type(value).__name__}; only JSON constants are supported")
        if id(value) in seen:
            self.fail(f, node, "Cyclic constants are not supported")
        seen.add(id(value))
        try:
            if type(value) is list:
                expr = ir.ListExpr(tuple(self.capture(item, f, node, seen).expr for item in value))
                return _Value(expr, _Info("readonly", "list", element=_READONLY))
            if any(type(key) is not str for key in value):
                self.fail(f, node, "Captured dictionaries require string keys")
            expr = ir.DictExpr(tuple(
                (ir.Literal(key), self.capture(item, f, node, seen).expr) for key, item in value.items()
            ))
            return _Value(expr, _Info("readonly", "dict", element=_STRING))
        finally:
            seen.remove(id(value))

    def data(self, value, f, node):
        if value.expr is None:
            self.fail(f, node, "The injected Lev context cannot be stored in data or returned")
        return value.expr

    def materialize(self, value, out, f, node, pure=False):
        if pure or value.expr is None:
            return value
        name = self.fresh("tmp")
        out.append(self.stmt(ir.Assign, f, node, name, value.expr))
        return _Value(ir.Var(name), value.info)

    def value(self, expr, info, out, f, node, pure):
        return self.materialize(_Value(expr, info), out, f, node, pure)

    def expr(self, node, out, f, pure=False):
        if isinstance(node, ast.Constant):
            return self.capture(node.value, f, node)
        if isinstance(node, ast.Name):
            if node.id in f.names:
                if node.id not in f.env:
                    self.fail(f, node, f"Local variable may be unbound: {node.id}")
                info = f.env[node.id]
                return _Value(None if info == _CONTEXT else ir.Var(f.names[node.id]), info)
            return self.materialize(self.capture(self.raw_name(f, node), f, node), out, f, node, pure)
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self.expr(item, out, f, pure) for item in node.elts]
            expressions = tuple(self.data(v, f, node) for v in values)
            if isinstance(node, ast.Tuple):
                expression = ir.TupleExpr(expressions)
                info = _Info("scalar", "tuple", tuple(v.info for v in values))
            else:
                expression = ir.ListExpr(expressions)
                info = _Info("local", "list")
            return self.value(expression, info, out, f, node, pure)
        if isinstance(node, ast.Dict):
            entries = []
            for key, item in zip(node.keys, node.values):
                if key is None:
                    self.fail(f, node, "Dictionary unpacking is not supported")
                k = self.data(self.expr(key, out, f, pure), f, key)
                v = self.data(self.expr(item, out, f, pure), f, item)
                entries.append((k, v))
            return self.value(ir.DictExpr(tuple(entries)), _Info("local", "dict"), out, f, node, pure)
        if isinstance(node, ast.Attribute):
            receiver = self.expr(node.value, out, f, pure)
            if receiver.info == _CONTEXT and node.attr == "state":
                return self.value(ir.State(), _READONLY, out, f, node, pure)
            response = receiver.info.response
            if response is None or node.attr not in _RESPONSES[response]:
                self.fail(f, node, "Only Lev.state and declared inference response fields can be read")
            return self.value(ir.Field(receiver.expr, node.attr), _RESPONSES[response][node.attr], out, f, node, pure)
        if isinstance(node, ast.Subscript):
            receiver = self.expr(node.value, out, f, pure)
            index = self.expr(node.slice, out, f, pure)
            info = _READONLY if receiver.info.origin == "readonly" else _UNKNOWN
            if receiver.info.container == "str":
                info = _STRING
            if receiver.info.items is not None and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
                try:
                    info = receiver.info.items[node.slice.value]
                except IndexError:
                    pass  # runtime IndexError, not a compiler failure
            return self.value(ir.GetItem(self.data(receiver, f, node), self.data(index, f, node)), info, out, f, node, pure)
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.Not, ast.UAdd, ast.USub):
            operand = self.data(self.expr(node.operand, out, f, pure), f, node)
            op = {ast.Not: "not", ast.UAdd: "positive", ast.USub: "negative"}[type(node.op)]
            return self.value(ir.Unary(op, operand), _SCALAR, out, f, node, pure)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a = self.expr(node.left, out, f, pure)
            b = self.expr(node.right, out, f, pure)
            info = _UNKNOWN
            if a.info.container == b.info.container == "list":
                info = _Info("local", "list")
            elif a.info.container == "str" or b.info.container == "str":
                info = _STRING
            elif a.info.items is not None and b.info.items is not None:
                info = _Info("scalar", "tuple", a.info.items + b.info.items)
            return self.value(ir.Binary("add", self.data(a, f, node), self.data(b, f, node)), info, out, f, node, pure)
        if isinstance(node, ast.BoolOp):
            return self.bool_expr(node, out, f, pure)
        if isinstance(node, ast.Compare):
            return self.compare(node, out, f, pure)
        if isinstance(node, ast.JoinedStr):
            parts = tuple(self.data(self.expr(v, out, f, pure), f, v) for v in node.values)
            return self.value(ir.FormatString(parts), _STRING, out, f, node, pure)
        if isinstance(node, ast.FormattedValue):
            value = self.data(self.expr(node.value, out, f, pure), f, node)
            spec = self.data(self.expr(node.format_spec, out, f, pure), f, node) if node.format_spec else None
            conversion = {-1: "none", 115: "s", 114: "r", 97: "a"}[node.conversion]
            return self.value(ir.FormatValue(value, conversion, spec), _STRING, out, f, node, pure)
        if isinstance(node, ast.Call):
            return self.call(node, out, f, pure)
        self.fail(f, node, f"Unsupported expression: {type(node).__name__}")

    def bool_expr(self, node, out, f, pure):
        op = "and" if isinstance(node.op, ast.And) else "or"
        if pure:
            values = [self.expr(v, out, f, True) for v in node.values]
            info = values[0].info
            for v in values[1:]:
                info = _merge(info, v.info)
            return _Value(ir.BoolOp(op, tuple(self.data(v, f, node) for v in values)), info)
        first = self.expr(node.values[0], out, f)
        result = self.fresh("bool")
        out.append(self.stmt(ir.Assign, f, node, result, self.data(first, f, node)))
        info = first.info
        for child in node.values[1:]:
            body = []
            value = self.expr(child, body, f)
            body.append(self.stmt(ir.Assign, f, child, result, self.data(value, f, child)))
            condition = ir.Var(result) if op == "and" else ir.Unary("not", ir.Var(result))
            out.append(self.stmt(ir.If, f, child, condition, tuple(body)))
            info = _merge(info, value.info)
        return _Value(ir.Var(result), info)

    def compare(self, node, out, f, pure):
        for op, left, right in zip(node.ops, [node.left, *node.comparators], node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and not any(
                isinstance(v, ast.Constant) and (v.value is None or type(v.value) is bool)
                for v in (left, right)
            ):
                self.fail(f, node, "Identity comparisons are supported only against None or booleans")
        ops = tuple(_COMPARISONS[type(op)] for op in node.ops)
        left = self.data(self.expr(node.left, out, f, pure), f, node)
        if pure:
            rights = tuple(self.data(self.expr(v, out, f, True), f, v) for v in node.comparators)
            return _Value(ir.Compare(left, ops, rights), _SCALAR)
        result = self.fresh("comparison")
        for index, (op, child) in enumerate(zip(ops, node.comparators)):
            body = out if index == 0 else []
            right = self.data(self.expr(child, body, f), f, child)
            body.append(self.stmt(ir.Assign, f, child, result, ir.Compare(left, (op,), (right,))))
            if index:
                out.append(self.stmt(ir.If, f, child, ir.Var(result), tuple(body)))
            left = right
        return _Value(ir.Var(result), _SCALAR)

    def arguments(self, node, out, f, pure):
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            self.fail(f, node, "Starred call arguments are not supported")
        args = [self.expr(a, out, f, pure) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg in kwargs:
                self.fail(f, node, f"Duplicate keyword: {kw.arg}")
            kwargs[kw.arg] = self.expr(kw.value, out, f, pure)
        return args, kwargs

    def bind(self, signature, args, kwargs, f, node):
        try:
            return signature.bind(*args, **kwargs)
        except TypeError as exc:
            self.fail(f, node, f"Invalid call arguments: {exc}")

    def call(self, node, out, f, pure):
        target = None
        receiver = None
        operation = None
        intrinsic = None
        if isinstance(node.func, ast.Name):
            target = self.raw_name(f, node.func)
        elif isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id not in f.names and self.raw_name(f, base) is json:
                if node.func.attr != "dumps":
                    self.fail(f, node, "Only json.dumps is supported")
                target = json.dumps
            else:
                receiver = self.expr(base, out, f, pure)
                if receiver.info == _CONTEXT and node.func.attr in _OPERATIONS:
                    operation = node.func.attr
                elif node.func.attr == "strip" and receiver.expr is not None:
                    intrinsic = "str.strip"
                else:
                    self.fail(f, node, "Unsupported method or dynamic call")
        else:
            self.fail(f, node, "Dynamic calls are not supported")
        if target is constrain:
            self.fail(f, node, "constrain must be a standalone statement")
        if pure and (operation is not None or isinstance(target, FunctionType) and target is not json.dumps):
            self.fail(f, node, "Constraint expressions must be pure; inference and helper calls are not allowed")
        args, kwargs = self.arguments(node, out, f, pure)
        if target is Example:
            bound = self.bind(inspect.signature(Example), args, kwargs, f, node)
            expression = ir.Intrinsic("example", tuple(
                self.data(bound.arguments[name], f, node) for name in ("state", "answer")
            ))
            return self.value(expression, _Info("readonly", "dict", element=_STRING),
                              out, f, node, pure)
        if operation:
            self.bind(inspect.signature(getattr(Lev, operation)), [None, *args], kwargs, f, node)
            expr_args = tuple(self.data(v, f, node) for v in args)
            expr_kwargs = tuple((k, self.data(v, f, node)) for k, v in kwargs.items())
            name = self.fresh(operation)
            out.append(self.stmt(ir.Infer, f, node, name, operation, expr_args, expr_kwargs))
            return _Value(ir.Var(name), _Info("readonly", response=operation))
        for function, name in _INTRINSICS:
            if target is function:
                intrinsic = name
                break
        if intrinsic:
            if receiver is not None:
                args = [receiver, *args]
            info = self.check_intrinsic(intrinsic, args, kwargs, f, node)
            expression = ir.Intrinsic(intrinsic, tuple(self.data(v, f, node) for v in args),
                                      tuple((k, self.data(v, f, node)) for k, v in kwargs.items()))
            return self.value(expression, info, out, f, node, pure)
        if isinstance(target, FunctionType) and target.__module__ == self.entry.__module__:
            return self.inline(target, args, kwargs, out, f, node)
        self.fail(f, node, "Only known intrinsics and same-module helper functions can be called")

    def check_intrinsic(self, name, args, kwargs, f, node):
        valid = True
        if name == "len":
            valid = len(args) == 1 and not kwargs
            info = _SCALAR
        elif name == "int":
            valid = len(args) <= 2 and set(kwargs) <= {"base"} and not (len(args) == 2 and kwargs)
            info = _SCALAR
        elif name == "range":
            valid = 1 <= len(args) <= 3 and not kwargs
            info = _Info("scalar", "range", element=_SCALAR)
        elif name == "enumerate":
            self.bind(inspect.signature(enumerate), args, kwargs, f, node)
            iterable = args[0] if args else kwargs["iterable"]
            element = _Info("scalar", "tuple", (_SCALAR, self.element(iterable.info)))
            info = _Info("scalar", "enumerate", element=element)
        elif name == "str.strip":
            valid = 1 <= len(args) <= 2 and not kwargs
            info = _STRING
        else:
            self.bind(inspect.signature(json.dumps), args, kwargs, f, node)
            valid = set(kwargs) <= {"obj", "skipkeys", "ensure_ascii", "check_circular", "allow_nan", "indent", "separators", "sort_keys"}
            info = _STRING
        if not valid:
            self.fail(f, node, f"Unsupported arguments for intrinsic {name}")
        return info

    def inline(self, function, args, kwargs, out, caller, call):
        if function in self.active:
            self.fail(caller, call, "Recursive helper calls are not supported")
        parsed = self.parse(function, caller, call)
        frame = self.frame(parsed, caller.calls + (self.span(caller, call),), helper=True)
        self.decorators(frame)
        bound = self.bind(parsed.signature, args, kwargs, caller, call)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            if not isinstance(value, _Value):
                value = self.capture(value, caller, call)
            frame.env[name] = value.info
            if value.expr is not None:
                out.append(self.stmt(ir.Assign, caller, call, frame.names[name], value.expr))
        out.append(self.stmt(ir.Assign, caller, call, frame.result, ir.Literal(None)))
        out.append(self.stmt(ir.Assign, caller, call, frame.done, ir.Literal(False)))
        self.active.append(function)
        try:
            body, falls, _ = self.block(parsed.node.body, frame)
        finally:
            self.active.pop()
        out.extend(body)
        infos = frame.return_infos + ([_SCALAR] if falls else [])
        info = infos[0] if infos else _SCALAR
        for other in infos[1:]:
            info = _merge(info, other)
        return _Value(ir.Var(frame.result), info)

    def element(self, info):
        if info.element is not None:
            return info.element
        if info.container == "str":
            return _STRING
        if info.container == "dict":
            return _SCALAR
        if info.items:
            element = info.items[0]
            for item in info.items[1:]:
                element = _merge(element, item)
            return element
        return _READONLY if info.origin == "readonly" else _UNKNOWN

    def target(self, node, info, f):
        if isinstance(node, ast.Name):
            # A join must never turn the Lev capability into ordinary data.
            if node.id in f.env and (f.env[node.id] == _CONTEXT) != (info == _CONTEXT):
                self.fail(f, node, "Cannot rebind a Lev context to data or vice versa")
            f.env[node.id] = info
            return ir.NameTarget(f.names[node.id])
        if isinstance(node, (ast.Tuple, ast.List)):
            if info.items is not None and len(info.items) == len(node.elts):
                infos = info.items
            else:
                infos = (self.element(info),) * len(node.elts)
            return ir.TupleTarget(tuple(self.target(n, i, f) for n, i in zip(node.elts, infos)))
        self.fail(f, node, "Unpacking and loop targets must contain only names")

    def assign(self, target, value, out, f):
        if isinstance(target, ast.Subscript):
            receiver = self.expr(target.value, out, f)
            index = self.expr(target.slice, out, f)
            if receiver.info.origin != "local" or receiver.info.container not in ("list", "dict"):
                self.fail(f, target, "Only provably local list/dict containers may be modified; inputs and captures are read-only")
            out.append(self.stmt(ir.SetItem, f, target, receiver.expr, self.data(index, f, target), self.data(value, f, target)))
            return
        compiled = self.target(target, value.info, f)
        if value.info == _CONTEXT:
            if not isinstance(compiled, ir.NameTarget):
                self.fail(f, target, "Cannot unpack a Lev context")
            return
        expression = self.data(value, f, target)
        if isinstance(compiled, ir.NameTarget):
            out.append(self.stmt(ir.Assign, f, target, compiled.name, expression))
        else:
            out.append(self.stmt(ir.Unpack, f, target, compiled, expression))

    def block(self, nodes, f):
        out = []
        returned = False
        for index, node in enumerate(nodes):
            falls, may_return = self.statement(node, out, f)
            returned |= may_return
            if not falls:
                if index + 1 < len(nodes):
                    self.fail(f, nodes[index + 1], "Unreachable statements after an unconditional return are not supported")
                return tuple(out), False, returned
            if may_return and f.done is not None and index + 1 < len(nodes):
                suffix, falls, suffix_returns = self.block(nodes[index + 1:], f)
                out.append(self.stmt(ir.If, f, node, ir.Unary("not", ir.Var(f.done)), suffix))
                return tuple(out), falls, returned or suffix_returns
        return tuple(out), True, returned

    def statement(self, node, out, f):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return True, False  # docstring
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id not in f.names:
                if self.raw_name(f, call.func) is constrain:
                    args, kwargs = self.arguments(call, out, f, True)
                    bound = self.bind(inspect.signature(constrain), args, kwargs, f, call)
                    condition = self.data(bound.arguments["condition"], f, call)
                    message = bound.arguments.get("message")
                    out.append(self.stmt(ir.Constrain, f, node, condition, self.data(message, f, call) if message else None))
                    return True, False
            value = self.expr(node.value, out, f)
            self.data(value, f, node)
            return True, False
        if isinstance(node, ast.Assign):
            value = self.expr(node.value, out, f)
            # Evaluate the RHS once even when a target rebinds a referenced name.
            value = self.materialize(value, out, f, node)
            for target in node.targets:
                self.assign(target, value, out, f)
            return True, False
        if isinstance(node, ast.If):
            condition = self.data(self.expr(node.test, out, f), f, node.test)
            a, b = f.fork(), f.fork()
            yes, a_falls, a_returns = self.block(node.body, a)
            no, b_falls, b_returns = self.block(node.orelse, b)
            if a_falls and b_falls:
                # Context/data mixtures cannot be represented as ordinary values.
                for name in a.env.keys() & b.env.keys():
                    if (a.env[name] == _CONTEXT) != (b.env[name] == _CONTEXT):
                        self.fail(f, node, f"Inconsistent Lev context binding across branches: {name}")
                f.env = _merge_env(a.env, b.env)
            elif a_falls:
                f.env = a.env
            elif b_falls:
                f.env = b.env
            out.append(self.stmt(ir.If, f, node, condition, yes, no))
            return a_falls or b_falls, a_returns or b_returns
        if isinstance(node, ast.For):
            if node.orelse:
                self.fail(f, node, "for/else is not supported")
            iterable = self.expr(node.iter, out, f)
            self.data(iterable, f, node)
            # Compute a conservative loop invariant before accepting writes. This
            # catches aliases that become read-only on a later iteration.
            initial = dict(f.env)
            invariant = initial
            while True:
                body_frame = f.fork()
                body_frame.env = dict(invariant)
                body_frame.loop_depth += 1
                target = self.target(node.target, self.element(iterable.info), body_frame)
                body, _, _ = self.block(node.body, body_frame)
                merged = _merge_env(initial, body_frame.env)
                merged = _merge_env(invariant, merged)
                if merged == invariant:
                    break
                invariant = merged
            f.env = invariant
            out.append(self.stmt(ir.For, f, node, target, iterable.expr, body))
            return True, False
        if isinstance(node, ast.Return):
            if f.loop_depth:
                self.fail(f, node, "Returns inside loops are not supported")
            value = self.expr(node.value, out, f) if node.value else _Value(ir.Literal(None), _SCALAR)
            expression = self.data(value, f, node)
            if f.result is None:
                out.append(self.stmt(ir.Return, f, node, expression))
            else:
                out.append(self.stmt(ir.Assign, f, node, f.result, expression))
                out.append(self.stmt(ir.Assign, f, node, f.done, ir.Literal(True)))
                f.return_infos.append(value.info)
            return False, True
        if isinstance(node, ast.Pass):
            return True, False
        self.fail(f, node, f"Unsupported statement: {type(node).__name__}")

    def compile(self):
        parsed = self.parse(self.entry)
        parameters = list(parsed.signature.parameters.values())
        if len(parameters) != 1 or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
        ) or parameters[0].default is not inspect.Parameter.empty:
            raise CompileError("Entry must accept exactly one positional Lev parameter without a default")
        frame = self.frame(parsed)
        frame.env[parameters[0].name] = _CONTEXT
        self.decorators(frame, entry=True)
        self.active.append(self.entry)
        body, falls, _ = self.block(parsed.node.body, frame)
        if falls:
            body += (self.stmt(ir.Return, frame, parsed.node, ir.Literal(None)),)
        config = get_program_config(self.entry)
        program = ir.Program(
            self.entry.__qualname__, ir.ProgramConfig(**config), body, self.span(frame, parsed.node)
        )
        program.to_dict()  # ensure the emitted tree is serializable
        return program


def compile_program(program: Callable[[Lev], JSONValue], *, optimize: bool = True) -> ir.Program:
    """Compile one function and snapshot its reachable constants/configuration.

    Importing the function is the caller's responsibility. Compilation performs
    no input validation, model calls, user function execution or remote work.
    Unsupported programs raise CompileError instead of falling back to Python.
    Set optimize=False to inspect the original lowering before optimization.
    """
    result = _Compiler(program).compile()
    if optimize:
        from lev.passes import optimize_program
        result = optimize_program(result)
    return result
