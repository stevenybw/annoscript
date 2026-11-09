"""Readable, non-executable notation for any version-1 AnnoScript IR program.

format_ir(program, show_source=False) preserves every statement, temporary name,
literal and control-flow block. It neither optimizes nor runs the program and
does not reconstruct the original helper functions. format_value/format_string
in the output denote the corresponding IR expressions, not Python functions.
"""

from __future__ import annotations

import keyword

from annoscript import ir


_COMPARE = {
    "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    "in": "in", "not_in": "not in", "is": "is", "is_not": "is not",
}


def _identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _name(name: str) -> str:
    return name if _identifier(name) else f"var[{name!r}]"


def _arguments(args, kwargs) -> str:
    parts = [_expression(arg) for arg in args]
    for name, value in kwargs:
        if _identifier(name):
            parts.append(f"{name}={_expression(value)}")
        else:
            parts.append(f"**{{{name!r}: {_expression(value)}}}")
    return ", ".join(parts)


def _base(node: ir.Expr) -> str:
    text = _expression(node)
    if isinstance(node, (ir.Var, ir.State, ir.GetItem, ir.Field, ir.Intrinsic)):
        return text
    return f"({text})"


def _expression(node: ir.Expr) -> str:
    match node:
        case ir.Literal(value):
            return repr(value)
        case ir.Var(name):
            return _name(name)
        case ir.State():
            return "anno.state"
        case ir.GetItem(value, index):
            return f"{_base(value)}[{_expression(index)}]"
        case ir.Field(value, name):
            return f"{_base(value)}.{name}"
        case ir.ListExpr(items):
            return "[" + ", ".join(map(_expression, items)) + "]"
        case ir.TupleExpr(items):
            return "(" + ", ".join(map(_expression, items)) + ("," if len(items) == 1 else "") + ")"
        case ir.DictExpr(entries):
            return "{" + ", ".join(f"{_expression(k)}: {_expression(v)}" for k, v in entries) + "}"
        case ir.Unary(op, value):
            prefix = {"not": "not ", "negative": "-", "positive": "+"}[op]
            return f"({prefix}{_expression(value)})"
        case ir.Binary("add", left, right):
            return f"({_expression(left)} + {_expression(right)})"
        case ir.Compare(left, ops, comparators):
            parts = [_expression(left)]
            for op, right in zip(ops, comparators):
                parts.extend((_COMPARE[op], _expression(right)))
            return "(" + " ".join(parts) + ")"
        case ir.BoolOp(op, values):
            return "(" + f" {op} ".join(map(_expression, values)) + ")"
        case ir.Intrinsic(name, args, kwargs):
            return f"{name}({_arguments(args, kwargs)})"
        case ir.FormatValue(value, conversion, spec):
            parts = [_expression(value)]
            if conversion != "none":
                parts.append(f"conversion={conversion!r}")
            if spec is not None:
                parts.append(f"spec={_expression(spec)}")
            return "format_value(" + ", ".join(parts) + ")"
        case ir.FormatString(parts):
            return "format_string(" + ", ".join(map(_expression, parts)) + ")"
    raise ValueError(f"Unsupported IR expression: {type(node).__name__}")


def _target(node: ir.Target) -> str:
    match node:
        case ir.NameTarget(name):
            return _name(name)
        case ir.TupleTarget(items):
            return "(" + ", ".join(map(_target, items)) + ("," if len(items) == 1 else "") + ")"
    raise ValueError(f"Unsupported IR target: {type(node).__name__}")


def _location(span: ir.SourceSpan) -> str:
    return f"{span.module}:{span.line}:{span.column} ({span.function})"


def _comment(text: str) -> str:
    # Keep embedded newlines/control characters in metadata on one comment line.
    return "# " + repr(text)[1:-1]


class _Formatter:
    def __init__(self, show_source):
        self.show_source = show_source
        self.lines = []

    def line(self, depth, text):
        self.lines.append("    " * depth + text)

    def block(self, body, depth):
        if not body:
            self.line(depth, "pass")
        for node in body:
            if self.show_source:
                text = f"[{node.id}] {_location(node.source.span)}"
                if node.source.calls:
                    text += "; inlined from " + " -> ".join(map(_location, node.source.calls))
                self.line(depth, _comment(text))
            match node:
                case ir.Assign(target, value):
                    self.line(depth, f"{_name(target)} = {_expression(value)}")
                case ir.Unpack(target, value):
                    self.line(depth, f"{_target(target)} = {_expression(value)}")
                case ir.SetItem(value, index, item):
                    self.line(depth, f"{_base(value)}[{_expression(index)}] = {_expression(item)}")
                case ir.If(condition, then_body, else_body):
                    self.line(depth, f"if {_expression(condition)}:")
                    self.block(then_body, depth + 1)
                    if else_body:
                        self.line(depth, "else:")
                        self.block(else_body, depth + 1)
                case ir.For(target, iterable, body):
                    self.line(depth, f"for {_target(target)} in {_expression(iterable)}:")
                    self.block(body, depth + 1)
                case ir.Infer(target, operation, args, kwargs):
                    self.line(depth, f"{_name(target)} = anno.{operation}({_arguments(args, kwargs)})")
                case ir.Constrain(condition, message):
                    text = f"constrain({_expression(condition)}"
                    if message is not None:
                        text += f", message={_expression(message)}"
                    self.line(depth, text + ")")
                case ir.Return(value):
                    self.line(depth, f"return {_expression(value)}")
                case _:
                    raise ValueError(f"Unsupported IR statement: {type(node).__name__}")


def format_ir(program: ir.Program, *, show_source: bool = False) -> str:
    """Render a Program as deterministic display text, ending with a newline.

    Load external data through Program.from_json/from_dict first. The output is
    an inspection format, not executable Python or a serialization format.
    """
    if not isinstance(program, ir.Program):
        raise TypeError("format_ir expects an ir.Program")
    formatter = _Formatter(show_source)
    formatter.line(0, f"# AnnoScript IR v{program.version} (display notation)")
    if show_source:
        formatter.line(0, _comment(_location(program.source)))
    if program.config.system_prompt is not None:
        formatter.line(0, f"system_prompt {program.config.system_prompt!r}")
    formatter.line(0, f"program {program.entry!r}:")
    formatter.block(program.body, 1)
    return "\n".join(formatter.lines) + "\n"
