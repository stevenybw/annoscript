"""Version 1 of Lev's structured, fully inlined intermediate language.

Blocks execute in order. Expressions are pure and evaluate left to right;
BoolOp and Compare short-circuit. Assignments bind values (including container
references), not copies. For evaluates its finite iterable once. Infer invokes
exactly one operation on the program's injected Lev and binds its response.
Field reads address response fields, never arbitrary Python attributes.

Constrain retains an expression over the bindings at its execution point. A
runtime must establish it before proceeding, or fail; validation alone is not
a repair solver. Branch decisions already consumed by control flow stay fixed.
Inlining introduces no solving scope. Retrying an inference requires invalidating
its dependent computations. Return delivers one JSON value after active
constraints have been satisfied. Tuples are permitted as intermediate values.

Statement IDs identify static sites; a runtime must additionally distinguish loop
iterations and repeated evaluations. Source metadata is diagnostic only.
Intrinsic names have fixed semantics: len, int, range, enumerate, str.strip and
json.dumps follow their Python counterparts on supported data values. strip
requires a string receiver. These names never resolve to user-provided code.

This module defines data and serialization, not an executor or solver. All node
collections are tuples. The wire representation uses explicit snake-case tags,
preserves dictionary entry order, and contains no Python objects or source code.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import math
from typing import ClassVar, Literal as TypingLiteral


def _encode(value):
    if isinstance(value, Node):
        return {"kind": value.kind, **{f.name: _encode(getattr(value, f.name)) for f in fields(value)}}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"IR requires immutable JSON-compatible data, got {type(value).__name__}")


@dataclass(frozen=True)
class Node:
    kind: ClassVar[str]


@dataclass(frozen=True)
class Expr(Node):
    pass


@dataclass(frozen=True)
class Literal(Expr):
    kind: ClassVar[str] = "literal"
    value: str | int | float | bool | None

    def __post_init__(self):
        if self.value is not None and type(self.value) not in (str, int, float, bool):
            raise ValueError("Literal values must be JSON scalars")
        _encode(self.value)


@dataclass(frozen=True)
class Var(Expr):
    kind: ClassVar[str] = "var"
    name: str


@dataclass(frozen=True)
class State(Expr):
    kind: ClassVar[str] = "state"


@dataclass(frozen=True)
class GetItem(Expr):
    kind: ClassVar[str] = "get_item"
    value: Expr
    index: Expr


@dataclass(frozen=True)
class Field(Expr):
    kind: ClassVar[str] = "field"
    value: Expr
    name: str


@dataclass(frozen=True)
class ListExpr(Expr):
    kind: ClassVar[str] = "list"
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class TupleExpr(Expr):
    kind: ClassVar[str] = "tuple"
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class DictExpr(Expr):
    kind: ClassVar[str] = "dict"
    entries: tuple[tuple[Expr, Expr], ...]


@dataclass(frozen=True)
class Unary(Expr):
    kind: ClassVar[str] = "unary"
    op: TypingLiteral["not", "negative", "positive"]
    value: Expr


@dataclass(frozen=True)
class Binary(Expr):
    kind: ClassVar[str] = "binary"
    op: TypingLiteral["add"]
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Compare(Expr):
    kind: ClassVar[str] = "compare"
    left: Expr
    ops: tuple[str, ...]
    comparators: tuple[Expr, ...]


@dataclass(frozen=True)
class BoolOp(Expr):
    kind: ClassVar[str] = "bool_op"
    op: TypingLiteral["and", "or"]
    values: tuple[Expr, ...]


@dataclass(frozen=True)
class Intrinsic(Expr):
    kind: ClassVar[str] = "intrinsic"
    name: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...] = ()


@dataclass(frozen=True)
class FormatValue(Expr):
    kind: ClassVar[str] = "format_value"
    value: Expr
    conversion: TypingLiteral["none", "s", "r", "a"] = "none"
    spec: Expr | None = None


@dataclass(frozen=True)
class FormatString(Expr):
    kind: ClassVar[str] = "format_string"
    parts: tuple[Expr, ...]


@dataclass(frozen=True)
class Target(Node):
    pass


@dataclass(frozen=True)
class NameTarget(Target):
    kind: ClassVar[str] = "name_target"
    name: str


@dataclass(frozen=True)
class TupleTarget(Target):
    kind: ClassVar[str] = "tuple_target"
    items: tuple[Target, ...]


@dataclass(frozen=True)
class SourceSpan:
    module: str
    function: str
    line: int
    column: int  # one-based Unicode character column
    end_line: int
    end_column: int  # exclusive, one-based


@dataclass(frozen=True)
class Source:
    span: SourceSpan
    calls: tuple[SourceSpan, ...] = ()  # outermost to innermost call site


@dataclass(frozen=True, kw_only=True)
class Stmt(Node):
    id: str
    source: Source


@dataclass(frozen=True)
class Assign(Stmt):
    kind: ClassVar[str] = "assign"
    target: str
    value: Expr


@dataclass(frozen=True)
class Unpack(Stmt):
    kind: ClassVar[str] = "unpack"
    target: TupleTarget
    value: Expr


@dataclass(frozen=True)
class SetItem(Stmt):
    kind: ClassVar[str] = "set_item"
    value: Expr
    index: Expr
    item: Expr


@dataclass(frozen=True)
class If(Stmt):
    kind: ClassVar[str] = "if"
    condition: Expr
    then_body: tuple[Stmt, ...]
    else_body: tuple[Stmt, ...] = ()


@dataclass(frozen=True)
class For(Stmt):
    kind: ClassVar[str] = "for"
    target: Target
    iterable: Expr
    body: tuple[Stmt, ...]


@dataclass(frozen=True)
class Infer(Stmt):
    kind: ClassVar[str] = "infer"
    target: str
    operation: TypingLiteral["noul", "choice", "score", "extract", "generate", "generate_list"]
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...]


@dataclass(frozen=True)
class Constrain(Stmt):
    kind: ClassVar[str] = "constrain"
    condition: Expr
    message: Expr | None = None


@dataclass(frozen=True)
class Return(Stmt):
    kind: ClassVar[str] = "return"
    value: Expr


@dataclass(frozen=True)
class ProgramConfig:
    system_prompt: str | None = None


@dataclass(frozen=True)
class Program:
    entry: str
    config: ProgramConfig
    body: tuple[Stmt, ...]
    source: SourceSpan
    format: str = "lev-ir"
    version: int = 1

    def to_dict(self) -> dict:
        """Return detached JSON data; absence of a prompt remains distinguishable."""
        data = _encode(self)
        if self.config.system_prompt is None:
            data["config"] = {}
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)
