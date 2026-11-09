"""Version 1 of AnnoScript's structured, fully inlined intermediate language.

Blocks execute in order. Expressions are pure and evaluate left to right;
BoolOp and Compare short-circuit. Assignments bind values (including container
references), not copies. For evaluates its finite iterable once. Infer invokes
exactly one operation on the program's injected AnnoScript and binds its response.
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
json.dumps follow their Python counterparts on supported data values. example
snapshots state and answer into a plain JSON object, without constructing Python
application objects. strip
requires a string receiver. These names never resolve to user-provided code.

This module defines data and serialization, not an executor or solver. All node
collections are tuples. The wire representation uses explicit snake-case tags,
preserves dictionary entry order, and contains no Python objects or source code.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from functools import lru_cache
import json
import math
from types import UnionType
from typing import ClassVar, Literal as TypingLiteral, Union, get_args, get_origin, get_type_hints


class IRDecodeError(ValueError):
    """Malformed or unsupported serialized IR, with a path into the document."""

    def __init__(self, message: str, path: str = "$"):
        self.message = message
        self.path = path
        super().__init__(f"{path}: {message}")


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

    @classmethod
    def from_dict(cls, data: dict) -> Program:
        """Restore version-1 IR from JSON data, without executing any code.

        Checks node categories, field types, operators, source ranges and unique
        statement IDs. This is structural validation, not type/ownership analysis
        or proof that the program's constraints can be satisfied. Unknown fields
        and versions are rejected. Collections are detached immutable tuples.
        """
        if type(data) is not dict:
            raise IRDecodeError("expected a program object")
        if data.get("format") != "lev-ir":
            raise IRDecodeError("expected format 'lev-ir'", "$.format")
        if type(data.get("version")) is not int or data["version"] != 1:
            raise IRDecodeError("unsupported IR version; expected 1", "$.version")
        try:
            return _Decoder().decode(data, Program, "$")
        except RecursionError as exc:
            raise IRDecodeError("IR nesting is too deep") from exc

    @classmethod
    def from_json(cls, text: str) -> Program:
        """Read JSON text; reject duplicate object keys and non-finite numbers."""
        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise IRDecodeError(f"duplicate JSON object key {key!r}")
                result[key] = value
            return result

        def constant(value):
            raise IRDecodeError(f"non-finite JSON number {value!r}")

        try:
            data = json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant)
        except json.JSONDecodeError as exc:
            raise IRDecodeError(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        except (RecursionError, UnicodeError) as exc:
            raise IRDecodeError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(data)


# A closed registry: a JSON tag can never import a class or resolve Python code.
_NODE_TYPES = {
    node.kind: node for node in (
        Literal, Var, State, GetItem, Field, ListExpr, TupleExpr, DictExpr,
        Unary, Binary, Compare, BoolOp, Intrinsic, FormatValue, FormatString,
        NameTarget, TupleTarget, Assign, Unpack, SetItem, If, For, Infer,
        Constrain, Return,
    )
}
_COMPARISON_OPS = frozenset(("eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is", "is_not"))
_INTRINSIC_NAMES = frozenset(("len", "int", "range", "enumerate", "str.strip", "json.dumps", "example"))
_RESPONSE_FIELDS = frozenset(("noul", "choice", "confidence", "probabilities", "score", "legend", "items", "text"))


@lru_cache(maxsize=None)
def _field_types(cls):
    # Only our trusted dataclasses reach this function, never document contents.
    return get_type_hints(cls)


class _Decoder:
    def __init__(self):
        self.statement_ids = set()

    def decode(self, value, expected, path):
        origin = get_origin(expected)
        args = get_args(expected)
        if origin in (Union, UnionType):
            if value is None and type(None) in args:
                return None
            options = tuple(t for t in args if t is not type(None))
            if len(options) == 1:
                return self.decode(value, options[0], path)
            # The only non-optional union in the schema is Literal.value.
            if type(value) in options:
                return self.decode(value, type(value), path)
            raise IRDecodeError("expected a JSON scalar", path)
        if origin is TypingLiteral:
            if not any(type(value) is type(option) and value == option for option in args):
                raise IRDecodeError(f"expected one of {args!r}", path)
            return value
        if origin is tuple:
            if type(value) is not list:
                raise IRDecodeError("expected a JSON array", path)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(self.decode(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
            if len(value) != len(args):
                raise IRDecodeError(f"expected an array of length {len(args)}", path)
            return tuple(self.decode(v, t, f"{path}[{i}]") for i, (v, t) in enumerate(zip(value, args)))
        if expected in (str, bool, int, float, type(None)):
            if type(value) is not expected:
                raise IRDecodeError(f"expected {expected.__name__}", path)
            if expected is float and not math.isfinite(value):
                raise IRDecodeError("numbers must be finite", path)
            return value
        if not isinstance(expected, type) or not is_dataclass(expected):
            raise TypeError(f"Unsupported internal IR schema type: {expected!r}")
        if type(value) is not dict:
            raise IRDecodeError("expected an object", path)

        cls = expected
        tagged = issubclass(expected, Node)
        if tagged:
            kind = value.get("kind")
            if type(kind) is not str or kind not in _NODE_TYPES:
                raise IRDecodeError(f"unknown node kind {kind!r}", f"{path}.kind")
            cls = _NODE_TYPES[kind]
            if not issubclass(cls, expected):
                raise IRDecodeError(f"{kind!r} is not a {expected.__name__} node", path)

        definitions = fields(cls)
        allowed = {f.name for f in definitions} | ({"kind"} if tagged else set())
        unknown = value.keys() - allowed
        if unknown:
            raise IRDecodeError(f"unknown fields: {', '.join(sorted(map(repr, unknown)))}", path)
        values = {}
        types = _field_types(cls)
        for f in definitions:
            if f.name in value:
                values[f.name] = self.decode(value[f.name], types[f.name], f"{path}.{f.name}")
            elif f.default is MISSING and f.default_factory is MISSING:
                raise IRDecodeError("missing required field", f"{path}.{f.name}")
        result = cls(**values)
        self.validate(result, path)
        return result

    def validate(self, node, path):
        if isinstance(node, Stmt):
            if not node.id or node.id in self.statement_ids:
                raise IRDecodeError("statement IDs must be nonempty and unique", f"{path}.id")
            self.statement_ids.add(node.id)
        if isinstance(node, SourceSpan):
            if min(node.line, node.column, node.end_line, node.end_column) < 1:
                raise IRDecodeError("source coordinates must be positive", path)
            if (node.end_line, node.end_column) < (node.line, node.column):
                raise IRDecodeError("source end precedes its start", path)
        if isinstance(node, (Var, NameTarget)) and not node.name:
            raise IRDecodeError("variable name must not be empty", f"{path}.name")
        if isinstance(node, (Assign, Infer)) and not node.target:
            raise IRDecodeError("target must not be empty", f"{path}.target")
        if isinstance(node, Compare):
            if not node.ops or len(node.ops) != len(node.comparators):
                raise IRDecodeError("comparison requires matching nonempty operators and operands", path)
            if any(op not in _COMPARISON_OPS for op in node.ops):
                raise IRDecodeError("unknown comparison operator", f"{path}.ops")
        if isinstance(node, BoolOp) and len(node.values) < 2:
            raise IRDecodeError("boolean operation requires at least two operands", f"{path}.values")
        if isinstance(node, Intrinsic) and node.name not in _INTRINSIC_NAMES:
            raise IRDecodeError(f"unknown intrinsic {node.name!r}", f"{path}.name")
        if isinstance(node, Field) and node.name not in _RESPONSE_FIELDS:
            raise IRDecodeError(f"unknown response field {node.name!r}", f"{path}.name")
        if isinstance(node, (Infer, Intrinsic)):
            names = [name for name, _ in node.kwargs]
            if any(not name for name in names) or len(names) != len(set(names)):
                raise IRDecodeError("keyword argument names must be nonempty and unique", f"{path}.kwargs")
