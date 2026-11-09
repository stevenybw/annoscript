"""Serializable dependency templates for a future dynamic AnnoScript scheduler.

Tasks are instantiated per request and loop coordinate. An edge's common loop
coordinates match through its carried loop (or the entire common prefix when
there is no carried loop), with the specified positive distance subtracted.
Deeper source coordinates are gathered and deeper destination coordinates are
broadcast, including nested loops whose trip counts vary between outer rounds.
Inactive control paths satisfy completion edges, but never produce values.

Infer sites demand a shared request, *after* their own dependencies are ready.
Sharing is local to a request, reuse scope, and dependency/repair generation.
It is not eager hoisting, global memoization, retrying, or a batch assignment.
Response values are read-only. Identity-observing programs disable sharing.

Eager execution omits failure_order and explicitly strict_only ordering edges,
allowing independent work to pass pending checks. Strict execution includes them,
preserving the sequential failure prefix (with equivalent requests shared).
Both report the first observed error, stop dispatch on failure, best-effort cancel
in-flight work, and await all active work, including checks, before success.
Resource exhaustion is outside the static expression safety model.

Speculative execution uses the same formal dependencies and serial loop rules
as eager. A separate bounded evaluator first speculates Decision requests across
unknown control gates, then batches generation requests on confirmed paths.
It never commits local effects or checks. Results and request-local errors are
adopted only on demand with matching arguments.

BoundExpr captures value bindings, not copies of their containers. State and
inference responses are read-only; Store addresses local storage. Branch.done
means its predicate is known; Join.done means its selected region has completed,
and Join.continue is false if that path returned (including nested returns).
Loop.enter activates iteration regions; Loop.done gathers all active iterations
and publishes exports. serial_modes additionally orders iterator advancement
after the preceding region finishes, preserving iterator faults in strict mode
and necessary mutable-iterator ordering in both modes.
Carry uses its initial value in iteration zero and update from the prior round.
Merge reads only the selected arm. Return supplies a candidate result: success
also requires completion of all activated required work in the root region.

Checks validate captured expressions on their original paths. Data, control and
memory dependencies apply in both modes; source-order check barriers apply only
in strict mode. A check alone does not serialize eager loop iterations. Checks
do not introduce solving scopes or authorize changing branch choices.
Any future repair must invalidate downstream values and shared request caches
for the affected dependency generation; this module implements no repair.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
import json
from types import UnionType
from typing import ClassVar, Literal, Union, get_args, get_origin, get_type_hints

from annoscript import ir

FailureMode = Literal["eager", "strict", "speculative"]


class PlanDecodeError(ValueError):
    def __init__(self, message, path="$"):
        self.path = path
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class Binding:
    name: str
    value: str


@dataclass(frozen=True)
class BoundExpr:
    expression: ir.Expr
    bindings: tuple[Binding, ...] = ()


@dataclass(frozen=True)
class Gate:
    task: str
    port: Literal["true", "false", "continue", "enter"]


@dataclass(frozen=True)
class Region:
    id: str
    parent: str | None
    kind: Literal["root", "then", "else", "loop"]
    owner: str | None
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class Value:
    id: str
    producer: str
    name: str | None = None


@dataclass(frozen=True)
class InstanceMap:
    source_loops: tuple[str, ...] = ()
    target_loops: tuple[str, ...] = ()
    carried_loop: str | None = None
    distance: int = 0


@dataclass(frozen=True)
class Dependency:
    source: str
    target: str
    kind: Literal["data", "control", "memory", "constraint", "failure_order"]
    mapping: InstanceMap
    port: Literal["done", "enter", "true", "false", "continue"] = "done"
    value: str | None = None
    hazard: Literal["RAW", "WAR", "WAW"] | None = None
    certainty: Literal["proven", "conservative"] = "proven"
    condition: Gate | None = None
    seed: str | None = None
    # Constraint ordering can be relaxed without dropping check input/memory
    # dependencies. Absent flags retain the stronger semantics of older plans.
    # failure_order edges are always strict-only, including legacy plans.
    strict_only: bool = False


@dataclass(frozen=True)
class InferRequest:
    id: str
    operation: Literal["noul", "choice", "score", "extract", "generate", "generate_list"]
    scope: tuple[str, ...]
    sites: tuple[str, ...]
    reusable: bool


@dataclass(frozen=True, kw_only=True)
class Task(ir.Node):
    id: str
    region: str
    origin: str
    source: ir.Source
    outputs: tuple[str, ...] = ()
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True)
class Eval(Task):
    kind: ClassVar[str] = "eval"
    expression: BoundExpr


@dataclass(frozen=True)
class Unpack(Task):
    kind: ClassVar[str] = "unpack"
    expression: BoundExpr
    target: ir.TupleTarget


@dataclass(frozen=True)
class Store(Task):
    kind: ClassVar[str] = "store"
    container: BoundExpr
    index: BoundExpr
    item: BoundExpr


@dataclass(frozen=True)
class Infer(Task):
    kind: ClassVar[str] = "infer_site"
    request: str
    arguments: str  # Eval output: (positional tuple, keyword-value tuple)
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Branch(Task):
    kind: ClassVar[str] = "branch"
    condition: BoundExpr
    then_region: str
    else_region: str


@dataclass(frozen=True)
class Join(Task):
    kind: ClassVar[str] = "join"
    branch: str


@dataclass(frozen=True)
class Merge(Task):
    kind: ClassVar[str] = "merge"
    branch: str
    then_value: str | None
    else_value: str | None


@dataclass(frozen=True)
class Carry(Task):
    kind: ClassVar[str] = "carry"
    loop: str
    initial: str
    update: str


@dataclass(frozen=True)
class Iteration(Task):
    kind: ClassVar[str] = "iteration"
    loop: str
    target: ir.Target


@dataclass(frozen=True)
class LoopExport:
    output: str
    initial: str
    update: str


@dataclass(frozen=True)
class Loop(Task):
    kind: ClassVar[str] = "loop"
    iterable: BoundExpr
    body: str
    exports: tuple[LoopExport, ...]
    # Iterator advancement waits for the preceding body in these modes.
    # This is independent of the data-carry edges between individual tasks.
    serial_modes: tuple[FailureMode, ...] = ("strict",)


@dataclass(frozen=True)
class Check(Task):
    kind: ClassVar[str] = "check"
    condition: BoundExpr
    message: BoundExpr | None


@dataclass(frozen=True)
class Return(Task):
    kind: ClassVar[str] = "return"
    expression: BoundExpr


_TASKS = {t.kind: t for t in (Eval, Unpack, Store, Infer, Branch, Join, Merge,
                             Carry, Iteration, Loop, Check, Return)}


@dataclass(frozen=True)
class ExecutionPlan:
    entry: str
    config: ir.ProgramConfig
    source: ir.SourceSpan
    regions: tuple[Region, ...]
    tasks: tuple[Task, ...]
    values: tuple[Value, ...]
    dependencies: tuple[Dependency, ...]
    requests: tuple[InferRequest, ...]
    format: str = "lev-execution-plan"
    version: int = 1

    def dependencies_for(self, mode: FailureMode = "eager") -> tuple[Dependency, ...]:
        """Select edges; region, loop, demand and root-completion rules still apply."""
        if mode not in ("eager", "strict", "speculative"):
            raise ValueError(f"unknown failure mode {mode!r}")
        return tuple(d for d in self.dependencies
                     if mode == "strict" or (d.kind != "failure_order" and not d.strict_only))

    def to_dict(self) -> dict:
        return ir._encode(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionPlan:
        try:
            plan = _decode(data, cls, "$")
            plan.validate()
            return plan
        except (ir.IRDecodeError, RecursionError) as exc:
            raise PlanDecodeError(str(exc)) from exc

    @classmethod
    def from_json(cls, text: str) -> ExecutionPlan:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise PlanDecodeError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        def constant(value):
            raise PlanDecodeError(f"non-finite number {value}")

        try:
            return cls.from_dict(json.loads(text, object_pairs_hook=pairs, parse_constant=constant))
        except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
            raise PlanDecodeError(f"invalid JSON: {exc}") from exc

    def validate(self) -> None:
        """Check references, ownership, availability, seeded carries and DAGs."""
        _validate(self)


def _decode(value, expected, path):
    origin, args = get_origin(expected), get_args(expected)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        return _decode(value, next(t for t in args if t is not type(None)), path)
    if origin is Literal:
        if not any(type(value) is type(a) and value == a for a in args):
            raise PlanDecodeError(f"expected one of {args}", path)
        return value
    if origin is tuple:
        if type(value) is not list:
            raise PlanDecodeError("expected array", path)
        return tuple(_decode(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
    if expected in (str, int, bool, type(None)):
        if type(value) is not expected:
            raise PlanDecodeError(f"expected {expected.__name__}", path)
        return value
    if isinstance(expected, type) and expected.__module__ == ir.__name__:
        return ir._Decoder().decode(value, expected, path)
    if type(value) is not dict:
        raise PlanDecodeError("expected object", path)
    cls = expected
    tagged = isinstance(expected, type) and issubclass(expected, Task)
    if tagged:
        kind = value.get("kind")
        cls = _TASKS.get(kind) if type(kind) is str else None
        if cls is None or not issubclass(cls, expected):
            raise PlanDecodeError("unknown or incorrect task kind", path)
    if not is_dataclass(cls):
        raise TypeError(f"unknown plan schema type {cls}")
    definitions = fields(cls)
    unknown = value.keys() - ({f.name for f in definitions} | ({"kind"} if tagged else set()))
    if unknown:
        raise PlanDecodeError(f"unknown fields {sorted(unknown)}", path)
    types, kwargs = get_type_hints(cls), {}
    for f in definitions:
        if f.name in value:
            kwargs[f.name] = _decode(value[f.name], types[f.name], f"{path}.{f.name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            raise PlanDecodeError("missing required field", f"{path}.{f.name}")
    return cls(**kwargs)


def expressions(task):
    return tuple(getattr(task, f.name) for f in fields(task)
                 if isinstance(getattr(task, f.name), BoundExpr))


def loop_path(region, regions):
    result = []
    while region is not None:
        r = regions[region]
        if r.kind == "loop":
            result.append(r.owner)
        region = r.parent
    return tuple(reversed(result))


def _validate(plan):
    from annoscript.passes._analysis import expression_reads

    def require(condition, message, path="$"):
        if not condition:
            raise PlanDecodeError(message, path)

    def index(items, name):
        result = {x.id: x for x in items}
        require(len(result) == len(items) and all(result), "IDs must be unique and nonempty", name)
        return result

    require(plan.format == "lev-execution-plan" and type(plan.version) is int and plan.version == 1,
            "unsupported execution plan format/version")
    ts, rs, vs, qs = (index(getattr(plan, name), f"$.{name}")
                      for name in ("tasks", "regions", "values", "requests"))
    require(len(set(ts) | set(rs) | set(vs) | set(qs)) == len(ts) + len(rs) + len(vs) + len(qs),
            "identifiers must be unique across the plan")
    require(sum(r.kind == "root" for r in rs.values()) == 1, "expected one root region")
    for r in rs.values():
        require((r.parent is None) == (r.kind == "root"), "incorrect region parent", r.id)
        require(r.kind != "root" or r.owner is None, "root region cannot have an owner", r.id)
        require(r.parent is None or r.parent in rs, "unknown parent region", r.id)
        seen, cur = set(), r
        while cur.parent is not None:
            require(cur.id not in seen, "cyclic region hierarchy", r.id)
            seen.add(cur.id)
            require(cur.parent in rs, "unknown ancestor region", r.id)
            cur = rs[cur.parent]
        require(len(set(r.tasks)) == len(r.tasks), "duplicate region task", r.id)
        require(all(t in ts and ts[t].region == r.id for t in r.tasks), "incorrect task ownership", r.id)
        if r.parent is not None:
            require(r.owner in ts and ts[r.owner].region == r.parent, "incorrect region owner", r.id)
            owner = ts[r.owner]
            require((isinstance(owner, Loop) and r.kind == "loop" and owner.body == r.id) or
                    (isinstance(owner, Branch) and ((r.kind == "then" and owner.then_region == r.id) or
                                                   (r.kind == "else" and owner.else_region == r.id))),
                    "region is not selected by its owner", r.id)
    for v in vs.values():
        require(v.producer in ts and v.id in ts[v.producer].outputs, "invalid value producer", v.id)

    def ref(v, where):
        require(v in vs, f"unknown value {v!r}", where)

    def ancestors(region):
        result = set()
        while region is not None:
            result.add(region)
            region = rs[region].parent
        return result

    def available(v, region, where):
        ref(v, where)
        require(ts[vs[v].producer].region in ancestors(region), "value is unavailable in this region", where)

    def data_edge(v, target, condition=None):
        require(any(d.target == target and d.value == v and d.kind == "data" and d.condition == condition
                    for d in plan.dependencies), "missing value dependency", target)

    for t in ts.values():
        require(t.region in rs and t.id in rs[t.region].tasks, "unlisted task", t.id)
        require(bool(t.origin), "missing source statement identity", t.id)
        require(len(set(t.outputs)) == len(t.outputs), "duplicate output", t.id)
        if isinstance(t, (Eval, Infer, Merge, Carry)):
            require(len(t.outputs) == 1, "task requires one output", t.id)
        elif isinstance(t, (Store, Branch, Join, Check, Return)):
            require(not t.outputs, "task cannot define outputs", t.id)
        elif isinstance(t, (Unpack, Iteration)):
            def count(target):
                return 1 if isinstance(target, ir.NameTarget) else sum(count(x) for x in target.items)
            require(count(t.target) == len(t.outputs), "unpack target/output mismatch", t.id)
        for v in t.outputs:
            ref(v, t.id)
            require(vs[v].producer == t.id, "output owned by another task", t.id)
        for b in expressions(t):
            names = [v.name for v in b.bindings]
            require(len(names) == len(set(names)) and set(names) == set(expression_reads(b.expression)),
                    "expression bindings do not match its variables", t.id)
            for binding in b.bindings:
                available(binding.value, t.region, t.id)
                data_edge(binding.value, t.id)
        for g in t.gates:
            require(g.task in ts, "unknown gate task", t.id)
            owner = ts[g.task]
            require((g.port in ("true", "false") and isinstance(owner, Branch)) or
                    (g.port == "continue" and isinstance(owner, Join)) or
                    (g.port == "enter" and isinstance(owner, Loop)), "incorrect gate port", t.id)
            require(any(d.source == g.task and d.target == t.id and d.port == g.port and d.kind == "control"
                        for d in plan.dependencies), "missing control dependency", t.id)
        if isinstance(t, Branch):
            for region, kind in ((t.then_region, "then"), (t.else_region, "else")):
                require(region in rs and rs[region].kind == kind and rs[region].owner == t.id,
                        "invalid branch region", t.id)
        if isinstance(t, (Join, Merge)):
            require(isinstance(ts.get(t.branch), Branch), "unknown branch", t.id)
            require(ts[t.branch].region == t.region, "branch and merge/join must share a region", t.id)
            require(any(d.source == t.branch and d.target == t.id and d.kind == "control" and d.port == "done"
                        for d in plan.dependencies), "missing branch dependency", t.id)
        if isinstance(t, Merge):
            branch = ts[t.branch]
            for v, arm, port in ((t.then_value, branch.then_region, "true"), (t.else_value, branch.else_region, "false")):
                if v is not None:
                    ref(v, t.id)
                    producer_region = ts[vs[v].producer].region
                    eligible, r = {arm}, rs[t.region]
                    while True:
                        eligible.add(r.id)
                        if r.parent is None:
                            break
                        r = rs[r.parent]
                    require(producer_region in eligible, "merge value belongs to the wrong arm", t.id)
                    data_edge(v, t.id, Gate(t.branch, port))
            require(t.then_value is not None or t.else_value is not None, "merge has no incoming values", t.id)
        if isinstance(t, Loop):
            require(t.body in rs and rs[t.body].owner == t.id and rs[t.body].kind == "loop", "invalid loop body", t.id)
            require(set(t.serial_modes) <= {"strict", "eager"} and "strict" in t.serial_modes,
                    "strict iteration sequencing is required", t.id)
            require(tuple(e.output for e in t.exports) == t.outputs, "loop exports do not match outputs", t.id)
            for e in t.exports:
                available(e.initial, t.region, t.id)
                data_edge(e.initial, t.id)
                ref(e.update, t.id)
                require(ts[vs[e.update].producer].region == t.body, "loop update must be in its body", t.id)
        if isinstance(t, (Carry, Iteration)):
            require(isinstance(ts.get(t.loop), Loop) and ts[t.loop].body == t.region, "incorrect iteration owner", t.id)
        if isinstance(t, Carry):
            available(t.initial, rs[t.region].parent, t.id)
            data_edge(t.initial, t.loop)
            ref(t.update, t.id)
            require(ts[vs[t.update].producer].region == t.region, "invalid carry update", t.id)
            require(t.loop not in loop_path(ts[vs[t.initial].producer].region, rs), "carry seed must precede loop", t.id)
        if isinstance(t, Infer):
            available(t.arguments, t.region, t.id)
            data_edge(t.arguments, t.id)
            require(len(t.keywords) == len(set(t.keywords)) and all(t.keywords), "invalid keyword names", t.id)
            require(t.request in qs and t.id in qs[t.request].sites, "invalid shared request", t.id)
    for q in qs.values():
        require(bool(q.sites) and len(q.sites) == len(set(q.sites)), "invalid request sites", q.id)
        for site in q.sites:
            require(isinstance(ts.get(site), Infer) and ts[site].request == q.id, "request/site mismatch", q.id)
            path = loop_path(ts[site].region, rs)
            require(path[:len(q.scope)] == q.scope, "request scope is not an enclosing scope", q.id)

    graph = {t: set() for t in ts}
    for d in plan.dependencies:
        require(d.source in ts and d.target in ts, "unknown dependency endpoint")
        require(type(d.strict_only) is bool, "strict_only must be a boolean", d.target)
        require(not d.strict_only or d.kind in ("constraint", "failure_order"),
                "data, control and memory dependencies cannot be strict-only", d.target)
        m = d.mapping
        require(m.source_loops == loop_path(ts[d.source].region, rs)
                and m.target_loops == loop_path(ts[d.target].region, rs), "incorrect instance mapping", d.target)
        require((m.carried_loop is None and m.distance == 0) or
                (m.carried_loop in m.source_loops and m.carried_loop in m.target_loops and m.distance > 0),
                "invalid carried distance", d.target)
        if d.value is not None:
            ref(d.value, d.target)
            require(vs[d.value].producer == d.source, "dependency has wrong value producer", d.target)
        if d.seed is not None:
            ref(d.seed, d.target)
        if d.condition:
            require(isinstance(ts.get(d.condition.task), Branch), "invalid dependency condition", d.target)
            require(d.condition.port in ("true", "false"), "invalid branch condition port", d.target)
        if d.kind == "memory":
            require(d.hazard is not None, "memory dependency requires a hazard", d.target)
        else:
            require(d.hazard is None, "non-memory dependency cannot carry a memory hazard", d.target)
        if not m.distance and d.port == "done":
            graph[d.target].add(d.source)
    # Structural loop/join completion is not a normal same-instance graph edge.
    remaining = {k: set(v) for k, v in graph.items()}
    while remaining:
        ready = {k for k, v in remaining.items() if not v}
        require(bool(ready), "unseeded zero-distance dependency cycle")
        remaining = {k: v - ready for k, v in remaining.items() if k not in ready}
    for t in ts.values():
        if isinstance(t, Carry):
            require(any(d.target == t.id and d.value == t.update and d.seed == t.initial
                        and d.mapping.carried_loop == t.loop and d.mapping.distance == 1
                        for d in plan.dependencies), "carry requires a seeded previous-iteration edge", t.id)
