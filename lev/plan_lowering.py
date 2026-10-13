"""Lower structured Lev IR into demand-driven dependency templates.

No source inspection, model execution, or optimization of the input Program.
Alias analysis is deliberately allocation-based, not index-disjointness based:
unknown accesses retain may-dependences, including dictionary insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import inspect

from lev import ir
from lev import plan as p
from lev.lev import Lev
from lev.passes._analysis import (block_writes, definitions, direct_reads,
                                 expression_reads, literal_key, walk_statements)


class PlanLoweringError(ValueError):
    def __init__(self, message, statement=None):
        self.source = statement.source if statement else None
        suffix = ""
        if statement:
            s = statement.source.span
            suffix = f" at {s.module}:{s.line}:{s.column} ({s.function})"
            for call in statement.source.calls:
                suffix += f" <- {call.function}:{call.line}:{call.column}"
        super().__init__(message + suffix)


def _names(target):
    if isinstance(target, ir.NameTarget):
        return [target.name]
    return [n for item in target.items for n in _names(item)]


def _entry_reads(body, bound=frozenset()):
    """Names read before a definite local definition, and definite definitions."""
    bound, reads = set(bound), set()
    for s in body:
        reads |= set(direct_reads(s)) - bound
        if isinstance(s, ir.If):
            a, ad = _entry_reads(s.then_body, bound)
            b, bd = _entry_reads(s.else_body, bound)
            reads |= a | b
            bound = ad & bd
        elif isinstance(s, ir.For):
            a, _ = _entry_reads(s.body, bound | definitions(s))
            reads |= a
        else:
            bound |= definitions(s)
    return reads, bound


def _expr_children(expr):
    for f in fields(expr):
        def visit(v):
            if isinstance(v, ir.Expr):
                yield v
            elif isinstance(v, tuple):
                for x in v:
                    yield from visit(x)
        yield from visit(getattr(expr, f.name))


def _has_identity(program):
    def visit(e):
        if isinstance(e, ir.Compare) and any(op in ("is", "is_not") for op in e.ops):
            # Singleton identity is a value test, not allocation observation.
            operands = (e.left,) + e.comparators
            if any(op in ("is", "is_not") and not any(isinstance(x, ir.Literal) and (x.value is None or type(x.value) is bool)
                                                      for x in operands[i:i+2])
                   for i, op in enumerate(e.ops)):
                return True
        return any(visit(c) for c in _expr_children(e))
    return any(visit(v) for s in walk_statements(program.body) for f in fields(s)
               for v in (getattr(s, f.name),) if isinstance(v, ir.Expr))


@dataclass
class _Flow:
    env: dict
    tail: str | None = None
    barrier: str | None = None
    gates: tuple = ()
    falls: bool = True
    returns: bool = False

    def copy(self):
        return replace(self, env=dict(self.env))


class _Lowerer:
    def __init__(self, program):
        self.program = program
        self.tasks = {}
        self.regions = {}
        self.values = {}
        self.edges = []
        self.identity = _has_identity(program)
        self.serial = 0
        self.requests = []

    def fresh(self, prefix):
        self.serial += 1
        return f"{prefix}.{self.serial}"

    def region(self, parent, kind, owner):
        rid = self.fresh("region")
        self.regions[rid] = p.Region(rid, parent, kind, owner, ())
        return rid

    def path(self, task):
        return p.loop_path(self.tasks[task].region, self.regions)

    def edge(self, source, target, kind, *, carried=None, distance=0, **kwargs):
        d = p.Dependency(source, target, kind,
                         p.InstanceMap(self.path(source), self.path(target), carried, distance), **kwargs)
        if d not in self.edges:
            self.edges.append(d)

    def value_edge(self, value, target, **kwargs):
        self.edge(self.values[value].producer, target, "data", value=value, **kwargs)

    def bound(self, expr, flow, s):
        names = sorted(expression_reads(expr))
        for name in names:
            if name not in flow.env:
                raise PlanLoweringError(f"variable {name!r} is not definitely assigned", s)
        return p.BoundExpr(expr, tuple(p.Binding(n, flow.env[n]) for n in names))

    def add(self, cls, s, region, flow, *, names=(), tid=None, sequence=True, **kwargs):
        tid = tid or self.fresh(s.id)
        outputs = tuple(self.fresh("value") for _ in names)
        task = cls(id=tid, region=region, origin=s.id, source=s.source,
                   outputs=outputs, gates=flow.gates, **kwargs)
        self.tasks[tid] = task
        r = self.regions[region]
        self.regions[region] = replace(r, tasks=r.tasks + (tid,))
        for name, value in zip(names, outputs):
            self.values[value] = p.Value(value, tid, name)
        for expr in p.expressions(task):
            for binding in expr.bindings:
                self.value_edge(binding.value, tid)
        for gate in flow.gates:
            self.edge(gate.task, tid, "control", port=gate.port)
        if flow.barrier and flow.barrier != tid:
            self.edge(flow.barrier, tid, "constraint", strict_only=True)
        if sequence and flow.tail:
            self.edge(flow.tail, tid, "failure_order")
        if sequence:
            flow.tail = tid
        return task

    def block(self, body, region, incoming):
        flow = incoming.copy()
        for s in body:
            if not flow.falls:
                break
            if isinstance(s, ir.Assign):
                t = self.add(p.Eval, s, region, flow, names=(s.target,), expression=self.bound(s.value, flow, s))
                flow.env[s.target] = t.outputs[0]
            elif isinstance(s, ir.Unpack):
                names = _names(s.target)
                t = self.add(p.Unpack, s, region, flow, names=names,
                             expression=self.bound(s.value, flow, s), target=s.target)
                flow.env.update(zip(names, t.outputs))
            elif isinstance(s, ir.SetItem):
                self.add(p.Store, s, region, flow, container=self.bound(s.value, flow, s),
                         index=self.bound(s.index, flow, s), item=self.bound(s.item, flow, s))
            elif isinstance(s, ir.Infer):
                args = ir.TupleExpr((ir.TupleExpr(s.args), ir.TupleExpr(tuple(v for _, v in s.kwargs))))
                a = self.add(p.Eval, s, region, flow, names=(None,), expression=self.bound(args, flow, s))
                t = self.add(p.Infer, s, region, flow, names=(s.target,), request="pending",
                             arguments=a.outputs[0], keywords=tuple(k for k, _ in s.kwargs))
                self.value_edge(a.outputs[0], t.id)
                flow.env[s.target] = t.outputs[0]
            elif isinstance(s, ir.Constrain):
                t = self.add(p.Check, s, region, flow, condition=self.bound(s.condition, flow, s),
                             message=self.bound(s.message, flow, s) if s.message is not None else None)
                # Strict checks wait for prior active work. Eager checks need
                # only their actual data, control and memory prerequisites.
                for previous in self.regions[region].tasks[:-1]:
                    self.edge(previous, t.id, "constraint", strict_only=True)
                flow.barrier = t.id
            elif isinstance(s, ir.Return):
                if p.loop_path(region, self.regions):
                    raise PlanLoweringError("return inside a loop is unsupported", s)
                self.add(p.Return, s, region, flow, expression=self.bound(s.value, flow, s))
                flow.falls = False
                flow.returns = True
            elif isinstance(s, ir.If):
                self.branch(s, region, flow)
            elif isinstance(s, ir.For):
                self.loop(s, region, flow)
            else:
                raise PlanLoweringError(f"unsupported statement {type(s).__name__}", s)
        return flow

    def branch(self, s, region, flow):
        tid = self.fresh(s.id)
        yes = self.region(region, "then", tid)
        no = self.region(region, "else", tid)
        t = self.add(p.Branch, s, region, flow, tid=tid, condition=self.bound(s.condition, flow, s),
                     then_region=yes, else_region=no)
        children = []
        for body, rid, port in ((s.then_body, yes, "true"), (s.else_body, no, "false")):
            child = flow.copy()
            child.returns = False
            child.gates += (p.Gate(t.id, port),)
            children.append(self.block(body, rid, child))
        a, b = children
        join = self.add(p.Join, s, region, flow, branch=t.id)
        self.edge(t.id, join.id, "control")
        # Join itself is a completion token: the chosen child must finish.
        if a.barrier != flow.barrier or b.barrier != flow.barrier:
            flow.barrier = join.id
        if a.returns or b.returns:
            flow.gates += (p.Gate(join.id, "continue"),)
        flow.returns |= a.returns or b.returns
        flow.falls = a.falls or b.falls
        if not flow.falls:
            return
        available = (a.env.keys() & b.env.keys()) if a.falls and b.falls else (a.env.keys() if a.falls else b.env.keys())
        merged = {}
        for name in sorted(available):
            av, bv = (a.env.get(name) if a.falls else None), (b.env.get(name) if b.falls else None)
            if av == bv:
                merged[name] = av
                continue
            m = self.add(p.Merge, s, region, flow, sequence=False, names=(name,), branch=t.id,
                         then_value=av, else_value=bv)
            self.edge(t.id, m.id, "control")
            for value, port in ((av, "true"), (bv, "false")):
                if value is not None:
                    self.value_edge(value, m.id, condition=p.Gate(t.id, port))
            merged[name] = m.outputs[0]
        flow.env = merged

    def loop(self, s, region, flow):
        tid = self.fresh(s.id)
        rid = self.region(region, "loop", tid)
        t = self.add(p.Loop, s, region, flow, tid=tid, iterable=self.bound(s.iterable, flow, s),
                     body=rid, exports=())
        old = dict(flow.env)
        writes = block_writes(s.body) | definitions(s)
        reads, definite = _entry_reads(s.body, definitions(s))
        carry_names = sorted((writes & old.keys()) & (reads | (writes - definite)))
        child = _Flow(dict(old), gates=flow.gates + (p.Gate(tid, "enter"),), barrier=flow.barrier)
        carries = []
        for name in carry_names:
            c = self.add(p.Carry, s, rid, child, names=(name,), sequence=False,
                         loop=tid, initial=old[name], update="pending")
            child.env[name] = c.outputs[0]
            carries.append((name, c))
        names = _names(s.target)
        item = self.add(p.Iteration, s, rid, child, names=names, loop=tid, target=s.target)
        child.env.update(zip(names, item.outputs))
        child = self.block(s.body, rid, child)
        exports = []
        for name in sorted(writes & old.keys()):
            value = self.fresh("value")
            self.values[value] = p.Value(value, tid, name)
            exports.append(p.LoopExport(value, old[name], child.env[name]))
            flow.env[name] = value
            self.value_edge(old[name], tid)
        self.tasks[tid] = replace(t, outputs=tuple(e.output for e in exports), exports=tuple(exports))
        for name, c in carries:
            update = child.env[name]
            self.tasks[c.id] = replace(c, update=update)
            self.value_edge(update, c.id, carried=tid, distance=1, seed=c.initial)
        if child.tail:
            self.edge(child.tail, item.id, "failure_order", carried=tid, distance=1)
        # Checks preserve source order in strict mode, but do not alone force
        # eager iteration sequencing. Memory analysis may still require it.
        has_check = any(isinstance(x, ir.Constrain) for x in walk_statements(s.body))
        if has_check:
            if child.tail:
                self.edge(child.tail, item.id, "constraint", carried=tid, distance=1, strict_only=True)
            flow.barrier = tid

    def finish(self):
        root = self.region(None, "root", None)
        flow = self.block(self.program.body, root, _Flow({}))
        if flow.falls:
            raise PlanLoweringError("program has a reachable path without Return")
        self.analyze_memory()
        self.share_requests()
        result = p.ExecutionPlan(self.program.entry, self.program.config, self.program.source,
                                 tuple(self.regions.values()), tuple(self.tasks.values()),
                                 tuple(self.values.values()), tuple(self.edges), tuple(self.requests))
        result.validate()
        return result

    def analyze_memory(self):
        """Monotone reachability sets; allocation IDs retain iteration scopes."""
        self.aliases = {v: set() for v in self.values}
        self.points = {v: set() for v in self.values}
        self.allocations = {}
        self.mutable = set()

        def reach(e, bindings, allocation):
            if isinstance(e, ir.Var):
                return set(self.aliases[bindings[e.name]])
            if isinstance(e, ir.State):
                return {"state"}
            if isinstance(e, (ir.ListExpr, ir.DictExpr, ir.TupleExpr)):
                result = {allocation}
                for c in _expr_children(e):
                    result |= reach(c, bindings, allocation)
                return result
            if isinstance(e, (ir.GetItem, ir.Field)):
                return reach(e.value, bindings, allocation)
            if isinstance(e, ir.Binary):
                return {allocation} | set().union(*(reach(c, bindings, allocation) for c in _expr_children(e)))
            if isinstance(e, ir.Intrinsic) and e.name == "enumerate":
                return {allocation} | set().union(*(reach(c, bindings, allocation) for c in e.args))
            return set()

        self.reach = reach

        def points(e, bindings, allocation):
            if isinstance(e, ir.Var):
                return set(self.points[bindings[e.name]])
            if isinstance(e, (ir.ListExpr, ir.DictExpr, ir.TupleExpr, ir.Binary)):
                return {allocation}
            return reach(e, bindings, allocation)

        for t in self.tasks.values():
            for v in t.outputs:
                self.allocations[v] = self.path(t.id)
        changed = True
        while changed:
            changed = False
            for t in self.tasks.values():
                results = []
                direct = []
                if isinstance(t, (p.Eval, p.Unpack)):
                    b = t.expression
                    results = [reach(b.expression, {x.name: x.value for x in b.bindings}, t.outputs[0])]
                    direct = [points(b.expression, {x.name: x.value for x in b.bindings}, t.outputs[0])]
                elif isinstance(t, p.Infer):
                    results = [{f"response:{t.id}"}]
                elif isinstance(t, p.Merge):
                    results = [set().union(*(self.aliases[v] for v in (t.then_value, t.else_value) if v))]
                    direct = [set().union(*(self.points[v] for v in (t.then_value, t.else_value) if v))]
                elif isinstance(t, p.Carry):
                    results = [self.aliases[t.initial] | self.aliases[t.update]]
                    direct = [self.points[t.initial] | self.points[t.update]]
                elif isinstance(t, p.Loop):
                    results = [self.aliases[e.initial] | self.aliases[e.update] for e in t.exports]
                    direct = [self.points[e.initial] | self.points[e.update] for e in t.exports]
                elif isinstance(t, p.Iteration):
                    loop = self.tasks[t.loop]
                    results = [reach(loop.iterable.expression, {x.name: x.value for x in loop.iterable.bindings}, t.outputs[0])]
                for i, v in enumerate(t.outputs):
                    new = results[min(i, len(results)-1)] if results else set()
                    if not new <= self.aliases[v]:
                        self.aliases[v] |= new
                        changed = True
                    new_direct = direct[min(i, len(direct)-1)] if direct else new
                    if not new_direct <= self.points[v]:
                        self.points[v] |= new_direct
                        changed = True
            # A store changes containment, not the identity of its target.
            # Repeat value analysis so subsequent subscript-derived aliases
            # also see objects inserted after the original allocation.
            for t in self.tasks.values():
                if not isinstance(t, p.Store):
                    continue
                b = t.container
                targets = points(b.expression, {x.name: x.value for x in b.bindings}, t.id)
                b = t.item
                inserted = reach(b.expression, {x.name: x.value for x in b.bindings}, t.id)
                for aliases in self.aliases.values():
                    if aliases & targets and not inserted <= aliases:
                        aliases |= inserted
                        changed = True
        self.effects = {}
        for t in self.tasks.values():
            reads, writes = set(), set()
            for b in p.expressions(t):
                bindings = {x.name: x.value for x in b.bindings}

                def accesses(e):
                    result = set()
                    if isinstance(e, (ir.GetItem, ir.Field, ir.Compare, ir.BoolOp, ir.Unary, ir.Binary, ir.Intrinsic, ir.FormatValue)):
                        for c in _expr_children(e):
                            result |= reach(c, bindings, t.outputs[0] if t.outputs else t.id)
                    for c in _expr_children(e):
                        result |= accesses(c)
                    return result
                reads |= accesses(b.expression)
                # Check truth-tests the result even when its expression is a
                # bare container Var. Protect that read (and its message) from
                # later writes now that eager checks are not global barriers.
                if isinstance(t, (p.Return, p.Loop, p.Unpack, p.Check)):
                    reads |= reach(b.expression, bindings, t.id)
            if isinstance(t, p.Infer):
                reads |= self.aliases[t.arguments]
            if isinstance(t, p.Store):
                b = t.container
                writes = points(b.expression, {x.name: x.value for x in b.bindings}, t.id)
                if not writes or "state" in writes or any(a.startswith("response:") for a in writes):
                    raise PlanLoweringError("write target is not a provably local container", t)
                self.mutable |= writes
            # enumerate and iteration consume an object, even without SetItem.
            if isinstance(t, (p.Loop, p.Unpack)):
                writes |= {a for a in reads if self.is_iterator(a)}
            self.effects[t.id] = (reads, writes)
        self.mutable |= set().union(*(writes for _, writes in self.effects.values()))
        self.carried_allocations = {}
        for t in self.tasks.values():
            if isinstance(t, p.Carry):
                self.carried_allocations.setdefault(t.loop, set()).update(self.aliases[t.outputs[0]])
        tasks = list(self.tasks.values())
        for i, a in enumerate(tasks):
            for b in tasks[i+1:]:
                if self.exclusive(a.region, b.region):
                    continue
                self.conflicts(a, b)
        # Loop-carried hazards exist only for storage shared between iterations.
        for a in tasks:
            for b in tasks:
                common = []
                for x, y in zip(self.path(a.id), self.path(b.id)):
                    if x != y:
                        break
                    common.append(x)
                for loop in common:
                    self.conflicts(a, b, loop)
        for t in tuple(self.tasks.values()):
            if isinstance(t, p.Loop):
                read = self.effects[t.id][0]
                if any(read & writes for task, (_, writes) in self.effects.items()
                       if t.id in self.path(task)):
                    self.tasks[t.id] = replace(t, serial_modes=("eager", "strict"))

    def is_iterator(self, allocation):
        value = self.values.get(allocation)
        t = self.tasks.get(value.producer) if value else None
        return isinstance(t, p.Eval) and isinstance(t.expression.expression, ir.Intrinsic) and t.expression.expression.name == "enumerate"

    def exclusive(self, left, right):
        def choices(region):
            result = {}
            while region:
                r = self.regions[region]
                if r.kind in ("then", "else"):
                    result[r.owner] = r.kind
                region = r.parent
            return result
        a, b = choices(left), choices(right)
        return any(k in b and b[k] != v for k, v in a.items())

    def conflicts(self, a, b, loop=None):
        # Loop iteration effects are represented by serial_modes, not by a
        # self-deadlocking edge from a loop's completion into its own body.
        if (isinstance(a, p.Loop) and a.id in self.path(b.id)) or (isinstance(b, p.Loop) and b.id in self.path(a.id)):
            return
        ar, aw = self.effects[a.id]
        br, bw = self.effects[b.id]
        for hazard, overlap in (("RAW", aw & br), ("WAR", ar & bw), ("WAW", aw & bw)):
            if loop:
                overlap = {x for x in overlap if loop not in self.allocations.get(x, ())
                           or x in self.carried_allocations.get(loop, ())}
            if overlap:
                self.edge(a.id, b.id, "memory", carried=loop, distance=1 if loop else 0,
                          hazard=hazard, certainty="conservative")

    def share_requests(self):
        """Value-number requests without executing expressions or user code."""
        memo = {}

        def value_key(v, stack=frozenset()):
            if v in memo:
                return memo[v]
            if v in stack or self.aliases[v] & self.mutable or any(self.is_iterator(a) for a in self.aliases[v]):
                return ("version", v)
            t = self.tasks[self.values[v].producer]
            if self.effects[t.id][0] & self.mutable:
                result = ("version", v)
            elif isinstance(t, p.Eval):
                result = expr_key(t.expression.expression, {b.name: b.value for b in t.expression.bindings}, stack | {v})
            else:
                result = ("version", v)
            memo[v] = result
            return result

        def expr_key(e, bindings, stack=frozenset()):
            if isinstance(e, ir.Var):
                return value_key(bindings[e.name], stack)
            if isinstance(e, ir.Literal):
                ty, val = literal_key(e.value)
                return ("literal", ty.__name__, val)
            if isinstance(e, ir.State):
                return ("state",)
            def key(x):
                if isinstance(x, ir.Expr):
                    return expr_key(x, bindings, stack)
                if isinstance(x, tuple):
                    return tuple(key(v) for v in x)
                return x
            return (e.kind,) + tuple(key(getattr(e, f.name)) for f in fields(e))

        def scope_for(key):
            paths = []
            def walk(x):
                if isinstance(x, tuple):
                    if len(x) == 2 and x[0] == "version":
                        paths.append(self.path(self.values[x[1]].producer))
                    else:
                        for v in x:
                            walk(v)
            walk(key)
            return max(paths, key=len, default=())

        groups = {}
        for t in tuple(self.tasks.values()):
            if not isinstance(t, p.Infer):
                continue
            a = self.tasks[self.values[t.arguments].producer]
            expr = a.expression.expression
            bindings = {b.name: b.value for b in a.expression.bindings}
            args = tuple(expr_key(x, bindings) for x in expr.items[0].items)
            kwargs = {n: expr_key(x, bindings) for n, x in zip(t.keywords, expr.items[1].items)}
            # Signature binding is static metadata inspection, never a call.
            original = next(s for s in walk_statements(self.program.body) if s.id == t.origin)
            operation = original.operation
            signature = inspect.signature(getattr(Lev, operation))
            try:
                bound = signature.bind(None, *args, **kwargs)
            except TypeError as exc:
                raise PlanLoweringError(f"invalid {operation} arguments: {exc}", original) from exc
            canonical = []
            for name, parameter in signature.parameters.items():
                if name == "self":
                    continue
                if name in bound.arguments:
                    canonical.append((name, bound.arguments[name]))
                else:
                    canonical.append((name, expr_key(ir.Literal(parameter.default), {})))
            key = (operation, tuple(canonical))
            # A mutable argument's static binding can refer to different
            # contents at each demand; conservatively keep these sites apart.
            unsafe = self.identity or bool(self.aliases[t.arguments] & self.mutable)
            unsafe |= any(self.is_iterator(x) for x in self.aliases[t.arguments])
            scope = self.path(t.id) if unsafe else scope_for(key)
            if unsafe:
                key = (key, t.id)
            group_key = (key, scope)
            if group_key not in groups:
                groups[group_key] = [self.fresh("request"), operation, scope, [], not unsafe]
            group = groups[group_key]
            group[3].append(t.id)
            self.tasks[t.id] = replace(t, request=group[0])
        self.requests = [p.InferRequest(qid, operation, scope, tuple(sites), reusable)
                         for qid, operation, scope, sites, reusable in groups.values()]


def lower_to_plan(program: ir.Program) -> p.ExecutionPlan:
    """Build a validated plan for either failure mode without running Python."""
    if not isinstance(program, ir.Program):
        raise TypeError("lower_to_plan expects an ir.Program")
    try:
        # Also validate constructed Programs, not only deserialized ones.
        checked = ir.Program.from_dict(program.to_dict())
        return _Lowerer(checked).finish()
    except ir.IRDecodeError as exc:
        raise PlanLoweringError(f"invalid Lev IR: {exc}") from exc
