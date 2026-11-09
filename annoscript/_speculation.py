"""A bounded, private argument evaluator; never commits Plan task effects.

Formal execution still owns branches, memory, checks, returns and loop progress.
Only normalized, reusable inference requests cross from this evaluator to the
transport scheduler. Values are rebuilt after formal progress or a response.
Each revision reaches Decision quiescence before collecting generation work;
generation requires confirmed control gates, even before formal branch progress.
"""

from copy import deepcopy

from annoscript import plan as p
from annoscript._expressions import expression, unpack
from annoscript.backends.base import normalize_call, DECISION_OPERATIONS
from annoscript.backends._bounded_grammar import ensure_satisfiable


class Unavailable(Exception):
    pass


def detached(value):
    """Reject consumable iterators, including those nested in containers."""
    seen = set()
    def check(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if type(item) in (list, tuple):
            for child in item:
                check(child)
        elif type(item) is dict:
            for child in item.values():
                check(child)
        elif hasattr(item, "__next__"):
            raise Unavailable()
    check(value)
    return deepcopy(value)


class Speculation:
    def __init__(self, run):
        self.run = run
        self.sites = tuple(t for t in run.plan.tasks if isinstance(t, p.Infer))
        self.requests = []
        self.scan = None
        self.revision = -1
        self.ready = False
        self.phase = "decision"
        self.selected_requests = set()

    def viable(self, task, ctx):
        return all(self.run.port(g.task, ctx, g.port) is not False for g in task.gates)

    def memory_ready(self, task, ctx):
        return all(self.run.dependency(d, ctx) for d in self.run.incoming[task.id]
                   if d.kind == "memory")

    def port(self, tid, ctx, port, visiting):
        """Read a control decision without launching regions or committing effects."""
        run = self.run
        known = run.port(tid, ctx, port)
        if type(known) is bool:
            return known
        task = run.tasks[tid]
        parent = ctx[:len(run.paths[tid])]
        key = tid, parent, port
        yield
        if key in visiting or not self.memory_ready(task, parent):
            raise Unavailable()
        visiting = visiting | {key}
        yield from self.selected(task, parent, visiting)
        if isinstance(task, p.Branch) and port in ("true", "false"):
            selected = bool((yield from self.bound(task.condition, parent, visiting)))
            return selected if port == "true" else not selected
        if isinstance(task, p.Loop) and port == "enter":
            child = ctx[:len(parent) + 1]
            if len(child) > len(parent) and ((tid, child) in self.items or (tid, child) in run.items):
                return True
        # Join/return completion and mutable iteration remain formally owned.
        raise Unavailable()

    def selected(self, task, ctx, visiting=frozenset()):
        for gate in task.gates:
            if not (yield from self.port(gate.task, ctx, gate.port, visiting)):
                raise Unavailable()

    def possible(self, task, ctx, visiting=frozenset()):
        for gate in task.gates:
            try:
                known = yield from self.port(gate.task, ctx, gate.port, visiting)
            except Unavailable:
                continue
            if not known:
                return False
        return True

    def lookup(self, key, call):
        for request in self.requests:
            if request.key == key and self.same(request.call, call):
                return request
        return None

    def retire_mismatches(self, key, call, task, ctx):
        for request in self.requests:
            if request.key == key and not self.same(request.call, call):
                origin = task.id, ctx
                if origin in request.origins:
                    request.origins.remove(origin)

    @staticmethod
    def same(left, right):
        return (left.operation == right.operation and left.arguments == right.arguments
                and left.extraction_source == right.extraction_source)

    def request(self, task, ctx, arguments):
        run = self.run
        definition = run.requests[task.request]
        if not definition.reusable:
            raise Unavailable()
        args, values = arguments
        call = normalize_call(run.call_id(), definition.operation, args,
                              dict(zip(task.keywords, values)), run.context)
        key = run.request_key(task, ctx)
        request = run.cache.get(key)
        if request is not None:
            if not self.same(request.call, call):
                raise Unavailable()
        else:
            request = self.lookup(key, call)
        if request is None:
            kind = "decision" if call.operation in DECISION_OPERATIONS else "llm"
            if kind != self.phase:
                raise Unavailable()
            request = run.new_request(call, [], key=key, speculative=True)
            self.requests.append(request)
            if call.operation not in DECISION_OPERATIONS:
                try:
                    ensure_satisfiable(call)
                except Exception as exc:
                    request.error, request.status = exc, "error"
                    run.runtime._record("speculation_error_deferred", run, call=call.id,
                                        code=getattr(exc, "code", "invalid_arguments"))
            run.runtime._record("infer_speculated", run, task=task.id, path=ctx,
                                call=call.id, operation=call.operation)
        origin = task.id, ctx
        if origin not in request.origins:
            request.origins.append(origin)
            request.discarded = False
        if self.phase == "llm" and call.operation not in DECISION_OPERATIONS:
            self.selected_requests.add(id(request))
        return request

    def bound(self, bound, ctx, visiting):
        env = {}
        for binding in bound.bindings:
            env[binding.name] = yield from self.value(binding.value, ctx, visiting)
        # Allocations remain private and cannot be addressed by formal Store.
        return expression(bound.expression, env, self.run.input.state, {})

    def value(self, vid, ctx, visiting=frozenset()):
        run = self.run
        task = run.tasks[run.values[vid].producer]
        ctx = ctx[:len(run.paths[task.id])]
        key = vid, ctx
        yield
        if key in visiting:
            raise Unavailable()
        if key in self.memo:
            return self.memo[key]
        if not self.viable(task, ctx) or not self.memory_ready(task, ctx):
            raise Unavailable()
        if self.phase == "llm":
            yield from self.selected(task, ctx, visiting)
        elif not (yield from self.possible(task, ctx, visiting)):
            raise Unavailable()
        if key in run.outputs:
            return detached(run.outputs[key])
        visiting = visiting | {key}
        if isinstance(task, p.Eval):
            values = [(yield from self.bound(task.expression, ctx, visiting))]
        elif isinstance(task, p.Unpack):
            value = yield from self.bound(task.expression, ctx, visiting)
            values = unpack(task.target, detached(value))
        elif isinstance(task, p.Infer):
            arguments = yield from self.value(task.arguments, ctx, visiting)
            request = self.request(task, ctx, arguments)
            if request.status != "done":
                raise Unavailable()
            values = [request.response]
        elif isinstance(task, p.Merge):
            selected = yield from self.port(task.branch, ctx, "true", visiting)
            source = task.then_value if selected else task.else_value
            if source is None:
                raise Unavailable()
            values = [(yield from self.value(source, ctx, visiting))]
        elif isinstance(task, p.Carry):
            source = task.initial if ctx[-1] == 0 else task.update
            parent = ctx[:-1] if ctx[-1] == 0 else ctx[:-1] + (ctx[-1] - 1,)
            values = [(yield from self.value(source, parent, visiting))]
        elif isinstance(task, p.Iteration):
            item_key = task.loop, ctx
            if item_key not in self.items and item_key not in run.items:
                raise Unavailable()
            item = self.items[item_key] if item_key in self.items else run.items[item_key]
            values = unpack(task.target, detached(item))
        else:
            raise Unavailable()
        for output, value in zip(task.outputs, values):
            self.memo[output, ctx] = detached(value)
        return self.memo[key]

    def contexts(self, path, ctx=()):
        if not path:
            yield ctx
            return
        run = self.run
        loop = run.tasks[path[0]]
        if not self.viable(loop, ctx) or not self.memory_ready(loop, ctx):
            return
        if self.phase == "llm":
            try:
                yield from self.selected(loop, ctx)
            except Unavailable:
                return
        inst = run.instances.get((loop.id, ctx))
        existing = inst.iterations if inst else []
        start = next((i for i in existing if not run.region_done(loop.body, ctx + (i,))), len(existing))
        window = 1 if "eager" in loop.serial_modes else run.runtime.loop_window
        sequence = None
        # Mutable iteration must remain owned by the formal executor.
        if "eager" not in loop.serial_modes:
            try:
                sequence = yield from self.bound(loop.iterable, ctx, frozenset())
                if type(sequence) is dict:
                    sequence = tuple(sequence)
                if type(sequence) not in (str, list, tuple, range):
                    sequence = None
            except Exception:
                sequence = None
        for index in range(start, start + window):
            yield None  # Charge loop expansion to the local scheduling budget.
            child = ctx + (index,)
            if (loop.id, child) in run.items:
                item = run.items[loop.id, child]
            elif sequence is not None and index < len(sequence):
                item = sequence[index]
            else:
                break
            try:
                self.items[loop.id, child] = detached(item)
            except Exception:
                break
            yield from self.contexts(path[1:], child)

    def candidates(self):
        for task in self.sites:
            definition = self.run.requests[task.request]
            kind = "decision" if definition.operation in DECISION_OPERATIONS else "llm"
            if not definition.reusable or kind != self.phase:
                continue
            for ctx in self.contexts(self.run.paths[task.id]):
                if ctx is None:
                    yield
                    continue
                yield
                if not self.viable(task, ctx) or not self.memory_ready(task, ctx):
                    continue
                inst = self.run.instances.get((task.id, ctx))
                if inst is not None and inst.status == "done":
                    continue
                try:
                    if self.phase == "llm":
                        yield from self.selected(task, ctx)
                    elif not (yield from self.possible(task, ctx)):
                        continue
                    arguments = yield from self.value(task.arguments, ctx)
                    self.request(task, ctx, arguments)
                except Exception:
                    # Argument failures on an unselected path have no effects.
                    # Formal execution re-evaluates and reports them on demand.
                    continue

    def advance(self, budget):
        if self.revision != self.run.revision:
            self.scan = None
            self.ready = False
            self.revision = self.run.revision
            self.phase = "decision"
            self.selected_requests.clear()
        count = 0
        while count < budget:
            if self.ready:
                self.prune()
                if (self.phase != "decision" or not self.run.local_ready or any(
                        r.call.operation in DECISION_OPERATIONS and r.status in ("queued", "inflight")
                        and (not r.discarded or r.status == "inflight") for r in self.run.all_requests())):
                    break
                self.phase, self.ready = "llm", False
                count += 1
                continue
            if self.scan is None:
                self.memo, self.items = {}, {}
                self.scan = self.candidates()
            try:
                next(self.scan)
            except StopIteration:
                self.scan = None
                self.ready = True
                continue
            count += 1
        return count

    def prune(self, *, final=False):
        for request in self.requests:
            if request.waiters or request.discarded:
                continue
            if final or not any(self.viable(self.run.tasks[tid], ctx) for tid, ctx in request.origins):
                request.discarded = True
                self.run.runtime._record("speculation_discarded", self.run, call=request.call.id,
                                         status=request.status, failed=request.error is not None)
