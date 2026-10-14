"""Dependency-driven execution with runtime-owned, per-state Decision batches.

Only network requests get asyncio Tasks. Local Plan tasks are advanced by one
event-loop owner; no source code, Python program or test executor is executed.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from time import monotonic

from lev import plan as p
from lev._data import snapshot
from lev._expressions import expression, unpack
from lev.backends.base import (InferenceContext, normalize_call, validate_response,
                               DECISION_OPERATIONS, BackendFatalError, InferenceError)
from lev.client import Result, ResultError


_WAIT, _SKIP, _END = object(), object(), object()


@dataclass(frozen=True)
class RunInput:
    index: int
    state: object


class ConstraintError(ValueError):
    pass


@dataclass
class _Instance:
    task: p.Task
    ctx: tuple
    status: str = "pending"
    result: object = None
    iterator: object = None
    iterations: list = field(default_factory=list)
    exhausted: bool = False
    entered: bool = False


@dataclass
class _Request:
    call: object
    waiters: list
    status: str = "queued"
    response: object = None


class AsyncResultStream:
    """A single-consumer submission stream; closing cancels only this submission."""
    def __init__(self, runtime, submission_id, plan, inputs, mode):
        self.runtime, self.id, self.plan, self.mode = runtime, submission_id, plan, mode
        self.pending = deque(inputs)
        self.runs = set()
        self.queue = asyncio.Queue()
        self.closed = False
        self.finished = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.runtime._bind()
        if self.closed:
            raise StopAsyncIteration
        try:
            item = await self.queue.get()
        except BaseException:
            await self.aclose()
            raise
        if item is _END:
            await self.aclose()
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            await self.aclose()
            raise item
        return item

    async def __aenter__(self):
        if self.closed:
            raise RuntimeError("result stream is closed")
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        self.runtime._bind()
        if not self.closed:
            self.closed = True
            self.pending.clear()
            for run in list(self.runs):
                self.runtime._remove_run(run, cancel=True)
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queue.put_nowait(_END)
            self.runtime._submissions.pop(self.id, None)
            self.runtime._wake.set()


class _Run:
    def __init__(self, runtime, stream, input, run_id):
        self.runtime, self.stream, self.input, self.id = runtime, stream, input, run_id
        self.plan, self.mode = stream.plan, stream.mode
        self.context = InferenceContext(input.state, self.plan.config.system_prompt)
        self.tasks = {t.id: t for t in self.plan.tasks}
        self.regions = {r.id: r for r in self.plan.regions}
        self.values = {v.id: v for v in self.plan.values}
        self.requests = {r.id: r for r in self.plan.requests}
        self.incoming = {t.id: [] for t in self.plan.tasks}
        for dep in self.plan.dependencies_for(self.mode):
            self.incoming[dep.target].append(dep)
        self.paths = {}
        for t in self.plan.tasks:
            path, region = [], self.regions[t.region]
            while region.parent is not None:
                if region.kind == "loop":
                    path.append(region.owner)
                region = self.regions[region.parent]
            self.paths[t.id] = tuple(reversed(path))
        self.instances, self.region_runs, self.outputs, self.items = {}, {}, {}, {}
        self.pending = deque()
        self.allocations = {}
        self.cache = {}
        self.return_values = []
        self.done = False
        self.root = next(r.id for r in self.plan.regions if r.kind == "root")
        self.launch(self.root, ())

    def launch(self, region, ctx):
        keys = []
        for tid in self.regions[region].tasks:
            key = tid, ctx
            self.instances[key] = _Instance(self.tasks[tid], ctx)
            self.pending.append(key)
            keys.append(key)
        self.region_runs[region, ctx] = keys

    def instance(self, tid, ctx):
        ctx = ctx[:len(self.paths[tid])]
        found = self.instances.get((tid, ctx))
        if found is not None:
            return found
        region = self.regions[self.tasks[tid].region]
        if region.kind in ("then", "else"):
            owner = self.instance(region.owner, ctx)
            if owner is _SKIP:
                return _SKIP
            if owner is not _WAIT and owner.status == "done" and (owner.result is _SKIP or bool(owner.result) != (region.kind == "then")):
                return _SKIP
        return _WAIT

    def port(self, tid, ctx, port="done"):
        inst = self.instance(tid, ctx)
        if inst is _SKIP:
            return False
        if inst is _WAIT:
            return _WAIT
        if port == "enter" and inst.entered:
            return True
        if inst.status != "done":
            return _WAIT
        if inst.result is _SKIP:
            return False
        if port == "false":
            return not inst.result
        if port in ("true", "continue"):
            return bool(inst.result)
        return True

    def dependency(self, dep, ctx):
        if dep.condition:
            condition = self.port(dep.condition.task, ctx, dep.condition.port)
            if condition is _WAIT:
                return False
            if not condition:
                return True
        m = dep.mapping
        common = 0
        for a, b in zip(m.source_loops, m.target_loops):
            if a != b:
                break
            common += 1
        if m.carried_loop is not None:
            common = min(common, m.target_loops.index(m.carried_loop) + 1)
        prefix = list(ctx[:common])
        if m.carried_loop is not None:
            i = m.target_loops.index(m.carried_loop)
            prefix[i] -= m.distance
            if prefix[i] < 0:
                return True
        contexts = [tuple(prefix)]
        for loop in m.source_loops[common:]:
            expanded = []
            for parent in contexts:
                inst = self.instance(loop, parent)
                if inst is _WAIT or inst is not _SKIP and inst.status != "done":
                    return False
                if inst is not _SKIP and inst.result is not _SKIP:
                    expanded.extend(parent + (i,) for i in inst.iterations)
            contexts = expanded
        return all(self.port(dep.source, c, dep.port) is not _WAIT for c in contexts)

    def value(self, vid, ctx):
        producer = self.values[vid].producer
        return self.outputs[vid, ctx[:len(self.paths[producer])]]

    def expr(self, bound, ctx):
        env = {b.name: self.value(b.value, ctx) for b in bound.bindings}
        return expression(bound.expression, env, self.input.state, self.allocations)

    def region_done(self, region, ctx):
        return all(self.instances[k].status == "done" for k in self.region_runs[region, ctx])

    def region_returned(self, region, ctx):
        return any((isinstance(i.task, p.Return) and i.result is not _SKIP) or
                   (isinstance(i.task, p.Join) and i.result is False)
                   for i in (self.instances[k] for k in self.region_runs[region, ctx]))

    def finish(self, inst, values=(), result=None):
        for vid, value in zip(inst.task.outputs, values):
            self.outputs[vid, inst.ctx] = value
        inst.status, inst.result = "done", result
        self.runtime._record("task_done", self, task=inst.task.id, path=inst.ctx, skipped=result is _SKIP)

    def step(self, inst):
        t, ctx = inst.task, inst.ctx
        if inst.status == "pending":
            for gate in t.gates:
                decision = self.port(gate.task, ctx, gate.port)
                if decision is _WAIT:
                    return False
                if not decision:
                    self.finish(inst, result=_SKIP)
                    return True
            if not all(self.dependency(d, ctx) for d in self.incoming[t.id]):
                return False
        if isinstance(t, p.Loop):
            if not inst.entered:
                inst.iterator = iter(self.expr(t.iterable, ctx))
                inst.entered, inst.status = True, "loop"
                return True
            active = sum(not self.region_done(t.body, ctx + (i,)) for i in inst.iterations)
            if not inst.exhausted and active < (1 if self.mode in t.serial_modes else self.runtime.loop_window):
                try:
                    item = next(inst.iterator)
                except StopIteration:
                    inst.exhausted = True
                else:
                    index = len(inst.iterations)
                    self.items[t.id, ctx + (index,)] = item
                    inst.iterations.append(index)
                    self.launch(t.body, ctx + (index,))
                return True
            if inst.exhausted and active == 0:
                values = [self.value(e.update, ctx + (inst.iterations[-1],)) if inst.iterations else self.value(e.initial, ctx) for e in t.exports]
                self.finish(inst, values)
                return True
            return False
        if isinstance(t, p.Join):
            branch = self.tasks[t.branch]
            selected = self.instance(t.branch, ctx).result
            region = branch.then_region if selected else branch.else_region
            if not self.region_done(region, ctx):
                return False
            self.finish(inst, result=not self.region_returned(region, ctx))
        elif isinstance(t, p.Eval):
            self.finish(inst, [self.expr(t.expression, ctx)])
        elif isinstance(t, p.Unpack):
            self.finish(inst, unpack(t.target, self.expr(t.expression, ctx)))
        elif isinstance(t, p.Store):
            container = self.expr(t.container, ctx)
            index = self.expr(t.index, ctx)
            item = self.expr(t.item, ctx)
            if self.allocations.get(id(container)) is not container:
                raise ValueError("Store requires a locally constructed container")
            container[index] = item
            self.finish(inst)
        elif isinstance(t, p.Infer):
            request = self.requests[t.request]
            key = request.id, ctx[:len(request.scope)]
            if not request.reusable:
                key += (t.id, ctx)
            args, kwvalues = self.value(t.arguments, ctx)
            # Every site's argument binding is checked even when its request is cached.
            call = normalize_call(f"{self.id}-q{len(self.cache)}", request.operation, args, dict(zip(t.keywords, kwvalues)), self.context)
            cached = self.cache.get(key)
            if cached is None:
                cached = self.cache[key] = _Request(call, [inst])
                self.runtime._record("infer_ready", self, task=t.id, path=ctx, call=call.id, operation=call.operation)
            else:
                cached.waiters.append(inst)
                self.runtime._record("infer_reuse", self, task=t.id, path=ctx, call=cached.call.id)
            inst.status = "infer"
            if cached.status == "done":
                self.finish(inst, [cached.response])
        elif isinstance(t, p.Branch):
            selected = bool(self.expr(t.condition, ctx))
            self.launch(t.then_region if selected else t.else_region, ctx)
            self.finish(inst, result=selected)
        elif isinstance(t, p.Merge):
            selected = self.instance(t.branch, ctx).result
            value = t.then_value if selected else t.else_value
            self.finish(inst, [self.value(value, ctx)] if value else (), result=None if value else _SKIP)
        elif isinstance(t, p.Carry):
            value = self.value(t.initial, ctx[:-1]) if ctx[-1] == 0 else self.value(t.update, ctx[:-1] + (ctx[-1]-1,))
            self.finish(inst, [value])
        elif isinstance(t, p.Iteration):
            self.finish(inst, unpack(t.target, self.items[t.loop, ctx]))
        elif isinstance(t, p.Check):
            condition = self.expr(t.condition, ctx)
            message = self.expr(t.message, ctx) if t.message is not None else None
            if not condition:
                raise ConstraintError(message or "Constraint failed")
            self.finish(inst)
        elif isinstance(t, p.Return):
            value = self.expr(t.expression, ctx)
            self.return_values.append(value)
            self.finish(inst, result=value)
        else:
            raise ValueError("unsupported plan task")
        return True

    def advance(self, budget):
        count = 0
        while count < budget and not self.done:
            changed = False
            for _ in range(len(self.pending)):
                inst = self.instances[self.pending.popleft()]
                if inst.status not in ("pending", "loop"):
                    continue
                try:
                    progress = self.step(inst)
                except Exception as exc:
                    self.runtime._fail_run(self, exc, inst)
                    return count + 1
                if inst.status in ("pending", "loop"):
                    self.pending.append((inst.task.id, inst.ctx))
                if progress:
                    changed = True
                    count += 1
                    if count >= budget:
                        break
            if not changed:
                break
        if not self.done and self.region_done(self.root, ()):
            try:
                if len(self.return_values) != 1:
                    raise ValueError("plan must produce exactly one return value")
                value = snapshot(self.return_values[0])
            except Exception as exc:
                self.runtime._fail_run(self, exc)
            else:
                self.stream.queue.put_nowait(Result(self.input.index, "success", value))
                self.runtime._record("run_done", self)
                self.runtime._remove_run(self)
        return count


class LevRuntime:
    def __init__(self, decision_backend, llm_backend, *, max_active_runs=8,
                 max_decision_inflight=4, max_llm_inflight=4, max_batch_questions=32,
                 loop_window=32, local_task_budget=1024, trace=False):
        for name, value in (("max_active_runs", max_active_runs), ("max_decision_inflight", max_decision_inflight),
                            ("max_llm_inflight", max_llm_inflight), ("max_batch_questions", max_batch_questions),
                            ("loop_window", loop_window), ("local_task_budget", local_task_budget)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        self.decision_backend, self.llm_backend = decision_backend, llm_backend
        self.trace = [] if trace else None
        self._loop = self._driver = None
        self._wake = None
        self._submissions, self._runs, self._inflight = {}, {}, {}
        self._events = deque()
        self._sequence = 0
        self.closed = False

    def _bind(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._wake = asyncio.Event()
        elif self._loop is not loop:
            raise RuntimeError("LevRuntime belongs to another event loop")

    def _id(self, prefix):
        self._sequence += 1
        return f"{prefix}{self._sequence}"

    def _record(self, event, run=None, **fields):
        if self.trace is not None:
            self.trace.append({"event": event, "time": monotonic(),
                               **({"run": run.id, "submission": run.stream.id, "index": run.input.index} if run else {}), **fields})

    async def __aenter__(self):
        self._bind()
        if self.closed:
            raise RuntimeError("runtime is closed")
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def submit(self, plan, inputs, *, mode="eager"):
        self._bind()
        if self.closed:
            raise RuntimeError("runtime is closed")
        if self._driver is not None and self._driver.done():
            raise BackendFatalError("runtime scheduler has stopped", code="runtime_failed")
        if not isinstance(plan, p.ExecutionPlan):
            raise TypeError("submit requires ExecutionPlan")
        plan.validate()
        plan.dependencies_for(mode)
        snapshots, indices = [], set()
        for item in inputs:
            if not isinstance(item, RunInput) or type(item.index) is not int or item.index < 0 or item.index in indices:
                raise ValueError("inputs require unique nonnegative indices")
            indices.add(item.index)
            snapshots.append(RunInput(item.index, snapshot(item.state, state=True)))
        stream = AsyncResultStream(self, self._id("submission"), plan, snapshots, mode)
        self._submissions[stream.id] = stream
        if self._driver is None:
            self._driver = asyncio.create_task(self._drive())
        self._wake.set()
        return stream

    def _remove_run(self, run, cancel=False):
        run.done = True
        run.stream.runs.discard(run)
        self._runs.pop(run.id, None)
        if cancel:
            for future, (owner, _, _, _) in list(self._inflight.items()):
                if owner is run:
                    future.cancel()

    def _fail_run(self, run, error, inst=None):
        if run.done:
            return
        if isinstance(error, BackendFatalError):
            self._fail_submission(run.stream, error)
            return
        code = getattr(error, "code", "constraint_unsatisfied" if isinstance(error, ConstraintError) else "execution_failed")
        message = str(error)
        if inst is not None:
            span = inst.task.source.span
            message += f" [{span.module}:{span.line}:{span.column}, task={inst.task.id}, path={inst.ctx}]"
            for call in inst.task.source.calls:
                message += f" [inlined at {call.module}:{call.line}:{call.column}]"
        run.stream.queue.put_nowait(Result(run.input.index, "error", error=ResultError(code, message)))
        self._record("run_failed", run, code=code, task=inst.task.id if inst else None, path=inst.ctx if inst else ())
        self._remove_run(run, cancel=True)

    def _fail_submission(self, stream, error):
        stream.pending.clear()
        for run in list(stream.runs):
            self._remove_run(run, cancel=True)
        # Fatal transport errors must be delivered before buffered successes.
        while not stream.queue.empty():
            stream.queue.get_nowait()
        stream.queue.put_nowait(error)
        stream.finished = True

    def _admit(self):
        available = [s for s in self._submissions.values() if not s.closed and not s.finished and s.pending]
        while available and len(self._runs) < self.max_active_runs:
            for stream in available:
                if stream.pending and len(self._runs) < self.max_active_runs:
                    run = _Run(self, stream, stream.pending.popleft(), self._id("run"))
                    stream.runs.add(run)
                    self._runs[run.id] = run
                    self._submissions.pop(stream.id)
                    self._submissions[stream.id] = stream
                    self._record("run_started", run, entry=stream.plan.entry)
            available = [s for s in available if s.pending]

    def _send(self, run, requests, kind):
        calls = [r.call for r in requests]
        for request in requests:
            request.status = "inflight"
        batch_id = self._id("http")
        backend = self.decision_backend if kind == "decision" else self.llm_backend
        self._record("http_submitted", run, batch=batch_id, backend=kind, model=getattr(backend, "model", None), calls=[c.id for c in calls], questions=len(calls))
        coroutine = self.decision_backend.infer_batch(run.context, calls) if kind == "decision" else self.llm_backend.infer(run.context, calls[0])
        future = asyncio.create_task(coroutine)
        self._inflight[future] = run, requests, kind, batch_id
        future.add_done_callback(self._completed)

    def _completed(self, future):
        self._events.append(future)
        self._wake.set()

    def _receive(self):
        while self._events:
            future = self._events.popleft()
            run, requests, kind, batch = self._inflight.pop(future)
            metadata = {}
            try:
                output = future.result()
                metadata = output.metadata
                responses = output.responses if kind == "decision" else {requests[0].call.id: output.response}
                if set(responses) != {r.call.id for r in requests}:
                    raise InferenceError("backend returned incorrect call IDs", code="invalid_response")
                for request in requests:
                    validate_response(request.call, responses[request.call.id])
            except asyncio.CancelledError:
                self._record("http_cancelled", run, batch=batch, backend=kind)
            except Exception as exc:
                self._record("http_failed", run, batch=batch, backend=kind, **(getattr(exc, "metadata", {}) or metadata))
                failed = next((r for r in requests if r.call.id == getattr(exc, "call_id", None)), requests[0])
                self._fail_run(run, exc, failed.waiters[0])
            else:
                self._record("http_completed", run, batch=batch, backend=kind, **metadata)
                if run.done:
                    continue
                for request in requests:
                    request.status = "done"
                    request.response = responses[request.call.id]
                    for inst in request.waiters:
                        run.finish(inst, [request.response])

    def _dispatch(self):
        counts = {kind: sum(info[2] == kind for info in self._inflight.values()) for kind in ("decision", "llm")}
        # One batch/call per run per pass provides round-robin fairness.
        more = True
        while more:
            more = False
            for run in list(self._runs.values()):
                queued = [r for r in run.cache.values() if r.status == "queued"]
                for kind, limit in (("decision", self.max_decision_inflight), ("llm", self.max_llm_inflight)):
                    eligible = [r for r in queued if (r.call.operation in DECISION_OPERATIONS) == (kind == "decision") and r.status == "queued"]
                    if eligible and counts[kind] < limit:
                        self._send(run, eligible[:self.max_batch_questions if kind == "decision" else 1], kind)
                        self._runs.pop(run.id)
                        self._runs[run.id] = run
                        counts[kind] += 1
                        more = True

    async def _drive(self):
        try:
            while not self.closed:
                self._wake.clear()
                self._receive()
                self._admit()
                progress = 0
                for run in list(self._runs.values()):
                    progress += run.advance(self.local_task_budget)
                self._dispatch()
                for stream in list(self._submissions.values()):
                    if not stream.closed and not stream.finished and not stream.pending and not stream.runs:
                        stream.finished = True
                        stream.queue.put_nowait(_END)
                if progress:
                    await asyncio.sleep(0)
                else:
                    if self._runs and not self._inflight:
                        for run in list(self._runs.values()):
                            self._fail_run(run, ValueError("plan cannot make progress"))
                        continue
                    await self._wake.wait()
        except asyncio.CancelledError:
            pass
        except Exception:
            for stream in list(self._submissions.values()):
                if not stream.closed:
                    self._fail_submission(stream, BackendFatalError("runtime scheduler failed", code="runtime_failed"))

    async def aclose(self):
        self._bind()
        if self.closed:
            return
        self.closed = True
        for stream in list(self._submissions.values()):
            await stream.aclose()
        if self._driver is not None:
            self._driver.cancel()
        pending = list(self._inflight)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, *([self._driver] if self._driver else []), return_exceptions=True)
        self._receive()
        backends = [self.decision_backend]
        if self.llm_backend is not self.decision_backend:
            backends.append(self.llm_backend)
        await asyncio.gather(*(backend.aclose() for backend in backends))
