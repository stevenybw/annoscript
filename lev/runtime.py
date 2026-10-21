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
                               DECISION_OPERATIONS, BackendFatalError, InferenceError,
                               LLM_BATCH_DEFAULTS, validate_batch_options)
from lev.client import Result, ResultError
from lev._speculation import Speculation


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
    repair_attempt: int = 0
    repair_parent: str | None = None
    repair_feedback: str | None = None
    key: object = None
    speculative: bool = False
    discarded: bool = False
    origins: list = field(default_factory=list)
    error: object = None
    error_batch: str | None = None
    repairable: bool = False


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
        self._call_sequence = 0
        self.revision = 0
        self.speculation = Speculation(self) if self.mode == "speculative" else None
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
        self.revision += 1
        self.runtime._record("task_done", self, task=inst.task.id, path=inst.ctx, skipped=result is _SKIP)

    def request_key(self, task, ctx):
        request = self.requests[task.request]
        key = request.id, ctx[:len(request.scope)]
        if not request.reusable:
            key += (task.id, ctx)
        return key

    def call_id(self):
        return f"{self.id}-q{self._call_sequence}"

    def new_request(self, call, waiters, **options):
        self._call_sequence += 1
        return _Request(call, waiters, **options)

    def all_requests(self):
        requests = list(self.cache.values())
        if self.speculation is not None:
            seen = {id(r) for r in requests}
            requests.extend(r for r in self.speculation.requests if id(r) not in seen)
        return requests

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
            execution_mode = "eager" if self.speculation is not None else self.mode
            if not inst.exhausted and active < (1 if execution_mode in t.serial_modes else self.runtime.loop_window):
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
            key = self.request_key(t, ctx)
            args, kwvalues = self.value(t.arguments, ctx)
            # Every site's argument binding is checked even when its request is cached.
            call = normalize_call(self.call_id(), request.operation, args, dict(zip(t.keywords, kwvalues)), self.context)
            cached = self.cache.get(key)
            if cached is None and self.speculation is not None:
                self.speculation.retire_mismatches(key, call, t, ctx)
                cached = self.speculation.lookup(key, call)
                if cached is not None:
                    self.cache[key] = cached
                    cached.discarded = False
                    self.runtime._record("speculation_adopted", self, task=t.id, path=ctx, call=cached.call.id)
            if cached is None:
                cached = self.cache[key] = self.new_request(call, [inst], key=key)
                self.runtime._record("infer_ready", self, task=t.id, path=ctx, call=call.id, operation=call.operation)
            else:
                cached.waiters.append(inst)
                self.runtime._record("infer_reuse", self, task=t.id, path=ctx, call=cached.call.id)
            inst.status = "infer"
            if cached.status == "done":
                self.finish(inst, [cached.response])
            elif cached.status == "error":
                if cached.repairable:
                    self.runtime._repair_batch(self, [cached], {cached.call.id: cached.error}, cached.error_batch)
                else:
                    raise cached.error
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
                 loop_window=32, local_task_budget=1024, trace=False,
                 max_llm_batch_calls=8, max_llm_batch_input_bytes=262144,
                 max_llm_batch_output_tokens=16384, llm_batch_default_tokens=1024,
                 llm_batch_repair_attempts=1, request_slots=None):
        for name, value in (("max_active_runs", max_active_runs), ("max_decision_inflight", max_decision_inflight),
                            ("max_llm_inflight", max_llm_inflight), ("max_batch_questions", max_batch_questions),
                            ("loop_window", loop_window), ("local_task_budget", local_task_budget)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        batch_options = {name: value for name, value in locals().copy().items() if name in LLM_BATCH_DEFAULTS}
        validate_batch_options(batch_options)
        for name, value in batch_options.items():
            setattr(self, name, value)
        self.decision_backend, self.llm_backend = decision_backend, llm_backend
        self.trace = [] if trace else None
        self._loop = self._driver = None
        self._wake = None
        self._submissions, self._runs, self._inflight = {}, {}, {}
        self._events = deque()
        self._sequence = 0
        self.closed = False
        self.request_slots = request_slots
        self._slot_leases = {}
        self._slot_wait_started = {}

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
        if run.speculation is not None:
            run.speculation.prune(final=True)
        run.done = True
        run.stream.runs.discard(run)
        self._runs.pop(run.id, None)
        if self.request_slots is not None:
            self.request_slots.cancel((self, run.id))
            for kind in ("decision", "llm"):
                since = self._slot_wait_started.pop((run.id, kind), None)
                if since is not None:
                    self._record("slot_wait_cancelled", run, backend=kind, wait_seconds=monotonic() - since)
        if cancel and self.request_slots is None:
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

    def _send(self, run, requests, kind, batch_plan=None, slot_release=None):
        calls = [r.call for r in requests]
        for request in requests:
            request.status = "inflight"
        batch_id = self._id("http")
        backend = self.decision_backend if kind == "decision" else self.llm_backend
        details = {}
        if batch_plan is not None:
            details = {"input_bytes": batch_plan.input_bytes, "max_tokens": batch_plan.max_tokens}
        if requests[0].repair_attempt:
            details.update(repair_attempt=requests[0].repair_attempt, repair_parent=requests[0].repair_parent)
        self._record("http_submitted", run, batch=batch_id, backend=kind, model=getattr(backend, "model", None),
                     calls=[c.id for c in calls], questions=len(calls),
                     **({"speculative_calls": [r.call.id for r in requests if not r.waiters]}
                        if run.speculation is not None else {}), **details)
        if kind == "decision":
            options = {"allow_partial": True} if (run.speculation is not None and
                       getattr(backend, "supports_partial_answers", False) is True) else {}
            coroutine = backend.infer_batch(run.context, calls, **options)
        elif batch_plan is not None:
            prepared = ({"prepared": batch_plan.prepared} if
                        getattr(backend, "supports_prepared_batch", False) is True else {})
            coroutine = backend.infer_batch(run.context, calls, max_tokens=batch_plan.max_tokens, **prepared)
        elif requests[0].repair_attempt:
            resolve = getattr(backend, "resolve_token_budget", None)
            repair_budget = (resolve(calls[0], self.llm_batch_default_tokens) if resolve else
                             calls[0].arguments.get("max_num_tokens") or self.llm_batch_default_tokens)
            coroutine = backend.infer(run.context, calls[0], repair_feedback=requests[0].repair_feedback,
                max_tokens_override=repair_budget)
        else:
            coroutine = backend.infer(run.context, calls[0])
        future = asyncio.create_task(coroutine)
        self._inflight[future] = run, requests, kind, batch_id
        if slot_release is not None:
            self._slot_leases[future] = slot_release
        future.add_done_callback(self._completed)

    def _completed(self, future):
        release = self._slot_leases.pop(future, None)
        if release is not None:
            release()
            run, _, kind, _ = self._inflight[future]
            self._record("slot_released", run, backend=kind)
        self._events.append(future)
        self._wake.set()

    def _repair_batch(self, run, requests, errors, batch):
        """Only batch content failures get one separately scheduled single-call repair."""
        for request in requests:
            error = errors.get(request.call.id)
            if error is not None and (isinstance(error, BackendFatalError)
                    or error.code not in ("invalid_response", "invalid_batch_output", "incomplete_batch_generation")
                    or request.repair_attempt >= self.llm_batch_repair_attempts):
                self._fail_run(run, error, request.waiters[0])
                return False
        for request in requests:
            if request.call.id not in errors:
                continue
            request.repair_attempt += 1
            request.repair_parent = batch
            request.repair_feedback = (
                "回答缺失、被截断或不符合任务要求。请检查结构、数量、非空要求和原文来源；字符长度为建议"
            )
            request.status = "queued"
            request.error = None
            request.repairable = False
        return True

    def _speculative_error(self, run, request, error, batch, *, repairable):
        if isinstance(error, BackendFatalError):
            self._fail_run(run, error)
            return
        request.status, request.error = "error", error
        request.error_batch, request.repairable = batch, repairable
        run.revision += 1
        if request.waiters:
            if repairable:
                self._repair_batch(run, [request], {request.call.id: error}, batch)
            else:
                self._fail_run(run, error, request.waiters[0])
        else:
            self._record("speculation_error_deferred", run, call=request.call.id,
                         code=getattr(error, "code", "execution_failed"))

    def _receive(self):
        while self._events:
            future = self._events.popleft()
            run, requests, kind, batch = self._inflight.pop(future)
            metadata = {}
            try:
                output = future.result()
                metadata = output.metadata
                is_batch = kind == "decision" or len(requests) > 1
                responses = output.responses if is_batch else {requests[0].call.id: output.response}
                errors = dict(getattr(output, "errors", {})) if is_batch else {}
                if (set(responses) & set(errors) or set(responses) | set(errors) != {r.call.id for r in requests}
                        or any(not isinstance(error, InferenceError) for error in errors.values())
                        or (kind == "decision" and errors and run.speculation is None)):
                    raise InferenceError("backend returned incorrect call IDs",
                        code="invalid_batch_output" if kind == "llm" and is_batch else "invalid_response")
                for request in requests:
                    if request.call.id in responses:
                        try:
                            validate_response(request.call, responses[request.call.id])
                        except InferenceError as exc:
                            if run.speculation is None and (kind != "llm" or len(requests) < 2):
                                raise
                            errors[request.call.id] = exc
            except asyncio.CancelledError:
                self._record("http_cancelled", run, batch=batch, backend=kind)
            except Exception as exc:
                self._record("http_failed", run, batch=batch, backend=kind, **(getattr(exc, "metadata", {}) or metadata))
                if run.speculation is not None:
                    if not run.done:
                        for request in requests:
                            self._speculative_error(run, request, exc, batch,
                                repairable=kind == "llm" and len(requests) > 1 and
                                getattr(exc, "code", None) in ("invalid_batch_output", "incomplete_batch_generation"))
                            if run.done:
                                break
                    continue
                if (not run.done and kind == "llm" and len(requests) > 1
                        and getattr(exc, "code", None) in ("invalid_batch_output", "incomplete_batch_generation")):
                    self._repair_batch(run, requests, {r.call.id: exc for r in requests}, batch)
                    continue
                failed = next((r for r in requests if r.call.id == getattr(exc, "call_id", None)), requests[0])
                self._fail_run(run, exc, failed.waiters[0])
            else:
                self._record("http_completed", run, batch=batch, backend=kind,
                             **({"failed_calls": list(errors)} if errors else {}), **metadata)
                if run.done:
                    continue
                if errors and run.speculation is not None:
                    for request in requests:
                        if request.call.id in errors:
                            self._speculative_error(run, request, errors[request.call.id], batch,
                                                    repairable=kind == "llm" and len(requests) > 1)
                            if run.done:
                                break
                    if run.done:
                        continue
                elif errors and not self._repair_batch(run, requests, errors, batch):
                    continue
                for request in requests:
                    if request.call.id in errors:
                        continue
                    request.status = "done"
                    request.error = None
                    request.response = responses[request.call.id]
                    run.revision += 1
                    for inst in request.waiters:
                        run.finish(inst, [request.response])

    def _dispatch(self):
        counts = {kind: sum(info[2] == kind for info in self._inflight.values()) for kind in ("decision", "llm")}
        # One batch/call per run per pass provides round-robin fairness.
        more = True
        while more:
            more = False
            for run in list(self._runs.values()):
                if run.speculation is not None:
                    if not run.speculation.ready:
                        continue
                    run.speculation.prune()
                queued = [r for r in run.all_requests() if r.status == "queued" and not r.discarded]
                for kind, limit in (("decision", self.max_decision_inflight), ("llm", self.max_llm_inflight)):
                    eligible = [r for r in queued if (r.call.operation in DECISION_OPERATIONS) == (kind == "decision") and r.status == "queued"]
                    if eligible and counts[kind] < limit:
                        release = None
                        if self.request_slots is not None:
                            owner = (self, run.id)
                            release = self.request_slots.acquire(kind, owner, self._wake.set)
                            key = (run.id, kind)
                            if release is None:
                                if key not in self._slot_wait_started:
                                    self._slot_wait_started[key] = monotonic()
                                    self._record("slot_wait", run, backend=kind)
                                continue
                            since = self._slot_wait_started.pop(key, None)
                            self._record("slot_acquired", run, backend=kind,
                                         wait_seconds=0 if since is None else monotonic() - since)
                        plan = None
                        if kind == "decision":
                            selected = eligible[:self.max_batch_questions]
                        else:
                            repairs = [r for r in eligible if r.repair_attempt]
                            if not repairs and any(r.status == "queued" and r.repair_attempt
                                    for other in self._runs.values() for r in other.all_requests()):
                                if release is not None:
                                    release()
                                continue
                            selected = (repairs or eligible)[:1]
                            if (not repairs and self.max_llm_batch_calls > 1
                                    and getattr(self.llm_backend, "supports_llm_batch", False) is True):
                                try:
                                    plan = self.llm_backend.plan_batch(run.context, [r.call for r in eligible],
                                        max_calls=self.max_llm_batch_calls, max_input_bytes=self.max_llm_batch_input_bytes,
                                        max_output_tokens=self.max_llm_batch_output_tokens, default_tokens=self.llm_batch_default_tokens)
                                except BaseException:
                                    if release is not None:
                                        release()
                                    raise
                                if plan is not None:
                                    selected = eligible[:len(plan.calls)]
                                    if tuple(r.call for r in selected) != plan.calls or len(selected) < 2:
                                        if release is not None:
                                            release()
                                        raise ValueError("backend selected an invalid batch prefix")
                        try:
                            self._send(run, selected, kind, plan, slot_release=release)
                        except BaseException:
                            if release is not None:
                                release()
                            raise
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
                    if not run.done and run.speculation is not None:
                        progress += run.speculation.advance(self.local_task_budget)
                self._dispatch()
                for stream in list(self._submissions.values()):
                    if not stream.closed and not stream.finished and not stream.pending and not stream.runs:
                        stream.finished = True
                        stream.queue.put_nowait(_END)
                if progress:
                    await asyncio.sleep(0)
                else:
                    waiting_for_slots = (self.request_slots is not None and any(
                        self.request_slots.waiting((self, run.id)) for run in self._runs.values()))
                    if self._runs and not self._inflight and not waiting_for_slots:
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
        if self.request_slots is None:
            for task in pending:
                task.cancel()
        # Shared slots represent actual work, including orphaned responses.
        # Preserve the transport's existing timeout and accounting on cancellation.
        draining = asyncio.gather(*pending, *([self._driver] if self._driver else []), return_exceptions=True)
        cancelled = False
        try:
            await asyncio.shield(draining)
        except asyncio.CancelledError:
            await draining
            cancelled = True
        self._receive()
        backends = [self.decision_backend]
        if self.llm_backend is not self.decision_backend:
            backends.append(self.llm_backend)
        await asyncio.gather(*(backend.aclose() for backend in backends))
        if cancelled:
            raise asyncio.CancelledError()
