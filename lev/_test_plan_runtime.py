"""Test-only dependency executor. No backend, retries, batching or solver.

Consumes the serialized Plan directly. asyncio futures model in-flight Infer
requests; fake providers may return either ordinary responses or awaitables.
Different launch orders exercise readiness rather than source-order execution.
"""

import asyncio
import inspect
import json
import operator
import random
from copy import deepcopy

from lev import ir
from lev.plan import ExecutionPlan


_SKIP = object()
_COMPARE = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
    "gt": operator.gt, "ge": operator.ge, "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b, "is": operator.is_, "is_not": operator.is_not,
}


def _expr(e, env, state):
    kind = e["kind"]
    ev = lambda x: _expr(x, env, state)
    if kind == "literal":
        return e["value"]
    if kind == "var":
        return env[e["name"]]
    if kind == "state":
        return state
    if kind == "get_item":
        return ev(e["value"])[ev(e["index"])]
    if kind == "field":
        return getattr(ev(e["value"]), e["name"])
    if kind in ("list", "tuple"):
        result = [ev(x) for x in e["items"]]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict":
        return {ev(k): ev(v) for k, v in e["entries"]}
    if kind == "unary":
        return {"not": operator.not_, "negative": operator.neg, "positive": operator.pos}[e["op"]](ev(e["value"]))
    if kind == "binary":
        return ev(e["left"]) + ev(e["right"])
    if kind == "compare":
        left = ev(e["left"])
        for op, node in zip(e["ops"], e["comparators"]):
            right = ev(node)
            if not _COMPARE[op](left, right):
                return False
            left = right
        return True
    if kind == "bool_op":
        value = ev(e["values"][0])
        for x in e["values"][1:]:
            if (e["op"] == "and" and not value) or (e["op"] == "or" and value):
                break
            value = ev(x)
        return value
    if kind == "intrinsic":
        funcs = {"len": len, "int": int, "range": range, "enumerate": enumerate,
                 "str.strip": str.strip, "json.dumps": json.dumps,
                 "example": lambda state, answer: deepcopy({"state": state, "answer": answer})}
        return funcs[e["name"]](*[ev(x) for x in e["args"]], **{k: ev(v) for k, v in e["kwargs"]})
    if kind == "format_value":
        value = ev(e["value"])
        spec = ev(e["spec"]) if e["spec"] is not None else ""
        if e["conversion"] != "none":
            value = {"s": str, "r": repr, "a": ascii}[e["conversion"]](value)
        return format(value, spec)
    if kind == "format_string":
        return "".join(ev(x) for x in e["parts"])
    raise AssertionError(kind)


def _unpack(target, value):
    if target["kind"] == "name_target":
        return [value]
    values = list(value)
    if len(values) != len(target["items"]):
        raise ValueError("Unpack length mismatch")
    result = []
    for t, v in zip(target["items"], values):
        result.extend(_unpack(t, v))
    return result


class _Executor:
    def __init__(self, plan, lev, mode, order):
        decoded = ExecutionPlan.from_dict(plan)
        self.plan = decoded.to_dict()
        self.dependencies = [ir._encode(d) for d in decoded.dependencies_for(mode)]
        self.mode, self.order, self.lev = mode, order, lev
        self.tasks = {t["id"]: t for t in self.plan["tasks"]}
        self.regions = {r["id"]: r for r in self.plan["regions"]}
        self.values = {v["id"]: v for v in self.plan["values"]}
        self.requests = {r["id"]: r for r in self.plan["requests"]}
        self.futures, self.outputs, self.region_runs = {}, {}, {}
        self.iterations, self.items, self.entered = {}, {}, {}
        self.cache, self.returns = {}, []
        self.failed = None
        self.random = random.Random(7)
        self.paths = {}
        for t in self.tasks.values():
            path, region = [], self.regions[t["region"]]
            while region["parent"] is not None:
                if region["kind"] == "loop":
                    path.append(region["owner"])
                region = self.regions[region["parent"]]
            self.paths[t["id"]] = tuple(reversed(path))
        self.incoming = {t: [] for t in self.tasks}
        for d in self.dependencies:
            self.incoming[d["target"]].append(d)

    def launch(self, region, ctx):
        ids = list(self.regions[region]["tasks"])
        if self.order == "reverse":
            ids.reverse()
        elif self.order == "random":
            self.random.shuffle(ids)
        keys = []
        for tid in ids:
            key = (tid, ctx)
            self.entered[key] = asyncio.Event()
            self.futures[key] = asyncio.create_task(self.run_task(tid, ctx))
            keys.append(key)
        self.region_runs[region, ctx] = keys

    async def ensure(self, tid, ctx):
        ctx = ctx[:len(self.paths[tid])]
        key = tid, ctx
        if key not in self.futures:
            region = self.regions[self.tasks[tid]["region"]]
            owner = region["owner"]
            if region["kind"] in ("then", "else"):
                selected = await self.ensure(owner, ctx)
                if selected is _SKIP or bool(selected) != (region["kind"] == "then"):
                    return _SKIP
            else:
                raise AssertionError(f"uninstantiated task {key}")
        return await self.futures[key]

    async def port(self, tid, ctx, port):
        if port == "enter":
            key = tid, ctx[:len(self.paths[tid])]
            await self.entered[key].wait()
            return True
        result = await self.ensure(tid, ctx)
        if result is _SKIP:
            return False
        if port == "false":
            return not result
        if port in ("true", "continue"):
            return bool(result)
        return True

    async def dependency(self, d, ctx):
        if d["condition"]:
            c = d["condition"]
            if not await self.port(c["task"], ctx, c["port"]):
                return
        m = d["mapping"]
        source, target = m["source_loops"], m["target_loops"]
        count = 0
        for a, b in zip(source, target):
            if a != b:
                break
            count += 1
        if m["carried_loop"] is not None:
            count = min(count, target.index(m["carried_loop"]) + 1)
        prefix = list(ctx[:count])
        if m["carried_loop"] is not None:
            i = target.index(m["carried_loop"])
            prefix[i] -= m["distance"]
            if prefix[i] < 0:
                return
        contexts = [tuple(prefix)]
        for loop in source[count:]:
            expanded = []
            for parent in contexts:
                result = await self.ensure(loop, parent)
                if result is not _SKIP:
                    expanded.extend(parent + (i,) for i in self.iterations[loop, parent])
            contexts = expanded
        for context in contexts:
            await self.port(d["source"], context, d["port"])

    async def value(self, value, ctx):
        producer = self.values[value]["producer"]
        ctx = ctx[:len(self.paths[producer])]
        result = await self.ensure(producer, ctx)
        if result is _SKIP:
            raise AssertionError(f"read of inactive value {value}")
        return self.outputs[value, ctx]

    async def expression(self, bound, ctx):
        env = {b["name"]: await self.value(b["value"], ctx) for b in bound["bindings"]}
        return _expr(bound["expression"], env, self.lev.state)

    async def complete_region(self, region, ctx):
        results = await asyncio.gather(*(self.futures[k] for k in self.region_runs[region, ctx]))
        return any(self.tasks[k[0]]["kind"] == "return" and result is not _SKIP or
                   self.tasks[k[0]]["kind"] == "join" and result is False
                   for k, result in zip(self.region_runs[region, ctx], results))

    async def invoke(self, operation, args, kwargs):
        if self.failed is not None:
            raise self.failed
        try:
            result = getattr(self.lev, operation)(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        except Exception as exc:
            if self.failed is None:
                self.failed = exc
            raise

    async def run_task(self, tid, ctx):
        try:
            return await self.task_body(tid, ctx)
        except Exception as exc:
            if self.failed is None:
                self.failed = exc
            raise

    async def task_body(self, tid, ctx):
        t = self.tasks[tid]
        for gate in t["gates"]:
            if not await self.port(gate["task"], ctx, gate["port"]):
                return _SKIP
        for d in self.incoming[tid]:
            await self.dependency(d, ctx)
        if self.failed:
            raise self.failed
        kind, values, result = t["kind"], [], None
        if kind == "eval":
            values = [await self.expression(t["expression"], ctx)]
        elif kind == "unpack":
            values = _unpack(t["target"], await self.expression(t["expression"], ctx))
        elif kind == "store":
            obj = await self.expression(t["container"], ctx)
            index = await self.expression(t["index"], ctx)
            item = await self.expression(t["item"], ctx)
            obj[index] = item
        elif kind == "infer_site":
            args, kwvalues = await self.value(t["arguments"], ctx)
            request = self.requests[t["request"]]
            key = request["id"], ctx[:len(request["scope"])]
            if not request["reusable"]:
                key += (tid, ctx)
            if key not in self.cache:
                self.cache[key] = asyncio.create_task(self.invoke(request["operation"], args, dict(zip(t["keywords"], kwvalues))))
            values = [await self.cache[key]]
        elif kind == "branch":
            result = bool(await self.expression(t["condition"], ctx))
            self.launch(t["then_region"] if result else t["else_region"], ctx)
        elif kind == "join":
            branch = self.tasks[t["branch"]]
            selected = await self.ensure(t["branch"], ctx)
            result = not await self.complete_region(branch["then_region"] if selected else branch["else_region"], ctx)
        elif kind == "merge":
            selected = await self.ensure(t["branch"], ctx)
            value = t["then_value"] if selected else t["else_value"]
            if value is None:
                return _SKIP
            values = [await self.value(value, ctx)]
        elif kind == "carry":
            values = [await self.value(t["initial"], ctx[:-1]) if ctx[-1] == 0 else
                      await self.value(t["update"], ctx[:-1] + (ctx[-1]-1,))]
        elif kind == "iteration":
            values = _unpack(t["target"], self.items[t["loop"], ctx])
        elif kind == "loop":
            iterable = await self.expression(t["iterable"], ctx)
            iterator = iter(iterable)
            self.iterations[tid, ctx] = []
            self.entered[tid, ctx].set()
            index = 0
            while True:
                if index and self.mode in t["serial_modes"]:
                    await self.complete_region(t["body"], ctx + (index-1,))
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                self.items[tid, ctx + (index,)] = item
                self.iterations[tid, ctx].append(index)
                self.launch(t["body"], ctx + (index,))
                index += 1
            for i in range(index):
                await self.complete_region(t["body"], ctx + (i,))
            values = [await self.value(e["update"], ctx + (index-1,)) if index else
                      await self.value(e["initial"], ctx) for e in t["exports"]]
        elif kind == "check":
            valid = await self.expression(t["condition"], ctx)
            message = await self.expression(t["message"], ctx) if t["message"] is not None else None
            if not valid:
                raise ValueError(message or "Constraint failed")
        elif kind == "return":
            result = await self.expression(t["expression"], ctx)
            self.returns.append(result)
        else:
            raise AssertionError(kind)
        for output, value in zip(t["outputs"], values):
            self.outputs[output, ctx] = value
        return result

    async def run(self):
        root = next(r["id"] for r in self.regions.values() if r["kind"] == "root")
        self.launch(root, ())
        try:
            await self.complete_region(root, ())
            assert len(self.returns) == 1, self.returns
            return self.returns[0]
        except Exception as exc:
            # A nested check may fail before another root task propagates its
            # own exception. Report observation order, not region-join order.
            raise self.failed if self.failed is not None else exc
        finally:
            pending = set(self.futures.values()) | set(self.cache.values())
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


def evaluate(plan, lev, *, mode="eager", order="forward"):
    """Run a serialized Plan against a deterministic fake provider."""
    return asyncio.run(asyncio.wait_for(_Executor(plan, lev, mode, order).run(), timeout=5))
