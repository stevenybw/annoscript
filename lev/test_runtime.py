"""Independent runtime semantics, bounded scheduling and per-state batching."""

import asyncio
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from lev._data import thaw
from lev.backends import BatchResult, InferenceOutput, BackendFatalError
from lev.compiler import compile_program
from lev.plan import ExecutionPlan
from lev import ir, plan as p
from lev.plan_lowering import lower_to_plan
from lev.runtime import LevRuntime, RunInput
from lev.lev import NoulResponse, ChoiceResponse, ScoreResponse, GenerateResponse, constrain
from lev import test_plan as fixtures
from lev import test_plan_constraints as checks
from lev._test_plan_runtime import evaluate


def mixed(lev):
    a = lev.noul("yes?").noul
    b = lev.choice("which?", {"a": "A", "b": "B"}).choice
    c = lev.score("score?", ["bad", "good"]).score
    return [a, b, c]


def decision_chain(lev):
    a = lev.choice("first", {"a": "A", "b": "B"}).choice
    b = lev.noul("independent").noul
    return lev.noul(a).noul + b


def decision_loop(lev):
    result = []
    for item in lev.state:
        result = result + [lev.noul(item).noul]
    return result


def changed_criteria(lev):
    criteria = {"a": "before", "b": "other"}
    alias = criteria
    first = lev.choice("choose", criteria).choice
    alias["a"] = "after"
    second = lev.choice("choose", criteria).choice
    return [first, second]


class FakeBackend:
    def __init__(self, table=None, fail=()):
        self.table, self.fail = table, fail
        self.calls, self.batches = [], []
        self.closed = False

    def answer(self, call):
        a = thaw(call.arguments)
        self.calls.append(fixtures._key(call.operation, (), a))
        if self.table is not None:
            result = self.table[self.calls[-1]]
            if isinstance(result, Exception):
                raise type(result)(str(result))
            return deepcopy(result)
        if a["instructions"] in self.fail:
            raise ValueError(a["instructions"])
        if call.operation == "generate":
            return GenerateResponse(a["instructions"])
        if call.operation == "noul":
            return NoulResponse(0.8)
        if call.operation == "choice":
            return ChoiceResponse(next(iter(a["criteria"])), 0.9, {k: float(i == 0) for i, k in enumerate(a["criteria"])})
        if call.operation == "score":
            return ScoreResponse({i: float(i == 0) for i in range(len(a["criteria"]))}, 0, dict(enumerate(a["criteria"])), 0.9)
        raise AssertionError(call.operation)

    async def infer(self, context, call):
        return InferenceOutput(self.answer(call))

    async def infer_batch(self, context, calls):
        self.batches.append((thaw(context.state), [c.operation for c in calls], [c.arguments["instructions"] for c in calls]))
        return BatchResult({c.id: self.answer(c) for c in calls}, {"usage": {"input_tokens": 100}})

    async def aclose(self):
        self.closed = True


async def execute(function, state, *, backend=None, mode="eager", **options):
    backend = backend or FakeBackend()
    async with LevRuntime(backend, backend, trace=True, **options) as runtime:
        plan = ExecutionPlan.from_json(lower_to_plan(compile_program(function)).to_json())
        before = plan.to_dict()
        stream = await runtime.submit(plan, [RunInput(0, state)], mode=mode)
        result = [r async for r in stream]
        assert before == plan.to_dict()
        return result[0], backend, runtime.trace


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_plan_shapes_against_independent_interpreter(self):
        cases = [
            (fixtures.independent, "state"), (fixtures.duplicates, "state"),
            (fixtures.branch_duplicates, []), (fixtures.branch_duplicates, [1]),
            (fixtures.unused_branch, []), (fixtures.loop_invariant, [True, False, True]),
            (fixtures.independent_loop, [str(i) for i in range(40)]),
            (fixtures.last_value, []), (fixtures.last_value, ["a", "b"]),
            (fixtures.conditional_last, ["a", "", "b"]),
            (fixtures.nested_loop, [["a", "b"], [], ["c"]]),
            (fixtures.local_container, ["a", "b"]), (fixtures.memory_snapshots, "x"),
            (fixtures.iteration_alias, ["a", "b", "c"]),
            (fixtures.nested_memory, [["a", "b"], [], ["c"]]),
            (fixtures.iterator_twice, ["a", "b"]),
            (fixtures.early_return, []), (fixtures.early_return, [True]),
            (fixtures.nested_return, [False]), (fixtures.nested_return, [True]),
            (fixtures.partial_nested_return, []), (fixtures.partial_nested_return, [False]),
            (checks.checked_memory, "x"), (checks.checked_rebinding, "x"),
            (checks.checked_shared, "same"), (checks.checked_memory_loop, ["a", "b"]),
        ]
        for fn, state in cases:
            for mode in ("eager", "strict"):
                with self.subTest(function=fn.__name__, mode=mode, state=state):
                    plan = lower_to_plan(compile_program(fn))
                    reference = fixtures.PureLev(deepcopy(state))
                    expected = await asyncio.to_thread(evaluate, plan.to_dict(), reference, mode=mode)
                    result, backend, _ = await execute(fn, state, mode=mode, loop_window=2)
                    self.assertEqual(result.status, "success", result.error)
                    self.assertEqual(result.value, expected)
                    expected_calls = [fixtures._key("generate", (x,), opts) for x, opts in reference.calls]
                    self.assertEqual(Counter(backend.calls), Counter(expected_calls))

    async def test_mixed_batch_and_usage_once(self):
        result, backend, trace = await execute(mixed, "shared")
        self.assertEqual(result.value, [0.8, "a", 0])
        self.assertEqual(backend.batches[0][1], ["noul", "choice", "score"])
        self.assertEqual(len(backend.batches), 1)
        self.assertEqual(sum(e.get("usage", {}).get("input_tokens", 0) for e in trace), 100)
        _, backend, _ = await execute(mixed, "shared", mode="strict")
        self.assertEqual([len(b[1]) for b in backend.batches], [1, 1, 1])

    async def test_dependency_forms_later_batch_and_loop_capacity(self):
        result, backend, _ = await execute(decision_chain, "x")
        self.assertEqual(result.value, 1.6)
        self.assertEqual([b[2] for b in backend.batches], [["first", "independent"], ["a"]])
        result, backend, _ = await execute(decision_loop, [str(i) for i in range(9)], max_batch_questions=3, loop_window=4)
        self.assertEqual(result.value, [0.8] * 9)
        self.assertTrue(all(len(b[1]) <= 3 for b in backend.batches))
        self.assertGreater(max(len(b[1]) for b in backend.batches), 1)

    async def test_mutated_criteria_preserves_call_snapshots(self):
        for mode in ("eager", "strict"):
            result, backend, _ = await execute(changed_criteria, "x", mode=mode)
            self.assertEqual(result.value, ["a", "a"])
            self.assertEqual(backend.calls, [fixtures._key("choice", ("choose", {"a": value, "b": "other"}), {}) for value in ("before", "after")])

    async def test_inputs_never_share_batches_and_are_snapshotted(self):
        backend = FakeBackend()
        async with LevRuntime(backend, backend) as runtime:
            source = {"text": "original"}
            stream = await runtime.submit(lower_to_plan(compile_program(mixed)), [RunInput(3, source), RunInput(9, source)])
            source["text"] = "changed"
            result = [r async for r in stream]
            self.assertEqual({r.index for r in result}, {3, 9})
            self.assertEqual(len(backend.batches), 2)
            self.assertTrue(all(b[0] == {"text": "original"} for b in backend.batches))

    async def test_failure_prefix_and_constraint_paths(self):
        for fn, state, fail, prefix in [
            (fixtures.independent, "x", ("a",), ["a"]),
            (fixtures.expression_failure, "x", (), ["first"]),
            (fixtures.unpack_failure, [1], (), []),
            (fixtures.unused_failure, "x", ("bad",), ["bad"]),
            (fixtures.independent_loop, ["bad", "later"], ("bad",), ["bad"]),
            (checks.checked_chain, "x", (), ["check"]),
            (checks.checked_container_truth, "x", (), []),
        ]:
            with self.subTest(function=fn.__name__):
                result, backend, _ = await execute(fn, state, backend=FakeBackend(fail=fail), mode="strict")
                self.assertEqual(result.status, "error")
                expected = [fixtures._key("generate", (x,), {}) for x in prefix]
                self.assertEqual(backend.calls, expected)
                self.assertIn("task=", result.error.message)

    async def test_eager_failure_cancels_pending_and_no_new_dispatch(self):
        class Controlled(FakeBackend):
            async def infer(self, context, call):
                name = call.arguments["instructions"]
                self.started.append(name)
                if name == "check":
                    await self.other_started.wait()
                    return InferenceOutput(GenerateResponse("bad"))
                if name == "independent":
                    self.other_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.cancelled.set()
                self.fail_unexpected = True
        backend = Controlled()
        backend.started, backend.other_started, backend.cancelled = [], asyncio.Event(), asyncio.Event()
        result, _, _ = await execute(checks.checked_chain, "x", backend=backend)
        self.assertEqual(result.error.code, "constraint_unsatisfied")
        self.assertCountEqual(backend.started, ["check", "independent"])
        self.assertTrue(backend.cancelled.is_set())

    async def test_concurrency_limit_close_and_submission_isolation(self):
        class Controlled(FakeBackend):
            async def infer(self, context, call):
                self.active += 1
                self.peak = max(self.active, self.peak)
                self.started.set()
                try:
                    await self.release.wait()
                    return InferenceOutput(GenerateResponse(call.arguments["instructions"]))
                finally:
                    self.active -= 1
        backend = Controlled()
        backend.active = backend.peak = 0
        backend.started, backend.release = asyncio.Event(), asyncio.Event()
        async with LevRuntime(backend, backend, max_llm_inflight=1) as runtime:
            plan = lower_to_plan(compile_program(fixtures.independent))
            first = await runtime.submit(plan, [RunInput(0, "x")])
            second = await runtime.submit(plan, [RunInput(1, "y")])
            await asyncio.wait_for(backend.started.wait(), 2)
            await first.aclose()
            backend.release.set()
            result = [r async for r in second]
            self.assertEqual(result[0].value, ["a", "b"])
            self.assertEqual(backend.peak, 1)
            self.assertFalse(backend.closed)

    async def test_active_run_bound_and_local_work_quantum(self):
        result, backend, trace = await execute(decision_loop, [str(i) for i in range(20)], local_task_budget=3, loop_window=2)
        self.assertEqual(result.value, [0.8] * 20)
        self.assertEqual(sum(len(b[1]) for b in backend.batches), 20)

    async def test_success_waits_for_unused_calls_and_checks(self):
        for function, state in ((checks.checked_constant_return, "x"), (checks.checked_nested_return, [True]),
                                (checks.checked_unused_loop, ["a", "b"])):
            class Delayed(FakeBackend):
                async def infer(self, context, call):
                    self.started.set()
                    await self.release.wait()
                    return InferenceOutput(GenerateResponse("bad"))
            backend = Delayed()
            backend.started, backend.release = asyncio.Event(), asyncio.Event()
            async with LevRuntime(backend, backend) as runtime:
                stream = await runtime.submit(lower_to_plan(compile_program(function)), [RunInput(0, state)])
                next_result = asyncio.create_task(stream.__anext__())
                await asyncio.wait_for(backend.started.wait(), 2)
                self.assertFalse(next_result.done())
                backend.release.set()
                result = await asyncio.wait_for(next_result, 2)
                self.assertEqual(result.error.code, "constraint_unsatisfied")

    async def test_first_observed_error_beats_source_order(self):
        class Failing(FakeBackend):
            async def infer(self, context, call):
                if call.arguments["instructions"] == "a":
                    self.first_started.set()
                    await asyncio.Event().wait()
                await self.first_started.wait()
                raise ValueError("b observed first")
        backend = Failing()
        backend.first_started = asyncio.Event()
        result, _, _ = await execute(fixtures.independent, "x", backend=backend)
        self.assertIn("b observed first", result.error.message)

    async def test_active_runs_and_loop_window_before_completion(self):
        class Delayed(FakeBackend):
            async def infer(self, context, call):
                self.started.set()
                await self.release.wait()
                return InferenceOutput(GenerateResponse(call.arguments["instructions"]))
        backend = Delayed()
        backend.started, backend.release = asyncio.Event(), asyncio.Event()
        async with LevRuntime(backend, backend, max_active_runs=1, loop_window=2, trace=True) as runtime:
            stream = await runtime.submit(lower_to_plan(compile_program(fixtures.independent_loop)),
                                          [RunInput(i, ["a", "b", "c", "d"]) for i in range(3)])
            await asyncio.wait_for(backend.started.wait(), 2)
            self.assertEqual(sum(e["event"] == "run_started" for e in runtime.trace), 1)
            self.assertEqual(sum(e["event"] == "infer_ready" for e in runtime.trace), 2)
            backend.release.set()
            self.assertEqual(len([r async for r in stream]), 3)

    async def test_deserialized_plan_cannot_write_input_state(self):
        plan = lower_to_plan(compile_program(fixtures.memory_snapshots))
        plan = replace(plan, tasks=tuple(replace(t, container=p.BoundExpr(ir.State())) if isinstance(t, p.Store) else t for t in plan.tasks))
        plan = ExecutionPlan.from_json(plan.to_json())
        state = {"x": "input"}
        backend = FakeBackend()
        async with LevRuntime(backend, backend) as runtime:
            stream = await runtime.submit(plan, [RunInput(0, state)], mode="strict")
            result = [r async for r in stream]
            self.assertEqual(result[0].status, "error")
            self.assertIn("locally constructed", result[0].error.message)
            self.assertEqual(state, {"x": "input"})

    async def test_invalid_plan_and_empty_submission(self):
        backend = FakeBackend()
        async with LevRuntime(backend, backend) as runtime:
            plan = lower_to_plan(compile_program(mixed))
            stream = await runtime.submit(plan, [])
            self.assertEqual([r async for r in stream], [])
            with self.assertRaises(ValueError):
                await runtime.submit(plan, [RunInput(0, "x"), RunInput(0, "y")])
            with self.assertRaises(ValueError):
                await runtime.submit(plan, [RunInput(0, "x")], mode="invalid")
            self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
