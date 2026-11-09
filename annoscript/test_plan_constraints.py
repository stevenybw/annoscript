"""Eager check speculation without weakening bindings, paths or memory order."""

import asyncio
from dataclasses import replace
import unittest

from annoscript import plan as p
from annoscript.compiler import compile_program
from annoscript.anno import GenerateResponse, constrain
from annoscript.plan_lowering import lower_to_plan
from annoscript._test_ir_runtime import evaluate as evaluate_ir
from annoscript._test_plan_runtime import _Executor, evaluate
from annoscript.test_plan import PureAnnoScript, checked_loop


def checked_chain(anno):
    text = anno.generate("check", max_length=512).text
    constrain(text == "ok", message="check failed")
    other = anno.generate("independent", max_length=512).text
    return anno.generate(other, max_length=512).text


def checked_constant_return(anno):
    text = anno.generate("check", max_length=512).text
    constrain(text == "ok", message="check failed")
    return "done"


def checked_nested_return(anno):
    if anno.state:
        if anno.state[0]:
            text = anno.generate("check", max_length=512).text
            constrain(text == "ok", message="check failed")
    return "done"


def checked_unused_loop(anno):
    for item in anno.state:
        text = anno.generate(item, max_length=512).text
        constrain(text == "ok", message="check failed")
    return "done"


def checked_data(anno):
    text = anno.generate("first", max_length=512).text
    constrain(text != "")
    return anno.generate(text + " second", max_length=512).text


def checked_control(anno):
    if anno.generate("predicate", max_length=512).text == "selected":
        text = anno.generate("selected", max_length=512).text
        constrain(text == "ok", message="selected failed")
    return anno.generate("independent", max_length=512).text


def checked_memory(anno):
    data = {"value": "initial"}
    alias = data
    alias["value"] = anno.generate("updated", max_length=512).text
    constrain(data["value"] == "updated", message=data["value"])
    alias["value"] = "later"
    return data["value"]


def checked_container_truth(anno):
    data = {}
    constrain(data, message="empty container")
    data["key"] = "later"
    return data


def checked_message(anno):
    data = {"message": "original diagnostic"}
    constrain(anno.state, message=data["message"])
    data["message"] = "changed diagnostic"
    return data


def checked_rebinding(anno):
    value = anno.generate("original", max_length=512).text
    constrain(value == "original")
    value = "rebound"
    return value


def checked_shared(anno):
    a = anno.generate("same", max_length=512).text
    constrain(a == anno.state, message="shared check failed")
    b = anno.generate("same", max_length=512).text
    return [a, b]


def checked_memory_loop(anno):
    data = {"value": "initial"}
    for item in anno.state:
        data["value"] = item
        text = anno.generate(item, max_length=512).text
        constrain(data["value"] == text, message="wrong iteration")
    return data["value"]


class ConstraintPlanTests(unittest.TestCase):
    def plan(self, function):
        return lower_to_plan(compile_program(function))

    def test_only_explicit_ordering_edges_are_relaxed(self):
        plan = self.plan(checked_memory)
        eager, strict = plan.dependencies_for(), plan.dependencies_for("strict")
        self.assertEqual(strict, plan.dependencies)
        self.assertTrue(any(d.kind == "constraint" and d.strict_only for d in strict))
        for d in strict:
            if d.kind in ("data", "control", "memory"):
                self.assertIn(d, eager)
                self.assertFalse(d.strict_only)
        self.assertFalse(any(d.kind in ("constraint", "failure_order") for d in eager))
        self.assertEqual(p.ExecutionPlan.from_json(plan.to_json()), plan)
        # Old plans remain conservative. The constraint *kind* is not itself
        # permission to remove an edge, and missing flags default to False.
        legacy = plan.to_dict()
        for d in legacy["dependencies"]:
            d.pop("strict_only")
        restored = p.ExecutionPlan.from_dict(legacy)
        self.assertTrue(any(d.kind == "constraint" for d in restored.dependencies_for()))
        for kind in ("data", "control", "memory"):
            original = self.plan(checked_control if kind == "control" else checked_memory)
            edge = next(d for d in original.dependencies if d.kind == kind)
            broken = replace(original, dependencies=tuple(replace(d, strict_only=True) if d == edge else d for d in original.dependencies))
            with self.assertRaisesRegex(p.PlanDecodeError, "cannot be strict-only"):
                p.ExecutionPlan.from_json(broken.to_json())

    def test_eager_constraint_failure_cancels_inflight_and_blocks_new_calls(self):
        class Concurrent(PureAnnoScript):
            async def generate(self, instructions, **kwargs):
                self.calls.append((instructions, kwargs))
                if instructions == "check":
                    await self.independent_started.wait()
                    return GenerateResponse("bad")
                if instructions == "independent":
                    self.independent_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled = True
                        raise
                raise AssertionError("dispatched after failure")

        async def exercise():
            anno = Concurrent()
            anno.independent_started, anno.cancelled = asyncio.Event(), False
            executor = _Executor(self.plan(checked_chain).to_dict(), anno, "eager", "reverse")
            with self.assertRaisesRegex(ValueError, "check failed"):
                await executor.run()
            self.assertCountEqual([x[0] for x in anno.calls], ["check", "independent"])
            self.assertTrue(anno.cancelled)

        asyncio.run(asyncio.wait_for(exercise(), 2))
        strict = PureAnnoScript()
        with self.assertRaisesRegex(ValueError, "check failed"):
            evaluate(self.plan(checked_chain).to_dict(), strict, mode="strict", order="reverse")
        self.assertEqual([x[0] for x in strict.calls], ["check"])

    def test_first_observed_infer_failure_can_precede_pending_check(self):
        class Concurrent(PureAnnoScript):
            async def generate(self, instructions, **kwargs):
                self.calls.append((instructions, kwargs))
                if instructions == "check":
                    self.check_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled = True
                        raise
                await self.check_started.wait()
                raise ValueError("independent failed first")

        async def exercise():
            anno = Concurrent()
            anno.check_started, anno.cancelled = asyncio.Event(), False
            executor = _Executor(self.plan(checked_chain).to_dict(), anno, "eager", "random")
            with self.assertRaisesRegex(ValueError, "independent failed first"):
                await executor.run()
            self.assertTrue(anno.cancelled)
            self.assertCountEqual([x[0] for x in anno.calls], ["check", "independent"])

        asyncio.run(asyncio.wait_for(exercise(), 2))

    def test_success_waits_for_root_nested_and_loop_checks(self):
        class Delayed(PureAnnoScript):
            async def generate(self, instructions, **kwargs):
                self.started.set()
                await self.release.wait()
                return GenerateResponse(self.response)

        async def exercise(function, state, valid):
            anno = Delayed(state)
            anno.started, anno.release = asyncio.Event(), asyncio.Event()
            anno.response = "ok" if valid else "bad"
            executor = _Executor(self.plan(function).to_dict(), anno, "eager", "reverse")
            task = asyncio.create_task(executor.run())
            try:
                await anno.started.wait()
                self.assertEqual(executor.returns, ["done"])  # candidate only
                self.assertFalse(task.done())
                anno.release.set()
                if valid:
                    self.assertEqual(await task, "done")
                else:
                    with self.assertRaisesRegex(ValueError, "check failed"):
                        await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        for function, state in ((checked_constant_return, None), (checked_nested_return, [True]),
                                (checked_unused_loop, ["check", "check"])):
            for valid in (False, True):
                with self.subTest(function=function.__name__, valid=valid):
                    asyncio.run(asyncio.wait_for(exercise(function, state, valid), 2))

    def test_data_and_branch_activation_still_gate_infer_and_check(self):
        for mode in ("eager", "strict"):
            for order in ("forward", "reverse", "random"):
                anno = PureAnnoScript()
                self.assertEqual(evaluate(self.plan(checked_data).to_dict(), anno, mode=mode, order=order), "first second")
                self.assertEqual([x[0] for x in anno.calls], ["first", "first second"])
                anno = PureAnnoScript()
                self.assertEqual(evaluate(self.plan(checked_control).to_dict(), anno, mode=mode, order=order), "independent")
                self.assertCountEqual([x[0] for x in anno.calls], ["predicate", "independent"])
                self.assertEqual(evaluate(self.plan(checked_nested_return).to_dict(), PureAnnoScript([]), mode=mode, order=order), "done")
                self.assertEqual(evaluate(self.plan(checked_unused_loop).to_dict(), PureAnnoScript([]), mode=mode, order=order), "done")

    def test_memory_versions_diagnostics_and_rebinding_match_ir(self):
        for function, state in ((checked_memory, None), (checked_container_truth, None), (checked_message, False),
                                (checked_rebinding, None), (checked_memory_loop, ["one", "two", "three"])):
            program = compile_program(function)
            failure, expected = None, None
            try:
                expected = evaluate_ir(program.to_dict(), PureAnnoScript(state))
            except ValueError as exc:
                failure = str(exc)
            plan = lower_to_plan(program)
            checks = {t.id for t in plan.tasks if isinstance(t, p.Check)}
            if function is not checked_rebinding:
                self.assertTrue(any(d.source in checks and d.kind == "memory" and d.hazard == "WAR"
                                    for d in plan.dependencies_for()))
            for mode in ("eager", "strict"):
                for order in ("forward", "reverse", "random"):
                    with self.subTest(function=function.__name__, mode=mode, order=order):
                        if failure:
                            with self.assertRaises(ValueError) as caught:
                                evaluate(plan.to_dict(), PureAnnoScript(state), mode=mode, order=order)
                            self.assertEqual(str(caught.exception), failure)
                        else:
                            self.assertEqual(evaluate(plan.to_dict(), PureAnnoScript(state), mode=mode, order=order), expected)

    def test_unoptimized_ir_retains_check_bindings_and_memory_reads(self):
        for function, state in ((checked_memory, None), (checked_rebinding, None),
                                (checked_memory_loop, ["one", "two"])):
            program = compile_program(function, optimize=False)
            expected = evaluate_ir(program.to_dict(), PureAnnoScript(state))
            plan = lower_to_plan(program)
            for mode in ("eager", "strict"):
                self.assertEqual(evaluate(plan.to_dict(), PureAnnoScript(state), mode=mode, order="reverse"), expected)

    def test_shared_results_do_not_skip_checks(self):
        plan = self.plan(checked_shared)
        self.assertEqual(len(plan.requests), 1)
        for mode in ("eager", "strict"):
            for state in ("same", "wrong"):
                anno = PureAnnoScript(state)
                if state == "same":
                    self.assertEqual(evaluate(plan.to_dict(), anno, mode=mode, order="reverse"), ["same", "same"])
                else:
                    with self.assertRaisesRegex(ValueError, "shared check failed"):
                        evaluate(plan.to_dict(), anno, mode=mode, order="reverse")
                self.assertEqual(len(anno.calls), 1)

    def test_checked_loop_allows_eager_infer_overlap_only(self):
        class Concurrent(PureAnnoScript):
            def __init__(self, state):
                super().__init__(state)
                self.active = self.peak = 0

            async def generate(self, instructions, **kwargs):
                self.calls.append((instructions, kwargs))
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.01)
                    return GenerateResponse(instructions)
                finally:
                    self.active -= 1

        plan = self.plan(checked_loop)
        for mode, peak in (("eager", 3), ("strict", 1)):
            anno = Concurrent(["one", "two", "three"])
            self.assertEqual(evaluate(plan.to_dict(), anno, mode=mode), anno.state)
            self.assertEqual(anno.peak, peak)
            self.assertEqual(anno.active, 0)
        self.assertEqual(evaluate(plan.to_dict(), PureAnnoScript([])), [])


if __name__ == "__main__":
    unittest.main()
