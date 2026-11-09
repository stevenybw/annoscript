"""Plan structure, scheduling, sharing, failure and business differential tests."""

import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from annoscript import ir, plan as p
from annoscript._test_artifacts import assert_saved_artifact
from annoscript.compiler import compile_program
from annoscript.plan_lowering import lower_to_plan, PlanLoweringError
from annoscript._test_plan_runtime import evaluate
from annoscript._test_ir_runtime import evaluate as evaluate_ir
from annoscript.anno import Anno, GenerateResponse, constrain, Example
from annoscript import test_compiler as compiler_tests


def _key(operation, args, kwargs):
    def freeze(x):
        if type(x) is Example:
            return freeze(x.to_dict())
        if isinstance(x, dict):
            return ("dict", tuple((freeze(k), freeze(v)) for k, v in x.items()))
        if isinstance(x, (tuple, list)):
            return (type(x).__name__, tuple(freeze(v) for v in x))
        return (type(x).__name__, x.hex() if type(x) is float else x)
    defaults = {"generate":{"max_length":512},"generate_list":{"max_items":8,"max_length_per_item":58},"extract":{"max_items":8,"max_length_per_item":58}}.get(operation,{})
    bound = inspect.signature(getattr(Anno, operation)).bind(None, *args, **{**defaults,**kwargs})
    bound.apply_defaults()
    return operation, freeze(tuple((k, v) for k, v in bound.arguments.items() if k != "self"))


class PureAnnoScript:
    def __init__(self, state=None, fail=()):
        self.state = state
        self.calls = []
        self.fail = fail

    def generate(self, instructions, **kwargs):
        self.calls.append((instructions, deepcopy(kwargs)))
        if instructions in self.fail:
            raise ValueError(instructions)
        return GenerateResponse(instructions)


def independent(anno):
    a = anno.generate("a", max_length=512).text
    b = anno.generate("b", max_length=512).text
    return [a, b]


def duplicates(anno):
    a = anno.generate("中文", max_length=512).text
    b = anno.generate(instructions="中文", min_length=0, max_length=512).text
    return [a, b, None]


def branch_duplicates(anno):
    if anno.state:
        a = anno.generate("same", max_length=512).text
    else:
        a = anno.generate("same", max_length=512).text
    b = anno.generate("same", max_length=512).text
    return [a, b]


def unused_branch(anno):
    if anno.state:
        anno.generate("same", max_length=512)
    return None


def loop_invariant(anno):
    result = []
    for item in anno.state:
        if item:
            text = anno.generate("same", max_length=512).text
            result = result + [text]
    return result


def independent_loop(anno):
    result = []
    for item in anno.state:
        result = result + [anno.generate(item, max_length=512).text]
    return result


def last_value(anno):
    result = "initial"
    for item in anno.state:
        result = anno.generate(item, max_length=512).text
    return result


def conditional_last(anno):
    result = "initial"
    for item in anno.state:
        if item:
            result = anno.generate(item, max_length=512).text
    return result


def nested_loop(anno):
    result = []
    for row in anno.state:
        for item in row:
            result = result + [anno.generate(item, max_length=512).text]
    return result


def local_container(anno):
    result = []
    for item in anno.state:
        data = {"x": item}
        data["x"] = item + "!"
        result = result + [anno.generate(data["x"], max_length=512).text]
    return result


def memory_snapshots(anno):
    data = {"x": "before"}
    alias = data
    a = json.dumps(data)
    first = anno.generate(a, max_length=512).text
    alias["x"] = "after"
    b = json.dumps(data)
    second = anno.generate(b, max_length=512).text
    return [first, second]


def iteration_alias(anno):
    data = {"x": "before"}
    result = []
    for item in anno.state:
        result = result + [anno.generate(data["x"], max_length=512).text]
        data["x"] = item
    return result


def nested_memory(anno):
    data = {"x": "initial"}
    result = []
    for row in anno.state:
        for item in row:
            result = result + [anno.generate(data["x"], max_length=512).text]
            data["x"] = item
    return result


def iterator_twice(anno):
    values = enumerate(anno.state)
    first = []
    for index, value in values:
        first = first + [value]
    second = []
    for index, value in values:
        second = second + [value]
    return [first, second]


def early_return(anno):
    if anno.state:
        return anno.generate("first", max_length=512).text
    return anno.generate("second", max_length=512).text


def nested_return(anno):
    if anno.state:
        if anno.state[0]:
            return anno.generate("a", max_length=512).text
        return anno.generate("b", max_length=512).text
    return anno.generate("c", max_length=512).text


def partial_nested_return(anno):
    if anno.state:
        if anno.state[0]:
            return anno.generate("early", max_length=512).text
        anno.generate("inner", max_length=512)
    return anno.generate("late", max_length=512).text


def shared_list(anno):
    first = ["before"]
    second = first
    snapshot = first[0]
    second[0] = "after"
    return [snapshot, first, second]


def nested_store(anno):
    data = {"inner": ["before"]}
    alias = data["inner"]
    first = alias[0]
    alias[0] = "after"
    return [first, data]


def iterator_unpack(anno):
    iterator = enumerate(anno.state)
    a, b = iterator
    rest = []
    for index, item in iterator:
        rest = rest + [item]
    return [a, b, rest]


def checked(anno):
    a = anno.generate("before", max_length=512).text
    constrain(a == anno.state, message="验证失败")
    b = anno.generate("after", max_length=512).text
    return [a, b]


def checked_branch(anno):
    if anno.state:
        constrain(False, message="branch failed")
    return anno.generate("after", max_length=512).text


def checked_loop(anno):
    result = []
    for item in anno.state:
        a = anno.generate(item, max_length=512).text
        constrain(a != "bad")
        result = result + [a]
    return result


def unused_failure(anno):
    anno.generate("bad", max_length=512)
    return "done"


def expression_failure(anno):
    a = anno.generate("first", max_length=512).text
    bad = int("bad")
    b = anno.generate("second", max_length=512).text
    return [a, b]


def unpack_failure(anno):
    first, second = anno.state
    return anno.generate("after", max_length=512).text


def short_circuit(anno):
    a = anno.state and anno.generate("yes", max_length=512).text
    b = True or anno.generate("never", max_length=512).text
    return [a, b]


class PlanTests(unittest.TestCase):
    def compile(self, function, raw=False):
        return lower_to_plan(compile_program(function, optimize=not raw))

    def compare(self, function, states, *, expected_calls=None):
        for raw in (True, False):
            plan = self.compile(function, raw)
            for state in states:
                reference = PureAnnoScript(deepcopy(state))
                expected = function(reference)
                for mode in ("eager", "strict"):
                    for order in ("forward", "reverse", "random"):
                        with self.subTest(function=function.__name__, state=state, raw=raw, mode=mode, order=order):
                            backend = PureAnnoScript(deepcopy(state))
                            self.assertEqual(evaluate(plan.to_dict(), backend, mode=mode, order=order), expected)
                            if expected_calls is not None:
                                self.assertEqual(len(backend.calls), expected_calls(state))

    def test_serialization_and_input_unchanged(self):
        raw = compile_program(duplicates)
        before = raw.to_json()
        plan = lower_to_plan(raw)
        self.assertEqual(raw.to_json(), before)
        self.assertEqual(plan.to_json(), lower_to_plan(raw).to_json())
        self.assertEqual(json.loads(plan.to_json()), plan.to_dict())
        self.assertEqual(p.ExecutionPlan.from_json(plan.to_json()), plan)
        self.assertIn("中文", plan.to_json())
        detached = plan.to_dict()
        detached["tasks"].clear()
        self.assertTrue(plan.tasks)
        self.assertGreater(len(plan.dependencies_for("strict")), len(plan.dependencies_for()))
        with self.assertRaises(ValueError):
            plan.dependencies_for("unknown")
        from annoscript.passes._analysis import walk_statements
        sources = {s.id: s.source for s in walk_statements(raw.body)}
        self.assertTrue(all(t.source == sources[t.origin] for t in plan.tasks))

    def test_duplicate_and_default_argument_sharing(self):
        plan = self.compile(duplicates)
        self.assertEqual(len(plan.requests), 1)
        self.compare(duplicates, [None], expected_calls=lambda _: 1)

    def test_branch_sharing_and_no_speculation(self):
        self.assertEqual(len(self.compile(branch_duplicates).requests), 1)
        self.compare(branch_duplicates, [True, False], expected_calls=lambda _: 1)
        self.compare(unused_branch, [True, False], expected_calls=lambda s: int(s))
        self.compare(short_circuit, [True, False], expected_calls=lambda s: int(s))

    def test_loop_invariant_demand_sharing(self):
        self.assertEqual(self.compile(loop_invariant).requests[0].scope, ())
        self.compare(loop_invariant, [[], [False, False], [True, False, True]], expected_calls=lambda s: int(any(s)))

    def test_loop_carried_data_and_false_dependencies(self):
        plan = self.compile(independent_loop)
        carries = [t for t in plan.tasks if isinstance(t, p.Carry)]
        self.assertEqual(len(carries), 1)
        self.assertTrue(any(d.mapping.distance == 1 and d.kind == "data" for d in plan.dependencies))
        sites = {t.id for t in plan.tasks if isinstance(t, p.Infer)}
        self.assertFalse(any(d.mapping.distance and d.target in sites for d in plan.dependencies))
        self.compare(independent_loop, [[], ["a", "b", "a"]])
        self.compare(last_value, [[], ["a", "b"]])
        self.assertFalse(any(isinstance(t, p.Carry) for t in self.compile(last_value).tasks))
        self.compare(conditional_last, [[], ["a", "", "b", ""]])

    def test_nested_loops(self):
        self.compare(nested_loop, [[], [[], []], [["a", "b"], [], ["c"]]])
        self.compare(nested_memory, [[], [["a", "b"], [], ["c"], ["d", "e", "f"]]])

    def test_mutation_and_iterators(self):
        self.compare(memory_snapshots, [None])
        self.compare(iteration_alias, [[], ["a", "b", "c"]])
        self.compare(local_container, [[], ["a", "b"]])
        self.compare(iterator_twice, [[], ["a", "b"]])
        self.compare(iterator_unpack, [["a", "b"]])
        self.compare(shared_list, [None])
        plan = self.compile(local_container)
        self.assertFalse(any(d.kind == "memory" and d.mapping.distance for d in plan.dependencies))
        plan = self.compile(iteration_alias)
        self.assertTrue(any(d.kind == "memory" and d.mapping.distance for d in plan.dependencies))

    def test_nested_container_ir(self):
        # Subscript-derived writable aliases are rejected by the current source
        # compiler; Plan lowering also accepts independently constructed IR.
        program = compile_program(shared_list)
        source = program.body[0].source
        L, V = ir.Literal, ir.Var
        body = (
            ir.Assign("data", ir.DictExpr(((L("inner"), ir.ListExpr((L("before"),))),)), id="a", source=source),
            ir.Assign("alias", ir.GetItem(V("data"), L("inner")), id="b", source=source),
            ir.Assign("first", ir.GetItem(V("alias"), L(0)), id="c", source=source),
            ir.SetItem(V("alias"), L(0), L("after"), id="d", source=source),
            ir.Return(ir.ListExpr((V("first"), V("data"))), id="e", source=source),
        )
        plan = lower_to_plan(replace(program, body=body))
        for order in ("forward", "reverse", "random"):
            self.assertEqual(evaluate(plan.to_dict(), PureAnnoScript(), order=order), nested_store(PureAnnoScript()))

    def test_early_returns_and_identity(self):
        self.compare(early_return, [True, False], expected_calls=lambda _: 1)
        self.compare(nested_return, [[], [False], [True]], expected_calls=lambda _: 1)
        self.compare(partial_nested_return, [[], [False], [True]],
                     expected_calls=lambda s: 2 if s and not s[0] else 1)
        # The source compiler restricts identity comparisons, but the serialized
        # expression dialect can represent them and lowering must stay safe.
        program = compile_program(duplicates, optimize=False)
        sites = [s for s in program.body if isinstance(s, ir.Infer)]
        ret = program.body[-1]
        program = replace(program, body=program.body[:-1] +
                          (replace(ret, value=ir.Compare(ir.Var(sites[0].target), ("is",), (ir.Var(sites[1].target),))),))
        plan = lower_to_plan(program)
        self.assertTrue(all(not q.reusable for q in plan.requests))
        anno = PureAnnoScript()
        self.assertIs(evaluate(plan.to_dict(), anno), False)
        self.assertEqual(len(anno.calls), 2)

    def test_constraints_preserve_expression_and_bindings(self):
        raw = compile_program(checked)
        plan = lower_to_plan(raw)
        check = next(t for t in plan.tasks if isinstance(t, p.Check))
        original = next(s for s in raw.body if isinstance(s, ir.Constrain))
        self.assertEqual(check.condition.expression, original.condition)
        self.assertEqual(check.message.expression, original.message)
        for mode in ("eager", "strict"):
            for function, state, message in ((checked, "wrong", "验证失败"), (checked_branch, True, "branch failed"),
                                             (checked_loop, ["bad", "later"], "Constraint failed")):
                anno = PureAnnoScript(state)
                with self.assertRaisesRegex(ValueError, message):
                    evaluate(self.compile(function).to_dict(), anno, mode=mode, order="reverse")
                if mode == "strict":
                    self.assertFalse(any(call[0] in ("after", "later") for call in anno.calls))

    def test_strict_failure_prefix_and_unused_calls(self):
        for function, state, expected, failure in ((independent, None, ["a"], ("a",)),
                                                  (expression_failure, None, ["first"], ()),
                                                  (unpack_failure, [1], [], ()),
                                                  (unused_failure, None, ["bad"], ("bad",)),
                                                  (independent_loop, ["bad", "later"], ["bad"], ("bad",))):
            anno = PureAnnoScript(state, fail=failure)
            with self.assertRaises(ValueError):
                evaluate(self.compile(function).to_dict(), anno, mode="strict", order="reverse")
            self.assertEqual([x[0] for x in anno.calls], expected)

    def test_eager_concurrency_and_first_observed_error(self):
        class Concurrent(PureAnnoScript):
            def __init__(self, state=None):
                super().__init__(state)
                self.active = 0
                self.peak = 0

            async def generate(self, instructions, **kwargs):
                self.calls.append((instructions, kwargs))
                self.active += 1
                self.peak = max(self.active, self.peak)
                try:
                    await asyncio.sleep(0.005 if instructions == "b" else 0.02)
                    if instructions in self.fail:
                        raise ValueError(instructions)
                    return GenerateResponse(instructions)
                finally:
                    self.active -= 1
        plan = self.compile(independent)
        anno = Concurrent()
        anno.fail = ("a", "b")
        with self.assertRaisesRegex(ValueError, "b"):
            evaluate(plan.to_dict(), anno)
        self.assertEqual(anno.peak, 2)
        self.assertEqual(anno.active, 0)
        anno = Concurrent(["a", "b", "c"])
        self.assertEqual(evaluate(self.compile(independent_loop).to_dict(), anno), ["a", "b", "c"])
        self.assertEqual(anno.peak, 3)
        anno = Concurrent(["a", "b"])
        evaluate(self.compile(independent_loop).to_dict(), anno, mode="strict")
        self.assertEqual(anno.peak, 1)
        anno = Concurrent([True, True, True])
        self.assertEqual(evaluate(self.compile(loop_invariant).to_dict(), anno), ["same"] * 3)
        self.assertEqual(len(anno.calls), 1)  # in-flight requests also coalesce

    def test_reject_invalid_serialized_plans(self):
        data = self.compile(independent_loop).to_dict()
        for mutate in (
            lambda d: d.update(version=2),
            lambda d: d.update(extra=True),
            lambda d: d["values"][0].update(producer="absent"),
            lambda d: d["regions"][0].update(parent=d["regions"][0]["id"]),
            lambda d: d["dependencies"][0].update(source="absent"),
            lambda d: d["dependencies"][0]["mapping"].update(distance=-1),
            lambda d: d["dependencies"].clear(),
            lambda d: d["tasks"][0].update(kind=[]),
            lambda d: d["tasks"][0].update(outputs=[]),
        ):
            broken = deepcopy(data)
            mutate(broken)
            with self.assertRaises(p.PlanDecodeError):
                p.ExecutionPlan.from_dict(broken)
        with self.assertRaises(p.PlanDecodeError):
            p.ExecutionPlan.from_json('{"version":1,"version":1}')
        with self.assertRaises(p.PlanDecodeError):
            p.ExecutionPlan.from_json('{"value": NaN}')

    def test_bad_merge_and_cycle(self):
        data = self.compile(branch_duplicates).to_dict()
        merge = next(t for t in data["tasks"] if t["kind"] == "merge")
        merge["then_value"], merge["else_value"] = merge["else_value"], merge["then_value"]
        with self.assertRaises(p.PlanDecodeError):
            p.ExecutionPlan.from_dict(data)
        data = self.compile(independent).to_dict()
        edge = deepcopy(next(d for d in data["dependencies"] if d["kind"] == "data"))
        edge.update(source=edge["target"], value=None, kind="control")
        data["dependencies"].append(edge)
        with self.assertRaisesRegex(p.PlanDecodeError, "cycle"):
            p.ExecutionPlan.from_dict(data)

    def test_type_preserving_request_keys(self):
        original = compile_program(independent)
        source = original.body[0].source
        body = tuple(ir.Infer(f"v{i}", "generate", (ir.Literal(x),), (("max_length",ir.Literal(512)),), id=f"s{i}", source=source)
                     for i, x in enumerate((True, 1, 1.0, 0.0, -0.0)))
        program = replace(original, body=body + (ir.Return(ir.Literal(None), id="ret", source=source),))
        self.assertEqual(len(lower_to_plan(program).requests), 5)

    def test_no_python_or_model_execution_during_lowering(self):
        program = compile_program(compiler_tests.configured)
        def forbidden(*args, **kwargs):
            raise AssertionError("model executed")
        forbidden.__signature__ = inspect.signature(Anno.generate)
        with patch.object(compiler_tests, "configured", side_effect=AssertionError("program executed")),\
             patch.object(compiler_tests, "_never_check", side_effect=AssertionError("precondition executed")),\
             patch.object(Anno, "generate", forbidden):
            lower_to_plan(program)
        restored = ir.Program.from_json(program.to_json())
        self.assertEqual(lower_to_plan(program).to_dict(), lower_to_plan(restored).to_dict())

    def test_all_six_operations(self):
        function = compiler_tests.all_operations
        expected = function(compiler_tests.Probe({"text": "source"}, text="fixed"))
        plan = self.compile(function)
        self.assertEqual(len({q.operation for q in plan.requests}), 6)
        for mode in ("eager", "strict"):
            self.assertEqual(evaluate(plan.to_dict(), compiler_tests.Probe({"text": "source"}, text="fixed"), mode=mode, order="reverse"), expected)

    def test_lowering_diagnostics(self):
        program = compile_program(independent)
        s = program.body[0]
        broken = replace(program, body=(ir.Return(ir.Var("missing"), id=s.id, source=s.source),))
        with self.assertRaisesRegex(PlanLoweringError, "not definitely assigned.*test_plan"):
            lower_to_plan(broken)
        broken = replace(program, body=(ir.SetItem(ir.State(), ir.Literal("x"), ir.Literal(1), id=s.id, source=s.source),
                                        ir.Return(ir.Literal(None), id="ret", source=s.source)))
        with self.assertRaisesRegex(PlanLoweringError, "local container"):
            lower_to_plan(broken)
        with self.assertRaises(TypeError):
            lower_to_plan("not a Program")

    def test_existing_compiler_semantics(self):
        for function, state in ((compiler_tests.expression_order, None), (compiler_tests.loop_helpers, [[], ["special"]]),
                                (compiler_tests.formatting, ["中文", "x"]), (compiler_tests.implicit_return, False)):
            program = compile_program(function)
            expected = evaluate_ir(program.to_dict(), PureAnnoScript(state))
            for order in ("reverse", "random"):
                self.assertEqual(evaluate(lower_to_plan(program).to_dict(), PureAnnoScript(state), order=order), expected)


if __name__ == "__main__":
    unittest.main()
