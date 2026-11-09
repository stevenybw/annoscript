"""Optimization regressions, including IR that did not come from our compiler."""

from copy import deepcopy
from dataclasses import replace
import json
import random
import unittest
from unittest.mock import patch

from lev import ir
from lev._test_ir_runtime import evaluate
from lev.compiler import compile_program
from lev.passes import optimize_program
from lev.passes._analysis import walk_statements
from lev.passes.control_flow import simplify_control_flow
from lev.passes.dead_assignments import eliminate_dead_assignments
from lev.passes.folding import fold_constants
from lev.passes.propagation import propagate_values
from lev.passes.temporaries import merge_temporaries
from lev.test_compiler import Probe


L = ir.Literal
V = ir.Var


def alias_snapshot(lev):
    data = {"value": "before"}
    alias = data
    first = data["value"]
    alias["value"] = "after"
    return [first, data, alias]


def consume_iterator(lev):
    iterator = enumerate(lev.state)
    alias = iterator
    result = []
    for index, value in iterator:
        result = result + [value]
    for index, value in alias:
        result = result + ["must not happen"]
    return result


def loop_snapshots(lev):
    current = "initial"
    initial = current
    result = []
    for value in lev.state:
        previous = current
        current = value
        result = result + [previous, current, initial]
    return [current, result]


def unused_failure(lev):
    lev.generate("before", max_length=512)
    unused = int(lev.state)
    lev.generate("after", max_length=512)
    return None


class PassTests(unittest.TestCase):
    def setUp(self):
        self.serial = 0
        self.span = ir.SourceSpan("unavailable.module", "entry", 1, 1, 2, 1)
        self.source = ir.Source(self.span, (self.span,))

    def s(self, cls, *args, **kwargs):
        self.serial += 1
        return cls(*args, **kwargs, id=f"statement-{self.serial}", source=self.source)

    def program(self, *body):
        return ir.Program("entry", ir.ProgramConfig("中文规则"), tuple(body), self.span)

    def outcome(self, program, state=None):
        context = Probe(deepcopy(state))
        try:
            value = evaluate(program.to_dict(), context)
            # Preserve distinctions such as False/0 and -0.0/0.0 in comparisons.
            outcome = ("return", json.dumps(value, ensure_ascii=False))
        except Exception as exc:
            outcome = (type(exc).__name__, str(exc))
        return outcome, context.calls, context.state

    def equivalent(self, program, states=(None,), transform=optimize_program):
        before = program.to_json()
        optimized = transform(program)
        for state in states:
            with self.subTest(state=state):
                self.assertEqual(self.outcome(program, state), self.outcome(optimized, state))
        self.assertEqual(program.to_json(), before)
        return optimized

    def test_copy_rebinding_preserves_old_value(self):
        program = self.program(
            self.s(ir.Assign, "a", ir.State()),
            self.s(ir.Assign, "b", V("a")),
            self.s(ir.Assign, "c", V("b")),
            self.s(ir.Assign, "a", L("new")),
            self.s(ir.Return, ir.ListExpr((V("a"), V("b"), V("c")))),
        )
        self.equivalent(program, ("old", None), propagate_values)
        self.equivalent(program, ("old", None))

    def test_branch_join_preserves_scalar_types_and_signed_zero(self):
        for left, right in ((True, 1), (1, 1.0), (0.0, -0.0)):
            program = self.program(
                self.s(ir.If, ir.State(), (self.s(ir.Assign, "x", L(left)),),
                       (self.s(ir.Assign, "x", L(right)),)),
                self.s(ir.Return, V("x")),
            )
            result = self.equivalent(program, (True, False))
            self.assertIsInstance(result.body[-1].value, ir.Var)

    def test_branch_with_return_does_not_pollute_join(self):
        program = self.program(
            self.s(ir.If, ir.State(), (self.s(ir.Return, L("early")),),
                   (self.s(ir.Assign, "x", L("late")),)),
            self.s(ir.Return, V("x")),
        )
        result = self.equivalent(program, (False, True), propagate_values)
        self.assertEqual(result.body[-1].value, L("late"))

    def test_copy_and_constant_loop_invalidation(self):
        program = self.program(
            self.s(ir.Assign, "value", L("first")),
            self.s(ir.Assign, "snapshot", V("value")),
            self.s(ir.Assign, "last", L(None)),
            self.s(ir.For, ir.NameTarget("value"), ir.State(), (
                self.s(ir.Assign, "last", V("value")),
            )),
            self.s(ir.Return, ir.ListExpr((V("value"), V("snapshot"), V("last")))),
        )
        self.equivalent(program, ([], ["a"], ["a", "b"]))

    def test_loop_liveness_keeps_previous_iteration(self):
        program = self.program(
            self.s(ir.Assign, "a", L(1)),
            self.s(ir.Assign, "b", L(2)),
            self.s(ir.For, ir.NameTarget("i"), ir.State(), (
                self.s(ir.Assign, "previous", V("a")),
                self.s(ir.Assign, "a", V("b")),
                self.s(ir.Assign, "b", ir.Binary("add", V("previous"), V("b"))),
            )),
            self.s(ir.Return, ir.TupleExpr((V("a"), V("b")))),
        )
        self.equivalent(program, ([], [0], [0, 0, 0, 0]), eliminate_dead_assignments)
        self.equivalent(program, ([], [0], [0, 0, 0, 0]))

    def test_loop_only_bindings_can_still_be_unbound(self):
        program = self.program(
            self.s(ir.For, ir.NameTarget("i"), ir.State(), (
                self.s(ir.Assign, "x", V("i")),
            )),
            self.s(ir.Assign, "unused", V("x")),
            self.s(ir.Return, L(None)),
        )
        self.equivalent(program, ([], [1]))

    def test_scalar_folding_and_fstrings(self):
        expressions = (
            ir.Unary("negative", L(3)),
            ir.Binary("add", L("中"), L("文")),
            ir.Compare(L(1), ("lt", "lt"), (L(2), L(3))),
            ir.Intrinsic("len", (L("中文"),)),
            ir.Intrinsic("int", (L("ff"),), (("base", L(16)),)),
            ir.Intrinsic("str.strip", (L("  abc  "),)),
            ir.FormatString((L("label_"), ir.FormatValue(L("brand")))),
            ir.FormatValue(L("中文"), "a", L("")),
        )
        for expression in expressions:
            program = self.program(self.s(ir.Return, expression))
            result = self.equivalent(program, transform=fold_constants)
            self.assertIsInstance(result.body[0].value, ir.Literal)

    def test_folding_retains_nonempty_and_dynamic_format_specs(self):
        for spec in (L("03d"), ir.State()):
            program = self.program(self.s(ir.Return, ir.FormatValue(L(3), spec=spec)))
            result = self.equivalent(program, ("03d",), fold_constants)
            self.assertIsInstance(result.body[0].value, ir.FormatValue)

    def test_folding_retains_exceptions_and_nonfinite_results(self):
        expressions = (
            ir.Intrinsic("int", (L("bad"),)),
            ir.Binary("add", L(1), L("bad")),
            ir.Binary("add", L(1e308), L(1e308)),
            ir.FormatValue(L("x"), spec=L(None)),
            ir.Compare(L(1), ("lt",), (L("x"),)),
        )
        for expression in expressions:
            program = self.program(self.s(ir.Return, expression))
            result = self.equivalent(program, transform=fold_constants)
            self.assertEqual(result.body[0].value, expression)

    def test_folding_does_not_create_large_constants(self):
        expression = ir.Binary("add", L("x" * 40000), L("y" * 40000))
        program = self.program(self.s(ir.Return, expression))
        self.assertEqual(fold_constants(program).body[0].value, expression)

    def test_folding_leaves_non_scalar_intrinsics(self):
        for name, args in (("range", (L(2),)), ("enumerate", (L("ab"),)),
                           ("json.dumps", (L(1),))):
            expression = ir.Intrinsic(name, args)
            program = self.program(self.s(ir.Assign, "value", expression), self.s(ir.Return, L(None)))
            self.assertEqual(fold_constants(program), program)

    def test_short_circuit_does_not_evaluate_skipped_operands(self):
        bad = ir.Intrinsic("int", (L("bad"),))
        for expr in (ir.BoolOp("and", (L(False), bad)), ir.BoolOp("or", (L("yes"), bad)),
                     ir.Compare(L(3), ("lt", "lt"), (L(2), bad))):
            program = self.program(self.s(ir.Return, expr))
            result = self.equivalent(program)
            self.assertIsInstance(result.body[0].value, ir.Literal)

    def test_control_flow_removes_only_unreachable_inference(self):
        program = self.program(
            self.s(ir.If, L(False), (self.s(ir.Infer, "bad", "generate", (L("never"),), ()),)),
            self.s(ir.Infer, "unused", "generate", (L("always"),), ()),
            self.s(ir.Return, L(None)),
            self.s(ir.Infer, "bad2", "generate", (L("after return"),), ()),
        )
        result = self.equivalent(program)
        calls = [s for s in walk_statements(result.body) if isinstance(s, ir.Infer)]
        self.assertEqual([s.args[0].value for s in calls], ["always"])

    def completion_program(self, initial=False, last=True):
        return self.program(
            self.s(ir.Assign, "arbitrary_flag", L(initial)),
            self.s(ir.Assign, "result", L(None)),
            self.s(ir.If, ir.State(), (
                self.s(ir.Assign, "result", L("early")),
                self.s(ir.Assign, "arbitrary_flag", L(last)),
            )),
            self.s(ir.If, ir.Unary("not", V("arbitrary_flag")), (
                self.s(ir.Infer, "generated", "generate", (L("late"),), ()),
                self.s(ir.Assign, "result", ir.Field(V("generated"), "text")),
            )),
            self.s(ir.Return, V("result")),
        )

    def test_completion_guards_form_one_if_else(self):
        program = self.completion_program()
        result = self.equivalent(program, (True, False))
        branches = [s for s in walk_statements(result.body) if isinstance(s, ir.If)]
        self.assertEqual(len(branches), 1)
        self.assertTrue(branches[0].else_body)
        self.assertNotIn("arbitrary_flag", result.to_json())
        self.assertEqual(branches[0].id, program.body[2].id)

    def test_completion_guard_requires_exact_boolean_states(self):
        for initial, last in ((True, True), (False, False), (0, 1)):
            program = self.completion_program(initial, last)
            result = self.equivalent(program, (True, False), simplify_control_flow)
            self.assertEqual(sum(isinstance(s, ir.If) for s in result.body), 2)

    def test_dead_copies_are_removed_but_failing_reads_remain(self):
        program = self.program(
            self.s(ir.Assign, "x", L(1)),
            self.s(ir.Assign, "unused_copy", V("x")),
            self.s(ir.Assign, "unused_read", ir.GetItem(ir.State(), L("missing"))),
            self.s(ir.Return, L(None)),
        )
        result = self.equivalent(program, ({}, {"missing": 1}), eliminate_dead_assignments)
        self.assertEqual(len(result.body), 2)
        self.assertEqual(result.body[0].target, "unused_read")

    def test_dead_unbound_reads_and_unpack_failures_remain(self):
        programs = (
            self.program(self.s(ir.Assign, "unused", V("absent")), self.s(ir.Return, L(None))),
            self.program(self.s(ir.Unpack, ir.TupleTarget((ir.NameTarget("a"), ir.NameTarget("b"))),
                                ir.State()), self.s(ir.Return, L(None))),
            self.program(self.s(ir.Assign, "unused", ir.DictExpr(((ir.ListExpr(()), L(1)),))),
                         self.s(ir.Return, L(None))),
        )
        for program in programs:
            result = self.equivalent(program, ([1], [1, 2]))
            self.assertEqual(len(result.body), 2)

    def test_adjacent_whole_expression_merges_keep_consumer_metadata(self):
        for cls in (ir.Assign, ir.Return, ir.If, ir.For):
            producer = self.s(ir.Assign, "temporary", ir.State())
            if cls is ir.Assign:
                consumer = self.s(cls, "target", V("temporary"))
            elif cls is ir.Return:
                consumer = self.s(cls, V("temporary"))
            elif cls is ir.If:
                consumer = self.s(cls, V("temporary"), ())
            else:
                consumer = self.s(cls, ir.NameTarget("i"), V("temporary"), ())
            program = self.program(producer, consumer, self.s(ir.Return, L(None)))
            result = self.equivalent(program, ([], [1]), merge_temporaries)
            self.assertEqual(result.body[0].id, consumer.id)
            self.assertEqual(result.body[0].source, consumer.source)
            self.assertEqual(len(result.body), 2)

    def test_merging_does_not_move_failure_into_short_circuit(self):
        program = self.program(
            self.s(ir.Assign, "temporary", ir.Intrinsic("int", (L("bad"),))),
            self.s(ir.Return, ir.BoolOp("and", (L(False), V("temporary")))),
        )
        self.assertEqual(merge_temporaries(program), program)
        self.equivalent(program)

    def test_merging_does_not_cross_inference_or_change_argument_order(self):
        program = self.program(
            self.s(ir.Assign, "temporary", ir.Intrinsic("int", (ir.State(),))),
            self.s(ir.Infer, "response", "generate", (L("after computation"),), ()),
            self.s(ir.Return, V("temporary")),
        )
        self.assertEqual(merge_temporaries(program), program)
        self.equivalent(program, ("bad", "123"))
        program = self.program(
            self.s(ir.Assign, "temporary", ir.Intrinsic("int", (ir.State(),))),
            self.s(ir.Infer, "response", "generate", (V("temporary"),), ()),
            self.s(ir.Return, L(None)),
        )
        self.assertEqual(merge_temporaries(program), program)

    def test_constraint_expressions_and_bindings_are_preserved(self):
        constraint = self.s(ir.Constrain, ir.Compare(ir.Intrinsic("len", (V("text"),)),
                                                     ("gt",), (L(0),)), V("message"))
        program = self.program(
            self.s(ir.Infer, "response", "generate", (L("before"),), ()),
            self.s(ir.Assign, "text", ir.State()),
            self.s(ir.Assign, "message", L("不能为空")),
            constraint,
            self.s(ir.Infer, "second", "generate", (L("after"),), ()),
            self.s(ir.Return, V("text")),
        )
        result = self.equivalent(program, ("", "ok"))
        self.assertEqual([s for s in result.body if isinstance(s, ir.Constrain)], [constraint])
        self.assertTrue(any(isinstance(s, ir.Assign) and s.target == "message" for s in result.body))
        self.assertTrue(any(isinstance(s, ir.Assign) and s.target == "text" for s in result.body))

    def test_constraints_are_not_assumptions_and_constant_constraints_stay(self):
        for condition in (True, False):
            constraint = self.s(ir.Constrain, L(condition), L("failure"))
            program = self.program(constraint, self.s(ir.Return, L(None)))
            result = self.equivalent(program)
            self.assertEqual(result.body[0], constraint)

    def test_shared_container_and_iterator_semantics_match_python(self):
        for function, states in ((alias_snapshot, (None,)),
                                 (consume_iterator, ([], ["a", "b"])),
                                 (loop_snapshots, ([], ["a"], ["a", "b", "c"]))):
            program = compile_program(function, optimize=False)
            result = self.equivalent(program, states)
            for state in states:
                expected = function(Probe(deepcopy(state)))
                self.assertEqual(evaluate(result.to_dict(), Probe(deepcopy(state))), expected)

    def test_model_call_prefix_and_runtime_failures_match_python(self):
        raw = compile_program(unused_failure, optimize=False)
        self.equivalent(raw, ("bad", "2"))
        for state in ("bad", "2"):
            context = Probe(state)
            try:
                value = unused_failure(context)
                expected = ("return", json.dumps(value))
            except Exception as exc:
                expected = (type(exc).__name__, str(exc))
            actual, calls, _ = self.outcome(optimize_program(raw), state)
            self.assertEqual(actual, expected)
            self.assertEqual(calls, context.calls)

    def test_determinism_idempotence_roundtrip_and_no_source_execution(self):
        raw = compile_program(loop_snapshots, optimize=False)
        with patch("lev.compiler.compile_program", side_effect=AssertionError("must not compile")), \
             patch.object(Probe, "generate", side_effect=AssertionError("must not execute")):
            restored = ir.Program.from_json(raw.to_json())
            result = self.equivalent(restored, ([], ["a", "b"]))
            self.assertEqual(result.to_json(), optimize_program(raw).to_json())
            self.assertEqual(result.to_json(), optimize_program(result).to_json())
            self.assertEqual(result.to_json(), ir.Program.from_json(result.to_json()).to_json())
        original_nodes = {s.id: s for s in walk_statements(raw.body)}
        nodes = list(walk_statements(result.body))
        self.assertEqual(len(nodes), len({s.id for s in nodes}))
        for statement in nodes:
            self.assertEqual(statement.source, original_nodes[statement.id].source)
        self.assertEqual(raw.config, result.config)


    def test_seeded_rebinding_and_branch_programs(self):
        rng = random.Random(807)
        names = ("a", "b", "c")
        for _ in range(60):
            body = [self.s(ir.Assign, name, L(rng.randrange(5))) for name in names]
            for _ in range(8):
                target, source = rng.choice(names), rng.choice(names)
                assignment = self.s(ir.Assign, target, V(source))
                if rng.choice((True, False)):
                    body.append(self.s(ir.If, ir.State(), (assignment,), (
                        self.s(ir.Assign, target, L(rng.randrange(5))),
                    )))
                else:
                    body.append(assignment)
            body.append(self.s(ir.Return, ir.ListExpr(tuple(map(V, names)))))
            self.equivalent(self.program(*body), (True, False))

    def test_public_api_rejects_non_program(self):
        with self.assertRaisesRegex(TypeError, "ir.Program"):
            optimize_program({})


if __name__ == "__main__":
    unittest.main()
