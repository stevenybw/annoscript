"""Compiler behavior and differential tests against ordinary Python execution."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch


from lev._test_artifacts import assert_saved_artifact
from lev.client import LevClient, precondition, system_prompt
from lev.compiler import CompileError, compile_program
from lev.lev import (
    ChoiceResponse, ExtractResponse, GenerateListResponse, GenerateResponse,
    NoulResponse, ScoreResponse, constrain,
)
from lev._test_ir_runtime import evaluate, walk_json


def _twice(value):
    return value + value


def _title(lev, suffix="!"):
    text = lev.generate("title", max_length=512).text
    return text + suffix


def repeated_calls(lev):
    text = "caller"
    a = _twice(lev.generate("once", max_length=512).text)
    b = _title(lev)
    c = _title(lev, suffix="?")
    return [a, b, c, text]


def _optional(value):
    if not value:
        return None, []
    if value == "special":
        return value, [value, value]
    result = [value]
    return value, result


def _nested(value):
    main, related = _optional(value)
    if main is None:
        return related
    return related + [main]


def loop_helpers(lev):
    result = []
    for value in lev.state:
        result = result + [_nested(value)]
    return result


def _change(data):
    data["value"] = "after"
    return "changed"


def expression_order(lev):
    data = {"value": "before"}
    result = [data["value"], _change(data), data["value"]]
    a = 1
    b = a = a + 1
    a, b = b + 1, a
    return [result, a, b]


def _pair(left, right="default"):
    return [left, right]


def keyword_order(lev):
    return _pair(right=lev.generate("right", max_length=512).text, left=lev.generate("left", max_length=512).text)


def short_circuit(lev):
    a = False and lev.generate("unreachable", max_length=512).text
    b = "present" or lev.generate("unreachable", max_length=512).text
    c = 3 < 2 < int(lev.generate("unreachable", max_length=512).text)
    d = True and lev.generate("reachable", max_length=512).text
    e = "" or lev.generate("fallback", max_length=512).text
    return [a, b, c, d, e]


def context_alias(lev):
    engine = lev
    return _title(engine)


def _falls_through(value):
    if value:
        return "yes"


def implicit_return(lev):
    return _falls_through(lev.state)


def constrained(lev):
    text = lev.generate("text", max_length=512).text
    if lev.state:
        constrain(len(text) > 0 and text.strip() != "", message="非空")
    return text


def all_operations(lev):
    a = lev.noul("binary").noul
    b = lev.choice("choose", {"a": "first"}).choice
    c = lev.score("rate", ["bad", "good"]).score
    d = lev.extract("extract", source_field="text", max_items=8, max_length_per_item=58).items
    e = lev.generate("generate", max_length=512).text
    f = lev.generate_list("list", max_items=8, max_length_per_item=58).items
    return [a, b, c, d, e, f]


def formatting(lev):
    result = []
    for index, value in enumerate(lev.state, start=2):
        result = result + [f"第{index:03d}项：{value!r}"]
    result = result + [json.dumps({"条目": result}, ensure_ascii=False)]
    return result


def _never_check(data):
    raise AssertionError("Preconditions must not execute while compiling")


@system_prompt("公共规则")
@precondition(_never_check)
def configured(lev):
    return _title(lev)


_CAPTURE = {"names": ["初始"]}


def captured(lev):
    return _CAPTURE


def recursive(lev):
    return recursive(lev)


def _mutual_a(lev):
    return _mutual_b(lev)


def _mutual_b(lev):
    return _mutual_a(lev)


def mutual_recursion(lev):
    return _mutual_a(lev)


def unsupported_while(lev):
    while lev.state:
        pass


def unsupported_comprehension(lev):
    return [item for item in lev.state]


def unsupported_loop_return(lev):
    for item in lev.state:
        return item


def unsupported_break(lev):
    for item in lev.state:
        break


def unsupported_try(lev):
    try:
        return lev.state
    except ValueError:
        return None


def dynamic_call(lev):
    function = _title
    return function(lev)


def missing_name(lev):
    return nonexistent_symbol


def unbound_name(lev):
    if lev.state:
        text = "set"
    return text


def write_input(lev):
    data = lev.state
    data["key"] = "bad"
    return data


def write_nested_input(lev):
    data = {"input": lev.state}
    alias = data["input"]
    alias["key"] = "bad"
    return data


def write_capture(lev):
    data = _CAPTURE
    data["names"] = []
    return data


def write_loop_alias(lev):
    data = {}
    for index in range(2):
        data["bad"] = index
        data = lev.state
    return data


def _bad_helper(data):
    data["key"] = "bad"
    return data


def write_helper_input(lev):
    return _bad_helper(lev.state)


def invalid_model_arguments(lev):
    return lev.generate("title", bogus=1, max_length=512).text


def effectful_constraint(lev):
    constrain(lev.generate("title", max_length=512).text != "")
    return None


def forbidden_python_call(lev):
    print("must not run")
    return None


def invalid_unpack(lev):
    a, *b = lev.state
    return a


def return_context(lev):
    return lev


def invalid_signature(a, b):
    return None


class Probe:
    """A deterministic model double with an observable call sequence."""

    def __init__(self, state=None, text=None):
        self.state = state
        self.calls = []
        self.text = text

    def generate(self, instructions, **kwargs):
        self.calls.append(("generate", instructions, deepcopy(kwargs)))
        text = self.text if self.text is not None else f"{instructions}:{len(self.calls)}"
        return GenerateResponse(text)

    def choice(self, instructions, criteria):
        self.calls.append(("choice", instructions, deepcopy(criteria)))
        return ChoiceResponse(next(iter(criteria)), 0.9, {key: 1.0 for key in criteria})

    def noul(self, instructions):
        self.calls.append(("noul", instructions))
        return NoulResponse(0.8)

    def score(self, instructions, criteria):
        self.calls.append(("score", instructions, deepcopy(criteria)))
        return ScoreResponse({0: 0.2, 1: 0.8}, 0.8, {0: "bad", 1: "good"}, 0.9)

    def extract(self, instructions, **kwargs):
        self.calls.append(("extract", instructions, deepcopy(kwargs)))
        return ExtractResponse([self.state[kwargs["source_field"]]])

    def generate_list(self, instructions, **kwargs):
        self.calls.append(("generate_list", instructions, deepcopy(kwargs)))
        return GenerateListResponse(["entry"])


class CompilerTests(unittest.TestCase):
    def compare(self, function, state=None):
        original = Probe(deepcopy(state))
        expected = function(original)
        for optimize in (False, True):
            compiled = Probe(deepcopy(state))
            program = compile_program(function, optimize=optimize)
            actual = evaluate(json.loads(program.to_json()), compiled)
            self.assertEqual(actual, expected)
            self.assertEqual(compiled.calls, original.calls)
            self.assertEqual(compiled.state, original.state)
        return actual, compiled.calls, program.to_dict()

    def test_repeated_calls_arguments_and_local_isolation(self):
        actual, calls, data = self.compare(repeated_calls)
        self.assertEqual(len(calls), 3)
        self.assertEqual(actual, ["once:1once:1", "title:2!", "title:3?", "caller"])
        self.assertFalse(any(n.get("kind") in ("call", "function") for n in walk_json(data)))

    def test_nested_returns_and_finite_loops(self):
        for state in ([], ["", "ordinary", "special", ""]):
            with self.subTest(state=state):
                self.compare(loop_helpers, state)

    def test_expression_order_and_assignment(self):
        self.assertEqual(self.compare(expression_order)[0], [["before", "changed", "after"], 3, 2])
        self.compare(keyword_order)

    def test_short_circuit_skips_inference(self):
        _, calls, _ = self.compare(short_circuit)
        self.assertEqual([c[1] for c in calls], ["reachable", "fallback"])

    def test_injected_context_alias(self):
        self.compare(context_alias)

    def test_implicit_return(self):
        self.assertIsNone(self.compare(implicit_return, False)[0])
        self.assertEqual(self.compare(implicit_return, True)[0], "yes")

    def test_all_six_operations_and_response_fields(self):
        _, calls, data = self.compare(all_operations, {"text": "source"})
        self.assertEqual(len(calls), 6)
        self.assertEqual({n["operation"] for n in walk_json(data) if n.get("kind") == "infer"},
                         {"noul", "choice", "score", "extract", "generate", "generate_list"})

    def test_fstrings_and_json_encoding(self):
        self.compare(formatting, ["中文", 'a"b', "line\nbreak"])

    def test_constraint_is_a_pure_expression_and_path_local(self):
        data = compile_program(constrained).to_dict()
        constraints = [n for n in walk_json(data) if n.get("kind") == "constrain"]
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0]["condition"]["kind"], "bool_op")
        self.assertIn("intrinsic", {n.get("kind") for n in walk_json(constraints[0])})
        self.assertEqual(evaluate(data, Probe(False, text=" ")), " ")
        with self.assertRaisesRegex(ValueError, "非空"):
            evaluate(data, Probe(True, text=" "))
        self.assertEqual(evaluate(data, Probe(True, text="ok")), "ok")

    def test_config_and_preconditions_are_separate(self):
        with patch.object(LevClient, "_validate_inputs", side_effect=AssertionError("must not run")):
            data = compile_program(configured).to_dict()
        self.assertEqual(data["config"], {"system_prompt": "公共规则"})
        serialized = json.dumps(data)
        self.assertNotIn("_never_check", serialized)
        self.assertNotIn("precondition", serialized)

    def test_captured_globals_are_detached_snapshots(self):
        program = compile_program(captured)
        _CAPTURE["names"].append("later")
        try:
            self.assertEqual(evaluate(program.to_dict(), Probe()), {"names": ["初始"]})
            self.assertEqual(evaluate(compile_program(captured).to_dict(), Probe()), _CAPTURE)
        finally:
            _CAPTURE["names"].pop()

    def test_compile_never_executes_the_function(self):
        import sys
        executed = []
        watched = {repeated_calls.__code__, _title.__code__, _twice.__code__}

        def trace(frame, event, arg):
            if event == "call" and frame.f_code in watched:
                executed.append(frame.f_code.co_name)

        previous = sys.getprofile()
        sys.setprofile(trace)
        try:
            compile_program(repeated_calls)
        finally:
            sys.setprofile(previous)
        self.assertEqual(executed, [])

    def test_determinism_and_inline_source_chain(self):
        first = compile_program(loop_helpers)
        second = compile_program(loop_helpers)
        self.assertEqual(first.to_json(), second.to_json())
        statements = [n for n in walk_json(first.to_dict()) if "source" in n and "id" in n]
        self.assertEqual(len({n["id"] for n in statements}), len(statements))
        self.assertTrue(any(len(n["source"]["calls"]) == 2 for n in statements))
        self.assertNotIn(str(Path(__file__).resolve().parent), first.to_json())

    def test_rejected_language_constructs_and_writes(self):
        cases = {
            recursive: "Recursive", mutual_recursion: "Recursive",
            unsupported_while: "While", unsupported_comprehension: "ListComp",
            unsupported_loop_return: "Returns inside loops", unsupported_break: "Break",
            unsupported_try: "Try", dynamic_call: "Cannot capture function",
            missing_name: "Unknown name", unbound_name: "may be unbound",
            write_input: "read-only", write_nested_input: "read-only", write_capture: "read-only",
            write_loop_alias: "read-only", write_helper_input: "read-only",
            invalid_model_arguments: "Invalid call arguments", effectful_constraint: "must be pure",
            forbidden_python_call: "Only known intrinsics", invalid_unpack: "only names",
            return_context: "cannot be stored", invalid_signature: "exactly one",
        }
        for function, message in cases.items():
            with self.subTest(function=function.__name__), self.assertRaisesRegex(CompileError, message):
                compile_program(function)

    def test_diagnostic_contains_source_and_inline_chain(self):
        with self.assertRaises(CompileError) as caught:
            compile_program(write_helper_input)
        error = caught.exception
        self.assertEqual(error.source.span.function, "_bad_helper")
        self.assertEqual(error.source.calls[-1].function, "write_helper_input")
        self.assertGreater(error.source.span.line, 0)
        self.assertIn("inlined from", str(error))

    def test_unavailable_source_and_closure(self):
        with self.assertRaisesRegex(CompileError, "source is unavailable"):
            compile_program(eval("lambda lev: None"))
        data = "captured"

        def closure(lev):
            return data

        with self.assertRaisesRegex(CompileError, "Closures"):
            compile_program(closure)
        with self.assertRaisesRegex(CompileError, "ordinary Python function"):
            compile_program(len)


if __name__ == "__main__":
    unittest.main()
