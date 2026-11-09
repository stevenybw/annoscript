"""Dependency projection, safe offline formats, CLI and saved-plan checks."""

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from annoscript import ir
from annoscript.compiler import compile_program
from annoscript.inspect import main
from annoscript.anno import constrain
from annoscript.plan import ExecutionPlan
from annoscript.plan_lowering import lower_to_plan
from annoscript.plan_formatter import analyze_plan, format_plan, format_plan_html, format_plan_mermaid
from annoscript import test_plan as examples


ROOT = Path(__file__).resolve().parents[1]


def data_chain(anno):
    a = anno.generate("first", max_length=512).text
    b = anno.generate(a + " next", max_length=512).text
    return b


def checked_arm(anno):
    if anno.state:
        a = anno.generate("first", max_length=512).text
        constrain(a != "")
    return anno.generate("after", max_length=512).text


def exclusive(anno):
    if anno.state:
        return anno.generate("yes", max_length=512).text
    else:
        return anno.generate("no", max_length=512).text


def true_recurrence(anno):
    text = "seed"
    for item in anno.state:
        text = anno.generate(text, max_length=512).text
    return text


def internal_loop(anno):
    text = anno.generate("first", max_length=512).text
    result = []
    for item in anno.state:
        result = result + [text + item]
    return anno.generate(json.dumps(result), max_length=512).text


def loop_export(anno):
    result = "initial"
    for item in anno.state:
        result = anno.generate(item, max_length=512).text
    return anno.generate(result, max_length=512).text


def dynamic_prompt(anno):
    return anno.generate(f"判断品牌 {anno.state}", max_length=512).text


def memory_path(anno):
    data = {"text": "initial"}
    alias = data
    a = anno.generate(data["text"], max_length=512).text
    alias["text"] = a
    return anno.generate(data["text"], max_length=512).text


def empty(anno):
    return None


class PlanProjectionTests(unittest.TestCase):
    def report(self, function, mode="eager", raw=False):
        return analyze_plan(lower_to_plan(compile_program(function, optimize=not raw)), mode=mode)

    def test_independent_calls_and_strict_failure_order(self):
        eager = self.report(examples.independent)
        self.assertEqual(eager["edges"], [])
        self.assertEqual(eager["batches"], [{"layer": 0, "operation": "generate", "sites": ["I01", "I02"]}])
        strict = self.report(examples.independent, "strict")
        self.assertEqual(strict["batches"], [])
        self.assertEqual([(e["source"], e["target"]) for e in strict["edges"]], [("I01", "I02")])
        self.assertIn("failure_order", strict["edges"][0]["kinds"])

    def test_data_path_projects_through_pure_tasks(self):
        for raw in (False, True):
            report = self.report(data_chain, raw=raw)
            edge, = report["edges"]
            self.assertEqual((edge["source"], edge["target"], edge["carried"]), ("I01", "I02", False))
            self.assertIn("data", edge["kinds"])
            self.assertGreater(len(edge["witness"]), 2)
            for left, right in zip(edge["witness"], edge["witness"][1:]):
                self.assertEqual(left["target"], right["source"])
            self.assertEqual(report["parallel"], [])

    def test_join_completion_and_constraint_barrier(self):
        report = self.report(checked_arm)
        self.assertEqual(report["edges"], [])
        self.assertEqual(len(report["checks"]), 1)
        self.assertEqual(report["parallel"][0]["sites"], ["I01", "I02"])
        report = self.report(checked_arm, "strict")
        edge, = report["edges"]
        self.assertIn("completion", edge["kinds"])
        self.assertIn("constraint", edge["kinds"])
        self.assertTrue(edge["guards"])
        self.assertEqual(len(report["checks"]), 1)
        self.assertEqual(report["parallel"], [])

    def test_memory_path_preserves_hazards_and_certainty(self):
        report = self.report(memory_path)
        edge, = report["edges"]
        self.assertIn("memory", edge["kinds"])
        self.assertIn("RAW", edge["hazards"])
        self.assertTrue(edge["conservative"])
        self.assertEqual(report["parallel"], [])

    def test_mutually_exclusive_sites_are_not_a_batch(self):
        report = self.report(exclusive)
        self.assertEqual(report["edges"], [])
        self.assertEqual(report["parallel"], [])
        self.assertEqual(report["batches"], [])
        self.assertTrue(all(n["gates"] for n in report["nodes"]))

    def test_result_accumulation_is_not_an_infer_recurrence(self):
        report = self.report(examples.independent_loop)
        self.assertEqual(report["edges"], [])
        loop, = report["loops"]
        self.assertTrue(loop["parallel_iterations"])
        self.assertTrue(loop["carried_values"])
        self.assertEqual(loop["batch_sites"], ["I01"])
        strict = self.report(examples.independent_loop, "strict")
        self.assertFalse(strict["loops"][0]["parallel_iterations"])
        self.assertTrue(any(e["carried"] and e["source"] == e["target"] for e in strict["edges"]))

    def test_true_recurrence_blocks_cross_iteration_batching(self):
        report = self.report(true_recurrence)
        edge, = report["edges"]
        self.assertEqual((edge["source"], edge["target"], edge["carried"]), ("I01", "I01", True))
        self.assertIn({"loop": "L01", "distance": 1}, edge["recurrences"])
        self.assertEqual(report["loops"][0]["batch_sites"], [])
        self.assertFalse(report["loops"][0]["parallel_iterations"])

    def test_internal_pure_loop_is_gathered_not_an_infer_recurrence(self):
        report = self.report(internal_loop)
        edge, = report["edges"]
        self.assertFalse(edge["carried"])
        self.assertEqual(edge["recurrences"], [])
        self.assertEqual(edge["via_loops"], ["L01"])
        self.assertEqual(report["unresolved_cycles"], [])

    def test_loop_exit_gathers_and_nested_coordinates_are_retained(self):
        report = self.report(loop_export)
        edge, = report["edges"]
        self.assertFalse(edge["carried"])
        self.assertEqual(edge["gather"], ["L01"])
        nested = self.report(examples.nested_loop, "strict")
        self.assertEqual(nested["nodes"][0]["loops"], ["L01", "L02"])
        loops = {r["loop"] for e in nested["edges"] for r in e["recurrences"]}
        self.assertEqual(loops, {"L01", "L02"})

    def test_shared_requests_are_not_counted_as_batch_work(self):
        report = self.report(examples.duplicates)
        self.assertEqual(report["nodes"][0]["request"], report["nodes"][1]["request"])
        self.assertEqual(report["batches"], [])
        invariant = self.report(examples.loop_invariant)
        self.assertEqual(invariant["loops"][0]["shared_sites"], ["I01"])
        self.assertEqual(invariant["loops"][0]["batch_sites"], [])

    def test_loop_check_only_enforces_strict_iteration_order(self):
        report = self.report(examples.checked_loop)
        self.assertFalse(report["loops"][0]["serial"])
        self.assertTrue(report["loops"][0]["parallel_iterations"])
        self.assertEqual(report["edges"], [])
        strict = self.report(examples.checked_loop, "strict")
        self.assertTrue(strict["loops"][0]["serial"])
        self.assertTrue(any("iteration_order" in e["kinds"] for e in strict["edges"]))

    def test_serialization_determinism_and_source_independence(self):
        plan = lower_to_plan(compile_program(dynamic_prompt))
        before = plan.to_dict()
        restored = ExecutionPlan.from_json(plan.to_json())
        with patch("annoscript.compiler.compile_program", side_effect=AssertionError("no compilation")), patch(
            "annoscript.plan_lowering.lower_to_plan", side_effect=AssertionError("no lowering")
        ):
            report = analyze_plan(restored)
            for formatter in (format_plan, format_plan_mermaid, format_plan_html):
                self.assertEqual(formatter(plan), formatter(restored))
        self.assertEqual(plan.to_dict(), before)
        self.assertIn("判断品牌 {", report["nodes"][0]["label"])
        self.assertIn("format_string", report["nodes"][0]["prompt"])
        self.assertIn("annoscript.test_plan_formatter:", format_plan(plan, show_source=True))
        report["nodes"][0]["label"] = "modified"
        self.assertNotEqual(report, analyze_plan(plan))

    def test_no_infer_and_invalid_mode(self):
        plan = lower_to_plan(compile_program(empty))
        self.assertEqual(analyze_plan(plan)["nodes"], [])
        self.assertIn("0 Infer sites", format_plan(plan))
        self.assertIn('"nodes": []', format_plan_html(plan))
        with self.assertRaises(ValueError):
            analyze_plan(plan, mode="invalid")


class PlanFormatCommandTests(unittest.TestCase):
    def setUp(self):
        self.plan = lower_to_plan(compile_program(examples.independent))

    def command(self, *args, text=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO(text or self.plan.to_json())), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            status = main(list(args))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_stdin_formats_and_mode(self):
        for fmt, formatter in (("text", format_plan), ("mermaid", format_plan_mermaid), ("html", format_plan_html)):
            status, out, err = self.command("-", "--format", fmt, "--mode", "strict", "--show-source")
            self.assertEqual((status, err), (0, ""))
            self.assertEqual(out, formatter(self.plan, mode="strict", show_source=True))

    def test_file_output_and_no_overwriting_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "输入.json", Path(directory) / "图.html"
            source.write_text(self.plan.to_json(), encoding="utf-8")
            self.assertEqual(self.command(str(source), "--format", "html", "-o", str(dest)), (0, "", ""))
            self.assertEqual(dest.read_text(encoding="utf-8"), format_plan_html(self.plan))
            before = source.read_bytes()
            status, out, err = self.command(str(source), "-o", str(source))
            self.assertEqual((status, out), (2, ""))
            self.assertIn("must differ", err)
            self.assertEqual(source.read_bytes(), before)

    def test_errors_use_strict_decoder_and_do_not_lower_ir(self):
        invalid = self.plan.to_dict()
        invalid["tasks"][0]["region"] = "missing"
        for text in (json.dumps(invalid), self.plan.to_json().replace('"version": 1', '"version": 1, "version": 1'),
                     '{"format":"lev-execution-plan","version":NaN}'):
            status, out, err = self.command("-", text=text)
            self.assertEqual((status, out), (2, ""))
            self.assertIn("annoscript.inspect:", err)
            self.assertNotIn("Traceback", err)
        status, out, err = self.command("-", "--format", "html", text=compile_program(empty).to_json())
        self.assertEqual((status, out), (2, ""))
        self.assertIn("lower the IR explicitly first", err)

    def test_untrusted_strings_cannot_inject_html_or_mermaid(self):
        program = compile_program(empty)
        prompt = '中文</script><script>alert(1)</script>"\nclick I01 "https://example.com" __PLAN_DATA__'
        statement = ir.Infer("reply", "generate", (ir.Literal(prompt),), (("max_length",ir.Literal(512)),), id="attack", source=ir.Source(program.source))
        plan = lower_to_plan(replace(program, entry=prompt, body=(statement,) + program.body))
        html = format_plan_html(plan)
        self.assertEqual(html.count("<script"), 2)
        self.assertEqual(html.count("</script>"), 2)
        self.assertNotIn("<script>alert(1)", html)
        payload = html.split('<script id="plan-data" type="application/json">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["eager"]["nodes"][0]["prompt"], prompt)
        mermaid = format_plan_mermaid(plan)
        self.assertNotIn("<script", mermaid)
        self.assertNotIn('\nclick ', mermaid)


if __name__ == "__main__":
    unittest.main()
