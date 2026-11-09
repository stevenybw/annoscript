from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from lev import ir
from lev.compiler import compile_program
from lev.formatter import format_ir
from lev.inspect import main


_ROOT = Path(__file__).resolve().parents[1]


def example_program():
    span = ir.SourceSpan("missing.source.module", "annotate", 1, 1, 12, 1)
    source = ir.Source(span, (ir.SourceSpan("caller", "entry", 5, 1, 5, 20),))
    result = ir.Var("decision")
    body = (
        ir.Assign("text", ir.GetItem(ir.State(), ir.Literal("text")), id="s1", source=source),
        ir.Infer("decision", "choice", (ir.Literal("判断事件"), ir.DictExpr((
            (ir.Literal("yes"), ir.Literal("事件")), (ir.Literal("no"), ir.Literal("非事件")),
        ))), (), id="s2", source=source),
        ir.If(ir.Compare(ir.Field(result, "choice"), ("eq",), (ir.Literal("yes"),)), (
            ir.Constrain(ir.Compare(ir.Intrinsic("len", (ir.Var("text"),)), ("gt",), (ir.Literal(0),)),
                         ir.Literal("必须有正文"), id="s3", source=source),
            ir.Return(ir.Var("text"), id="s4", source=source),
        ), (
            ir.Return(ir.Literal(None), id="s5", source=source),
        ), id="s6", source=source),
    )
    return ir.Program("demo", ir.ProgramConfig("公共规则"), body, span)


class FormatterTests(unittest.TestCase):
    def test_readable_model_call_branch_and_constraint(self):
        program = example_program()
        self.assertEqual(format_ir(program), "\n".join((
            "# Lev IR v1 (display notation)",
            "system_prompt '公共规则'",
            "program 'demo':",
            "    text = lev.state['text']",
            "    decision = lev.choice('判断事件', {'yes': '事件', 'no': '非事件'})",
            "    if (decision.choice == 'yes'):",
            "        constrain((len(text) > 0), message='必须有正文')",
            "        return text",
            "    else:",
            "        return None",
            "",
        )))
        self.assertEqual(format_ir(ir.Program.from_json(program.to_json())), format_ir(program))

    def test_source_comments_are_optional(self):
        program = example_program()
        self.assertNotIn("missing.source.module", format_ir(program))
        text = format_ir(program, show_source=True)
        self.assertIn("# [s2] missing.source.module:1:1 (annotate)", text)
        self.assertIn("inlined from caller:5:1 (entry)", text)
        self.assertIn("decision = lev.choice", text)

    def test_all_compiled_constructs_render_without_executing_models(self):
        from lev import test_compiler as examples
        for function in (
            examples.loop_helpers, examples.expression_order, examples.short_circuit,
            examples.constrained, examples.all_operations, examples.formatting,
        ):
            with self.subTest(function=function.__name__):
                program = compile_program(function)
                before = program.to_dict()
                text = format_ir(program)
                self.assertTrue(text.endswith("\n"))
                self.assertEqual(before, program.to_dict())
                self.assertEqual(text, format_ir(ir.Program.from_json(program.to_json())))
        loops = format_ir(compile_program(examples.loop_helpers))
        self.assertIn("for ", loops)
        formatting = format_ir(compile_program(examples.formatting))
        self.assertIn("format_string(", formatting)
        self.assertIn("format_value(", formatting)
        self.assertIn("conversion='r'", formatting)
        self.assertIn("json.dumps(", formatting)
        calls = format_ir(compile_program(examples.all_operations))
        for operation in ("noul", "choice", "score", "extract", "generate", "generate_list"):
            self.assertIn(f"lev.{operation}(", calls)

    def test_empty_blocks_literals_and_unusual_names(self):
        program = example_program()
        statement = ir.Return(ir.TupleExpr((ir.Literal("中文\n'{}\\"),)), id="s1", source=ir.Source(program.source))
        program = replace(program, config=ir.ProgramConfig(), body=(statement,))
        text = format_ir(program)
        self.assertIn(repr("中文\n'{}\\"), text)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertTrue(text.rstrip().endswith(",)"))
        self.assertNotIn("system_prompt", text)
        self.assertEqual(format_ir(replace(program, body=())).splitlines()[-1], "    pass")
        statement = replace(statement, value=ir.Var("has space"))
        self.assertIn("var['has space']", format_ir(replace(program, body=(statement,))))


class InspectCommandTests(unittest.TestCase):
    def run_command(self, *args, stdin=None):
        return subprocess.run(
            [sys.executable, "-m", "lev.inspect", *args], input=stdin,
            text=True, encoding="utf-8", capture_output=True, cwd=_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=10,
        )

    def test_arbitrary_path_and_source_flag(self):
        program = example_program()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "任意 program.json"
            path.write_text(program.to_json(), encoding="utf-8")
            result = self.run_command(str(path), "--show-source")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout, format_ir(program, show_source=True))

    def test_stdin_and_help(self):
        program = example_program()
        result = self.run_command("-", stdin=program.to_json())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, format_ir(program))
        result = self.run_command("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--show-source", result.stdout)

    def test_invalid_files_report_errors_without_tracebacks(self):
        for text in ("not JSON", '{"format":"lev-ir","version":99}'):
            with self.subTest(text=text):
                result = self.run_command("-", stdin=text)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("lev.inspect:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_command(str(Path(directory) / "missing.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("lev.inspect:", result.stderr)

    def test_main_can_be_embedded_without_exiting_process(self):
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(example_program().to_json())), patch("sys.stdout", output):
            status = main(["-"])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), format_ir(example_program()))


if __name__ == "__main__":
    unittest.main()
