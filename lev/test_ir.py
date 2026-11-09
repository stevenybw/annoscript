from dataclasses import FrozenInstanceError
from copy import deepcopy
import json
from pathlib import Path
import unittest

from lev import ir


class IRTests(unittest.TestCase):
    def program(self, expr, prompt=None):
        span = ir.SourceSpan("example", "annotate", 1, 1, 2, 1)
        return ir.Program("annotate", ir.ProgramConfig(prompt), (
            ir.Return(expr, id="return", source=ir.Source(span)),
        ), span)

    def test_unicode_null_and_detached_export(self):
        program = self.program(ir.DictExpr(((ir.Literal("标题"), ir.Literal(None)),)), "规则")
        text = program.to_json()
        self.assertIn("标题", text)
        self.assertIn("null", text)
        self.assertEqual(json.loads(text), program.to_dict())
        changed = program.to_dict()
        changed["body"].clear()
        changed["config"]["system_prompt"] = "changed"
        self.assertEqual(len(program.body), 1)
        self.assertEqual(program.config.system_prompt, "规则")

    def test_tuple_and_list_have_distinct_tags(self):
        expression = ir.TupleExpr((ir.ListExpr(()), ir.TupleExpr(())))
        encoded = self.program(expression).to_dict()["body"][0]["value"]
        self.assertEqual(encoded["kind"], "tuple")
        self.assertEqual([n["kind"] for n in encoded["items"]], ["list", "tuple"])

    def test_absent_and_empty_prompt(self):
        self.assertEqual(self.program(ir.Literal(None)).to_dict()["config"], {})
        self.assertEqual(self.program(ir.Literal(None), "").to_dict()["config"], {"system_prompt": ""})

    def test_frozen_nodes_and_invalid_values(self):
        literal = ir.Literal("x")
        with self.assertRaises(FrozenInstanceError):
            literal.value = "y"
        for value in (float("nan"), float("inf"), float("-inf"), [], {}, object()):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                ir.Literal(value)
        with self.assertRaises(ValueError):
            self.program(ir.ListExpr([])).to_json()

    def test_round_trip_preserves_types_and_detaches_data(self):
        original = self.program(ir.DictExpr((
            (ir.Literal("列表"), ir.ListExpr((ir.Literal(None),))),
            (ir.Literal("元组"), ir.TupleExpr((ir.Literal(True), ir.Literal(1.5)))),
        )), "")
        data = original.to_dict()
        restored = ir.Program.from_dict(data)
        self.assertEqual(restored, original)
        self.assertEqual(ir.Program.from_json(original.to_json()), original)
        self.assertIsInstance(restored.body, tuple)
        self.assertIsInstance(restored.body[0].value.entries, tuple)
        data["body"].clear()
        data["config"]["system_prompt"] = "changed"
        self.assertEqual(restored, original)


    def test_decode_defaults_and_empty_prompt(self):
        data = self.program(ir.Literal(None)).to_dict()
        del data["body"][0]["source"]["calls"]
        program = ir.Program.from_dict(data)
        self.assertEqual(program.body[0].source.calls, ())
        self.assertEqual(program.to_dict()["config"], {})
        data["config"] = {"system_prompt": ""}
        self.assertEqual(ir.Program.from_dict(data).config.system_prompt, "")

    def test_decode_reports_document_paths(self):
        base = self.program(ir.Literal(None)).to_dict()
        cases = [
            (("format",), "other", "$.format"),
            (("version",), 2, "$.version"),
            (("version",), True, "$.version"),
            (("version",), "1", "$.version"),
            (("body",), {}, "$.body"),
            (("body", 0, "kind"), "python.eval", "$.body[0].kind"),
            (("body", 0, "kind"), "literal", "$.body[0]"),
            (("body", 0, "value", "value"), [], "$.body[0].value.value"),
            (("body", 0, "value", "value"), float("inf"), "$.body[0].value.value"),
            (("body", 0, "id"), False, "$.body[0].id"),
            (("body", 0, "id"), "", "$.body[0].id"),
            (("config", "system_prompt"), 42, "$.config.system_prompt"),
            (("source", "line"), 0, "$.source"),
            (("source", "column"), True, "$.source.column"),
            (("source", "end_line"), 0, "$.source"),
        ]
        for keys, value, expected_path in cases:
            with self.subTest(keys=keys, value=value):
                data = deepcopy(base)
                parent = data
                for key in keys[:-1]:
                    parent = parent[key]
                parent[keys[-1]] = value
                with self.assertRaises(ir.IRDecodeError) as caught:
                    ir.Program.from_dict(data)
                self.assertEqual(caught.exception.path, expected_path)

    def test_decode_rejects_missing_extra_and_duplicate_fields(self):
        data = self.program(ir.Literal(None)).to_dict()
        del data["body"][0]["value"]
        with self.assertRaisesRegex(ir.IRDecodeError, "missing required field"):
            ir.Program.from_dict(data)
        data = self.program(ir.Literal(None)).to_dict()
        data["body"][0]["typo"] = 1
        with self.assertRaisesRegex(ir.IRDecodeError, "unknown fields"):
            ir.Program.from_dict(data)
        with self.assertRaisesRegex(ir.IRDecodeError, "duplicate JSON object key"):
            ir.Program.from_json('{"format":"lev-ir", "format":"other"}')
        data = self.program(ir.Literal(None)).to_dict()
        data["body"].append(deepcopy(data["body"][0]))
        with self.assertRaisesRegex(ir.IRDecodeError, "nonempty and unique"):
            ir.Program.from_dict(data)

    def test_decode_rejects_invalid_shapes_and_operators(self):
        expressions = [
            ir.DictExpr(((ir.Literal("x"),),)),
            ir.Unary("unknown", ir.Literal(True)),
            ir.Compare(ir.Literal(1), ("eq",), ()),
            ir.Compare(ir.Literal(1), ("exec",), (ir.Literal(2),)),
            ir.BoolOp("and", (ir.Literal(True),)),
            ir.Intrinsic("eval", (ir.Literal("1"),)),
            ir.Intrinsic("json.dumps", (), (("obj", ir.Literal(1)), ("obj", ir.Literal(2)))),
            ir.Field(ir.Var("response"), "__class__"),
            ir.Var(""),
        ]
        for expression in expressions:
            with self.subTest(expression=type(expression).__name__), self.assertRaises(ir.IRDecodeError):
                ir.Program.from_dict(self.program(expression).to_dict())

    def test_decode_rejects_malformed_json_and_nonfinite_numbers(self):
        with self.assertRaisesRegex(ir.IRDecodeError, "invalid JSON at line"):
            ir.Program.from_json('{broken')
        for number in ("NaN", "Infinity", "-Infinity", "1e999"):
            text = self.program(ir.Literal(None)).to_json().replace('"value": null', f'"value": {number}')
            with self.subTest(number=number), self.assertRaises(ir.IRDecodeError):
                ir.Program.from_json(text)
        with self.assertRaisesRegex(ir.IRDecodeError, "program object"):
            ir.Program.from_json("[]")


if __name__ == "__main__":
    unittest.main()
