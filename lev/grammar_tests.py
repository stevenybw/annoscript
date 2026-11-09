"""Required CPU-only XGrammar acceptance: python3.12 -m lev.grammar_tests.

Separate from unittest discovery: XGrammar is a test dependency, never a client
or execution dependency. No model, tokenizer files or GPU are used or downloaded.
Missing/wrong dependencies fail this entry point instead of skipping acceptance.
"""

from importlib.metadata import version
import json
import sys
import unittest
from types import SimpleNamespace

from lev.backends._grammar import ITEMS_GRAMMAR, BATCH_GRAMMAR, batch_grammar


class CompactGrammarTests(unittest.TestCase):
    grammar = ITEMS_GRAMMAR
    @classmethod
    def setUpClass(cls):
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("grammar acceptance requires Python 3.12")
        if version("xgrammar") != "0.2.0":
            raise RuntimeError("install requirements-test-grammar.txt (xgrammar==0.2.0)")
        import xgrammar as xgr
        cls.xgr = xgr
        grammar = xgr.Grammar.from_ebnf(cls.grammar)
        # Empty vocabulary: compile/match strings without any model/tokenizer.
        cls.compiled = xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=False).compile_grammar(grammar)

    def accepts(self, text):
        matcher = self.xgr.GrammarMatcher(self.compiled, terminate_without_stop_token=True)
        return matcher.accept_string(text) and matcher.is_terminated()

    def test_json_strings_unicode_and_escapes(self):
        for values in ([], [""], ["中文", "😀", " A B "], ['"\\/\b\f\n\r\t'],
                       ["\x00\x1f"], ["e\u0301", "\u2028\u2029"], ["条目"]):
            for ascii_only in (True, False):
                payload = json.dumps({"items": values}, ensure_ascii=ascii_only, separators=(",", ":"))
                with self.subTest(payload=payload):
                    self.assertTrue(self.accepts(payload))
                    self.assertEqual(json.loads(payload)["items"], values)
        self.assertTrue(self.accepts(r'{"items":["\/\u4E2D"]}'))

    def test_structure_has_no_whitespace_outside_strings(self):
        for payload in (' {"items":[]}', '{"items":[]}\n', '\n{"items":[]}',
                        '{ "items":[]}', '{"items" :[]}', '{"items": []}',
                        '{"items":[ ]}', '{"items":["a", "b"]}', '{"items":["a"] }'):
            with self.subTest(payload=payload):
                self.assertFalse(self.accepts(payload))

    def test_extra_fields_types_and_invalid_strings_are_rejected(self):
        for payload in ('{}', '[]', 'null', '{"items":null}', '{"items":"x"}',
                        '{"items":[1]}', '{"items":[true]}', '{"items":[null]}',
                        '{"items":[{}]}', '{"items":[[]]}', '{"items":[],"extra":0}',
                        '{"items":[],"items":[]}', '{"other":[]}', '{"items":["a",]}',
                        '{"items":["a\nb"]}', r'{"items":["\x20"]}',
                        r'{"items":["\u12Q4"]}', '{"items":["\x00"]}'):
            with self.subTest(payload=payload):
                self.assertFalse(self.accepts(payload))

    def test_incomplete_output_is_not_a_complete_match(self):
        payload = '{"items":["中文","条目"]}'
        for end in range(len(payload)):
            with self.subTest(end=end):
                self.assertFalse(self.accepts(payload[:end]))
        self.assertTrue(self.accepts(payload))

    def test_business_bounds_are_left_to_runtime(self):
        # Grammar remains reusable for all programs, not just application field bounds.
        self.assertTrue(self.accepts(json.dumps({"items": ["x" * 301] * 17}, separators=(",", ":"))))


class BatchGrammarTests(unittest.TestCase):
    grammar = BATCH_GRAMMAR
    setUpClass = classmethod(CompactGrammarTests.setUpClass.__func__)
    accepts = CompactGrammarTests.accepts

    def test_mixed_answers_unicode_and_escapes(self):
        for ascii_only in (False, True):
            payload = json.dumps({"answers": {
                "q1": ' 中文😀 "引号" \\路径\n\x00',
                "q2": ["", "条目"],
                "q3": [],
            }}, ensure_ascii=ascii_only, separators=(",", ":"))
            self.assertTrue(self.accepts(payload))
            for end in range(len(payload)):
                self.assertFalse(self.accepts(payload[:end]))

    def test_bad_types_extra_fields_and_outer_whitespace(self):
        good = '{"answers":{"q1":"x"}}'
        for payload in (" " + good, good + "\n", good.replace(':"x"', ': "x"'),
                        good.replace('"x"', 'null'), good.replace('"x"', 'true'),
                        good.replace('"x"', '[null]'), good.replace('"x"', '{"text":"x"}'),
                        good[:-1] + ',"extra":0}', '{}', '[]'):
            with self.subTest(payload=payload):
                self.assertFalse(self.accepts(payload))


class BoundBatchGrammarTests(unittest.TestCase):
    calls = [SimpleNamespace(id='extract"\n😀', operation="extract", arguments={}),
             SimpleNamespace(id="list", operation="generate_list", arguments={}),
             SimpleNamespace(id="text", operation="generate", arguments={})]
    grammar = batch_grammar(calls)
    setUpClass = classmethod(CompactGrammarTests.setUpClass.__func__)
    accepts = CompactGrammarTests.accepts

    def answer(self):
        return {"answers": {
            "q1": ["广州"], "q2": ["😀", "", '"\\\n'], "q3": "标题"}}

    def encode(self, value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def test_exact_mapping_unicode_escaping_and_completion(self):
        payload = self.encode(self.answer())
        self.assertTrue(self.accepts(payload))
        for end in range(len(payload)):
            self.assertFalse(self.accepts(payload[:end]))

    def test_wrong_type_for_extract_id_is_impossible(self):
        for value in ("广州", {"type": "extract", "items": ["广州"]}):
            answer = self.answer()
            answer["answers"]["q1"] = value
            self.assertFalse(self.accepts(self.encode(answer)))

    def test_unknown_missing_and_duplicate_ids_are_impossible(self):
        answer = self.answer()
        del answer["answers"]["q2"]
        self.assertFalse(self.accepts(self.encode(answer)))
        answer = self.answer()
        answer["answers"]["unknown"] = answer["answers"]["q2"]
        self.assertFalse(self.accepts(self.encode(answer)))
        good = self.encode(self.answer())
        self.assertFalse(self.accepts(good[:-2] + ',"q3":"x"}}'))
        self.assertFalse(self.accepts('{"answers":{}}'))


class NamedBatchGrammarTests(unittest.TestCase):
    calls = [SimpleNamespace(id="a", operation="extract", arguments={"field_name": ' 地点"\\\n😀 '}),
             SimpleNamespace(id="b", operation="generate_list", arguments={"field_name": "tags"}),
             SimpleNamespace(id="c", operation="generate", arguments={})]
    grammar = batch_grammar(calls)
    setUpClass = classmethod(CompactGrammarTests.setUpClass.__func__)
    accepts = CompactGrammarTests.accepts

    def test_named_and_automatic_keys_and_escaping(self):
        value = {"answers": {' 地点"\\\n😀 ': ["广州"], "tags": ["条目"], "q3": "标题"}}
        # Grammar key literals use JSON's ASCII escapes; decoded names are exact.
        encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        self.assertTrue(self.accepts(encoded))
        self.assertFalse(self.accepts(encoded.replace('tags', 'q2')))
        value["answers"]["q3"] = []
        self.assertFalse(self.accepts(json.dumps(value, ensure_ascii=True, separators=(",", ":"))))


class NullableBatchGrammarTests(unittest.TestCase):
    grammar = batch_grammar([
        SimpleNamespace(id="a", operation="generate", arguments={"nullable": True}),
        SimpleNamespace(id="b", operation="generate", arguments={}),
        SimpleNamespace(id="c", operation="generate_list", arguments={}),
    ])
    setUpClass = classmethod(CompactGrammarTests.setUpClass.__func__)
    accepts = CompactGrammarTests.accepts

    def test_only_nullable_text_accepts_null(self):
        for value in (None, "null", "中文", ""):
            good = json.dumps({"answers": {"q1": value, "q2": "标题", "q3": []}}, ensure_ascii=False, separators=(",", ":"))
            self.assertTrue(self.accepts(good))
        for value in ({"q1": [], "q2": "标题", "q3": []},
                      {"q1": None, "q2": None, "q3": []}, {"q1": None, "q2": "标题", "q3": None}):
            self.assertFalse(self.accepts(json.dumps({"answers": value}, separators=(",", ":"))))


if __name__ == "__main__":
    unittest.main()
