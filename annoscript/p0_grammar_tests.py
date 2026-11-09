"""Pinned decoder acceptance tests (run with XGrammar substring2, no inference)."""
import itertools
import json
import random
import unittest
from importlib.metadata import version
from types import SimpleNamespace
import xgrammar as xgr
from annoscript.backends._bounded_grammar import grammar


def compiled(text):
    return xgr.GrammarCompiler(xgr.TokenizerInfo([]),cache_enabled=False).compile_grammar(text)

def accepts(ctx,text):
    m=xgr.GrammarMatcher(ctx,terminate_without_stop_token=True)
    return m.accept_string(text) and m.is_terminated()

def wire(items):return json.dumps({'items':items},ensure_ascii=False,separators=(',',':'))

def call(op='extract',id='a',**args):
    source = args.pop('source', None)
    return SimpleNamespace(id=id,operation=op,arguments=args,extraction_source=source)


class BoundedGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):assert version('xgrammar')=='0.2.5+lev.substring2.3'

    def test_example_locations_never_expand_current_source(self):
        from annoscript._test_synthetic import REVIEW_INPUT, LAUNCH_INPUT
        from annoscript.anno import Example
        from annoscript.backends.base import normalize_call, InferenceContext
        c = normalize_call("q", "extract", ("地点",),
                           {"source_field": "text", "max_items": 2, "max_length_per_item": 80,
                            "examples": [Example(state=REVIEW_INPUT, answer=[]),
                                         Example(state=LAUNCH_INPUT, answer=["广州"])]},
                           InferenceContext(REVIEW_INPUT))
        g = compiled(grammar([c], extract_mode="substring_chars"))
        self.assertTrue(accepts(g, wire([])))
        self.assertTrue(accepts(g, wire(["条目"])))  # Grammar only guarantees source, not semantics.
        self.assertFalse(accepts(g, wire(["广州"])))

    def test_macro_language_consumed_length_not_sam_longest(self):
        rng=random.Random(20261018)
        sources=['abbb','ababa','abcab','aaaa','', 'a']+[''.join(rng.choices('abc',k=rng.randint(1,7))) for _ in range(16)]
        for source in sources:
            for limit in (0,1,2,4,10):
                for empty in (False,True):
                    args=', '.join(json.dumps(c) for c in source)
                    g=xgr.Grammar.from_ebnf('root ::= Substring('+args+(', ' if args else '')+f'allow_empty={str(empty).lower()}, max_chunks={limit})')
                    copies=[g,xgr.Grammar.from_ebnf(str(g)),xgr.Grammar.deserialize_json(g.serialize_json())]
                    expected={source[i:j] for i in range(len(source)) for j in range(i+1,min(len(source),i+limit)+1)}
                    if empty:expected.add('')
                    for clone in copies:
                        c=compiled(clone)
                        for n in range(4):
                            for parts in itertools.product('abc',repeat=n):
                                text=''.join(parts)
                                self.assertEqual(accepts(c,text),text in expected,(source,limit,empty,text))
        for bad in ['-1','1.2','true','"2"','2147483648']:
            with self.assertRaises(Exception):xgr.Grammar.from_ebnf('root ::= Substring("a", max_chunks='+bad+')')

    def test_empty_chunks_consume_a_block_in_nullable_mode(self):
        for empty in (True,False):
            g=compiled('root ::= Substring("a", "", "b", allow_empty='+str(empty).lower()+', max_chunks=2)')
            self.assertEqual(accepts(g,'ab'),not empty)
            self.assertEqual(accepts(g,''),empty)
            self.assertTrue(accepts(g,'a'));self.assertTrue(accepts(g,'b'))

    def test_extract_json_unicode_count_and_boundaries(self):
        for source in ['甲"乙\\丙\n😀','e\u0301／＠ @','\x00\x1f\b\f\r\t','\\u4e2d','aaaa','']:
            for maximum in (1,2,4):
                c=compiled(grammar([call(source=source,min_items=0,max_items=2,max_length_per_item=maximum)],extract_mode='substring_chars'))
                self.assertTrue(accepts(c,wire([])))
                self.assertFalse(accepts(c,wire([''])))
                for i in range(len(source)):
                    for j in range(i+1,len(source)+1):
                        self.assertEqual(accepts(c,wire([source[i:j]])),True,(source,i,j))
                if source:
                    self.assertFalse(accepts(c,wire([source[0]]*3)))
                for text in ['不存在','甲丙',source[::-1]]:
                    if text not in source:self.assertFalse(accepts(c,wire([text])))
                for bad in [r'{"items":["\u00"]}',r'{"items":["\"]}','{"items":["\n"]}']:
                    self.assertFalse(accepts(c,bad))

    def test_list_and_raw_generate_codepoints(self):
        for lo,hi in ((0,0),(0,1),(1,1),(1,2),(2,3)):
            g=grammar([call('generate_list',min_items=1,max_items=2,min_length_per_item=lo,max_length_per_item=hi)])
            c=compiled(g)
            for text in ('','a','甲','😀','e\u0301','"','\\','\n','甲😀中'):
                for escaped in (False,True):
                    value=json.dumps({'items':[text]},ensure_ascii=escaped,separators=(',',':'))
                    self.assertEqual(accepts(c,value),bool(text) or lo==0,(lo,hi,value))
            self.assertFalse(accepts(c,wire([])))
            self.assertFalse(accepts(c,wire(['a','a','a'])))
            raw=compiled(grammar([call('generate',min_length=lo,max_length=hi)]))
            for text in ('','a','甲','😀','e\u0301','\n','甲😀中'):
                self.assertEqual(accepts(raw,text),bool(text) or lo==0,(lo,hi,text))
        zero=compiled(grammar([call('generate_list',max_items=0)]))
        self.assertTrue(accepts(zero,wire([])));self.assertFalse(accepts(zero,wire([''])))

    def test_mixed_same_source_different_limits_and_isolation(self):
        cs=[call(id='a',source='甲乙丙',max_items=1,max_length_per_item=1),
            call(id='b',source='甲乙丙',min_items=1,max_items=2,max_length_per_item=2),
            call(id='c',source='丁戊',max_items=1,max_length_per_item=1),
            call('generate','d',max_length=2),call('generate_list','e',max_items=1,max_length_per_item=1)]
        g=grammar(cs,batch=True,extract_mode='substring_chars');self.assertEqual(g.count('Substring('),3)
        c=compiled(g)
        def wire_batch(a='甲',b='甲乙',cc='丁',d='😀',e=None):
            return json.dumps({'answers':{'q1':[a],'q2':[b],'q3':[cc],'q4':d,'q5':e or ['甲']}},ensure_ascii=False,separators=(',',':'))
        self.assertTrue(accepts(c,wire_batch()))
        for kwargs in ({'a':'丁'},{'cc':'甲'},{'e':['甲','乙']},{'b':''}):
            self.assertFalse(accepts(c,wire_batch(**kwargs)),kwargs)

    def test_named_keys_preserve_source_and_value_constraints(self):
        name = ' 地点"\\\n😀 '
        cs = [call(id='internal-a', field_name=name, source='广州', min_items=1, max_items=1),
              call('generate', 'internal-b', field_name='event_title', min_length=1, max_length=60)]
        g = compiled(grammar(cs, batch=True, extract_mode='substring_chars'))
        def encoded(place='广州', title='上市'):
            # Key literals use ASCII escapes; substring values retain source bytes.
            return ('{"answers":{' + json.dumps(name) + ':' + json.dumps([place], ensure_ascii=False)
                    + ',"event_title":' + json.dumps(title, ensure_ascii=False) + '}}')
        good = encoded()
        self.assertTrue(accepts(g, good))
        self.assertFalse(accepts(g, encoded('北京')))
        self.assertFalse(accepts(g, encoded(title='')))
        self.assertFalse(accepts(g, good.replace('event_title', 'q2')))

    def test_nullable_generate_single_and_mixed_rule_cache_isolation(self):
        optional = call('generate', 'a', min_length=1, max_length=24, nullable=True)
        ordinary = call('generate', 'b', min_length=1, max_length=24)
        single = compiled(grammar([optional]))
        for raw in ('null', '"null"', '"中文"', '"\\\""'):
            self.assertTrue(accepts(single, raw), raw)
        for raw in ('[]', '1', 'true', '{}', '""', '', '未加引号'):
            self.assertFalse(accepts(single, raw), raw)
        for calls in ([optional, ordinary], [ordinary, optional]):
            mixed = compiled(grammar(calls, batch=True))
            for values in ((None, '标题'), ('null', '标题'), ('标题', None), ('标题', ''), ('标题', '标题')):
                expected = all((v is None and c.arguments.get('nullable', False)) or
                               (isinstance(v, str) and bool(v)) for c, v in zip(calls, values))
                raw = json.dumps({'answers': dict(zip(('q1', 'q2'), values))}, ensure_ascii=False, separators=(',', ':'))
                self.assertEqual(accepts(mixed, raw), expected, (calls, values))

if __name__=='__main__':unittest.main()
