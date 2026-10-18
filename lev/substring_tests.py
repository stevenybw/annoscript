"""Acceptance against pinned XGrammar with the nonempty extension."""
import itertools
import json
import random
import unittest
from types import SimpleNamespace
from importlib.metadata import version
import xgrammar as xgr
from lev.backends._grammar import extract_grammar, batch_grammar


def compiler():
    return xgr.GrammarCompiler(xgr.TokenizerInfo([]), cache_enabled=False)


def accepts(compiled, text):
    m=xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    return m.accept_string(text) and m.is_terminated()


def wire(items):
    return json.dumps({'items':items},ensure_ascii=False,separators=(',',':'))


class SubstringGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert version('xgrammar') in ('0.2.5+lev.substring1','0.2.5+lev.substring2','0.2.5+lev.substring2.1','0.2.5+lev.substring2.2','0.2.5+lev.substring2.3')

    def test_nonempty_macro_and_serialization(self):
        for chunks in [('a','b'),('a','','b'),('','',''),()]:
            for empty in (True,False):
                args=', '.join(json.dumps(c) for c in chunks)
                g=xgr.Grammar.from_ebnf('root ::= Substring('+args+(', ' if args else '')+
                                      'allow_empty='+str(empty).lower()+')')
                for copy in [g,xgr.Grammar.from_ebnf(str(g)),xgr.Grammar.deserialize_json(g.serialize_json())]:
                    c=compiler().compile_grammar(copy)
                    self.assertEqual(accepts(c,''),empty)
                    for v in ['a','b','ab','ba','x']:
                        self.assertEqual(accepts(c,v), v in {''.join(chunks[i:j]) for i in range(len(chunks)) for j in range(i+1,len(chunks)+1)} and bool(v))

    def test_json_escaping_exhaustive_substrings(self):
        for source in ['甲"乙\\丙\n😀','\x00\x1f\b\f\r\t','e\u0301／＠ @','\\u4e2d','aaaa','']:
            c=compiler().compile_grammar(extract_grammar(source))
            self.assertTrue(accepts(c,wire([])))
            self.assertFalse(accepts(c,wire([''])))
            for i in range(len(source)):
                for j in range(i+1,len(source)+1):
                    self.assertTrue(accepts(c,wire([source[i:j]])),repr((source,i,j)))
            for bad in ['不存在',source+'不存在',source[::2]]:
                if bad and bad not in source:self.assertFalse(accepts(c,wire([bad])))
            for bad in ['{"items":["\\"]}',r'{"items":["\u00"]}','{"items":["\n"]}']:
                self.assertFalse(accepts(c,bad))

    def test_random_short_language(self):
        rng=random.Random(20261016)
        alphabet=['甲','乙','"','\\','\n','😀']
        for _ in range(30):
            source=''.join(rng.choices(alphabet,k=rng.randrange(1,7)))
            c=compiler().compile_grammar(extract_grammar(source))
            for n in range(3):
                for chars in itertools.product(alphabet,repeat=n):
                    value=''.join(chars)
                    self.assertEqual(accepts(c,wire([value])), bool(value) and value in source)

    def test_batch_isolation_and_other_operations(self):
        calls=[SimpleNamespace(id=i,operation=op,arguments={'source':source}) for i,op,source in
               [('a','extract','甲乙'),('b','extract','丙丁'),('c','extract','甲乙'),
                ('d','generate',''),('e','generate_list','')]]
        c=compiler().compile_grammar(batch_grammar(calls,extract_mode='substring_chars'))
        answers={'a':{'type':'extract','items':['甲']},'b':{'type':'extract','items':['丁']},
                 'c':{'type':'extract','items':['乙']},'d':{'type':'generate','text':'任意改写'},
                 'e':{'type':'generate_list','items':['','任意改写']}}
        def encode():return json.dumps({'answers':answers},ensure_ascii=False,separators=(',',':'))
        self.assertTrue(accepts(c,encode()))
        for bad in ['丁','甲丁','']:
            answers['a']['items']=[bad];self.assertFalse(accepts(c,encode()))
        answers['a']={'type':'generate','text':'甲'};self.assertFalse(accepts(c,encode()))


if __name__=='__main__':unittest.main()
