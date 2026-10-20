"""Wire compatibility and source binding without model calls."""
import json
import unittest
import httpx
from lev.backends import InferenceContext, LLMAPIBackend, InferenceError
from lev.backends.base import normalize_call
from lev.backends._grammar import ITEMS_GRAMMAR, extract_grammar, batch_grammar
from lev.backends._bounded_grammar import grammar
from lev.server.releases import manifest


class SubstringBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_repair_and_default_wire_equivalence(self):
        ctx = InferenceContext('甲"乙\\丙\n😀', '共享规则')
        call = normalize_call('x', 'extract', ('抽取',), {'max_items': 8, 'max_length_per_item': 58}, ctx)
        for repair in (None, 'invalid item'):
            bodies = []
            def handle(request):
                bodies.append(json.loads(request.content))
                return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                    'message': {'content': json.dumps({'items': ['甲"乙']}, ensure_ascii=False)}}]})
            for mode in ('validated', 'substring_chars'):
                b = LLMAPIBackend('https://test.invalid/v1', 'fixture', 'qwen', profile='qwen_vllm',
                                  extract_mode=mode, transport=httpx.MockTransport(handle))
                try:
                    result = await b.infer(ctx, call, repair_feedback=repair, max_tokens_override=100)
                    self.assertEqual(result.response.items, ['甲"乙'])
                finally:
                    await b.aclose()
            self.assertEqual(bodies[0].pop('structured_outputs'), {'grammar': grammar([call])})
            self.assertEqual(bodies[1].pop('structured_outputs'), {'grammar': grammar([call],extract_mode='substring_chars')})
            self.assertEqual(bodies[0], bodies[1])

    async def test_provider_ignores_grammar_runtime_still_rejects(self):
        ctx = InferenceContext('甲乙')
        call = normalize_call('x', 'extract', ('抽取',), {'max_items': 8, 'max_length_per_item': 58}, ctx)
        for text in ('', '丙', '乙甲'):
            calls = []
            def handle(request):
                calls.append(request)
                return httpx.Response(200, json={'choices': [{'finish_reason':'stop',
                    'message': {'content': json.dumps({'items':[text]})}}]})
            b = LLMAPIBackend('https://test.invalid', 'fixture', 'qwen', profile='qwen_vllm',
                              extract_mode='substring_chars', transport=httpx.MockTransport(handle))
            try:
                with self.assertRaises(InferenceError): await b.infer(ctx, call)
                self.assertEqual(len(calls), 1)
            finally: await b.aclose()

    async def test_configuration(self):
        for profile, mode in [('legacy','substring_chars'), ('qwen_vllm','unknown')]:
            with self.assertRaises(ValueError):
                LLMAPIBackend('https://test.invalid','fixture','qwen',profile=profile,extract_mode=mode)
        cfg={'decision':{'base_url':'https://d.invalid','model':'d','revision':'d1'},
             'generation':{'base_url':'https://g.invalid','model':'g','revision':'g1','profile':'qwen_vllm'}}
        old=manifest(cfg)
        self.assertNotIn('extract_mode', old['generation'])
        cfg['generation']['extract_mode']='substring_chars'
        with self.assertRaises(ValueError): manifest(cfg)
        cfg['generation']['structured_output_revision']='pinned-build'
        self.assertEqual(manifest(cfg)['generation']['extract_mode'],'substring_chars')

    async def test_mixed_batch_payload_and_source_deduplication(self):
        ctx=InferenceContext({'unrelated':'不能抽取','甲乙':'甲乙','丙丁':'丙丁'})
        calls=[normalize_call(str(i),op,('任务',),({'source_field':src,'max_items':8,'max_length_per_item':58} if op=='extract' else {'max_length':512} if op=='generate' else {'max_items':8,'max_length_per_item':58}),ctx)
               for i,(op,src) in enumerate([('extract','甲乙'),('extract','丙丁'),
                                            ('extract','甲乙'),('generate',None),('generate_list',None)])]
        a=LLMAPIBackend('https://test.invalid','fixture','qwen',profile='qwen_vllm')
        b=LLMAPIBackend('https://test.invalid','fixture','qwen',profile='qwen_vllm',extract_mode='substring_chars')
        try:
            pa,pb=a._batch_payload(ctx,calls,4096),b._batch_payload(ctx,calls,4096)
            ga=pa.pop('structured_outputs')['grammar'];gb=pb.pop('structured_outputs')['grammar']
            self.assertEqual(pa,pb)
            self.assertEqual(ga,grammar(calls,batch=True))
            self.assertEqual(gb.count(' ::= Substring('),2)
            self.assertNotIn('不能抽取',gb)
            self.assertEqual(gb.count('allow_empty=false'),2)
        finally: await a.aclose(); await b.aclose()


if __name__ == '__main__': unittest.main()
