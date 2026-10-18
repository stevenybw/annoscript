"""Mandatory suggestions, hard contracts, budget sharing and inference-free rejection."""
import asyncio
import copy
import json
import math
import unittest
import httpx
from lev.backends import InferenceContext, LLMAPIBackend, InferenceError
from lev.backends.base import normalize_call, validate_response
from lev.backends._generation_policy import effective_budget
from lev.backends._bounded_grammar import grammar
from lev.compiler import compile_program, CompileError
from lev.plan_lowering import lower_to_plan
from lev.lev import GenerateResponse, GenerateListResponse, ExtractResponse
from lev.test_generation_policy import policy


def missing_generate(lev):
    return lev.generate('x').text

def missing_list(lev):
    return lev.generate_list('x', max_items=1).items

def missing_extract(lev):
    return lev.extract('x', max_length_per_item=1).items

def oversized(lev):
    a = lev.generate('a', min_length=4, max_length=5).text
    b = lev.generate_list('b', max_items=1, min_length_per_item=2, max_length_per_item=3).items
    return [a,b]


def call(op, **kw):
    return normalize_call('x',op,('task',),kw,InferenceContext('甲乙丙丁'))

class AdvisoryTests(unittest.TestCase):
    def test_mandatory_at_compile_and_dynamic_normalization(self):
        for f in (missing_generate,missing_list,missing_extract):
            with self.assertRaises(CompileError):compile_program(f)
        for op, valid in [('generate',{'max_length':5}),('generate_list',{'max_items':2,'max_length_per_item':5}),('extract',{'max_items':2,'max_length_per_item':5})]:
            for field in valid:
                for value in (None,False,-1,1.2,float('inf')):
                    with self.subTest(op=op,field=field,value=value),self.assertRaises((ValueError,TypeError)):
                        call(op,**{**valid,field:value})
                opts=dict(valid);del opts[field]
                with self.assertRaises(TypeError):call(op,**opts)
        with self.assertRaises(ValueError):call('generate',min_length=6,max_length=5)

    def test_budgets_ratio_explicit_priority_zero_and_repair(self):
        for bad in (0,-1,True,float('inf'),float('nan'),'2',None):
            with self.assertRaises(ValueError):effective_budget(call('generate',max_length=10),bad)
        self.assertEqual(effective_budget(call('generate',max_length=3),1.5),5)
        self.assertEqual(effective_budget(call('generate',max_length=0)),1)
        self.assertEqual(effective_budget(call('extract',max_items=0,max_length_per_item=1)),32)
        self.assertEqual(effective_budget(call('extract',max_items=16,max_length_per_item=300)),9760)
        for op, kw in [('generate',{'max_length':80}),('generate_list',{'max_items':16,'max_length_per_item':16})]:
            c=call(op,**kw,max_num_tokens=7)
            self.assertEqual(effective_budget(c,100),7)
            self.assertEqual(effective_budget(c,100,999),7)
            self.assertEqual(effective_budget(c,100,3),3)
        self.assertEqual(effective_budget(call('generate_list',max_items=16,max_length_per_item=16)),672)
        self.assertEqual(effective_budget(call('extract',max_items=4,max_length_per_item=120)),1024)

    def test_suggestions_and_hard_guarantees(self):
        for c,r in [(call('generate',min_length=3,max_length=4),GenerateResponse('甲')),
                    (call('generate',min_length=3,max_length=4),GenerateResponse('甲'*100)),
                    (call('generate_list',min_length_per_item=3,max_length_per_item=4,max_items=1),GenerateListResponse(['甲'*100])),
                    (call('extract',max_items=1,max_length_per_item=1),ExtractResponse(['甲乙丙丁']))]:
            validate_response(c,r)
        for c,r in [(call('generate',min_length=3,max_length=4),GenerateResponse('')),
                    (call('generate_list',min_length_per_item=3,max_length_per_item=4,max_items=1),GenerateListResponse([''])),
                    (call('generate_list',max_length_per_item=4,max_items=1),GenerateListResponse(['甲','乙'])),
                    (call('extract',max_items=1,max_length_per_item=1),ExtractResponse(['甲丙']))]:
            with self.assertRaises(InferenceError):validate_response(c,r)

    def test_grammar_ignores_suggestions_keeps_source_and_count(self):
        c=call('extract',max_items=1,max_length_per_item=1)
        g=grammar([c],extract_mode='substring_chars')
        self.assertNotIn('max_chunks',g);self.assertIn('allow_empty=false',g)
        self.assertEqual(g,grammar([call('extract',max_items=1,max_length_per_item=100)],extract_mode='substring_chars'))
        self.assertIn('Regex(',grammar([call('generate_list',max_items=2,max_length_per_item=3)]))
        self.assertNotIn('json-char',grammar([call('generate_list',max_items=2,max_length_per_item=3)]))

class AdvisoryWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_over_suggestion_no_repair_runtime_and_metadata(self):
        from lev.runtime import LevRuntime,RunInput
        from lev.test_runtime import FakeBackend
        sent=[]
        def handler(req):
            body=json.loads(req.content);sent.append(body)
            tasks=json.loads(body['messages'][-1]['content'])['tasks']
            answers={k:({'type':'generate','text':'甲'*100} if v['operation']=='generate' else {'type':'generate_list','items':['乙'*100]}) for k,v in tasks.items()}
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':answers})}}]})
        b=LLMAPIBackend('https://offline.invalid','fixture','q',profile='qwen_vllm',request_policy=policy(),_token_counter=lambda p:1,transport=httpx.MockTransport(handler))
        rt=LevRuntime(FakeBackend(),b)
        try:
            out=[r async for r in (await rt.submit(lower_to_plan(compile_program(oversized)),[RunInput(0,'source')]))]
            self.assertEqual(out[0].status,'success');self.assertEqual(len(sent),1)
            self.assertEqual(sent[0]['max_tokens'],64+10+32+(6+32+8)+32)
            self.assertIn('suggestions',sent[0]['messages'][0]['content'])
        finally:await rt.aclose()

    async def test_direct_invalid_args_and_context_no_http(self):
        sent=[];events=[]
        b=LLMAPIBackend('https://offline.invalid','fixture','q',profile='qwen_vllm',request_policy=policy(),_token_counter=lambda p:8100,transport=httpx.MockTransport(lambda r:sent.append(r)),diagnostic_observer=events.append)
        try:
            with self.assertRaises(InferenceError):await b.infer(InferenceContext('甲'),call('generate',max_length=80))
            self.assertEqual(events[0]['output_budget'],160)
            self.assertEqual(events[0]['assumed_avg_num_tokens_per_char'],2)
            self.assertEqual(sent,[])
        finally:await b.aclose()

if __name__=='__main__':unittest.main()
