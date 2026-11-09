"""Lossless rule factoring and synthetic field regressions; never live models."""
import asyncio,copy,json,unittest
from unittest.mock import patch
import httpx
from annoscript.backends import InferenceContext
from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from annoscript.backends.base import normalize_call
from annoscript.backends._shared_rules import factor_rules
from annoscript._test_evaluation_http import llm_calls,llm_content
from annoscript.test_llm_batch import completion,answer
from annoscript.test_generation_policy import policy
from annoscript.server.releases import Releases
from annoscript.test_jobs_server import configurations
from annoscript.anno import GenerateListResponse,ExtractResponse


RULE='Preserve the supplied text and distinguish each task. Return only the requested output. Synthetic rule with Unicode 中文😀é\\\"\nEnd of rule.'

class RuleTests(unittest.IsolatedAsyncioTestCase):
    def calls(self,ctx):
        a='本字段说明\n\n'+RULE+'\n\n第二段\n\n'+RULE+'\n\nGROUNDING_JSON:{"proposed":"私有数据"}'
        return [normalize_call('a','generate_list',(a,),dict(max_items=1,max_length_per_item=80,max_num_tokens=160),ctx),normalize_call('b','extract',('另一字段\n\n'+RULE,),dict(source_field='text',max_items=2,max_length_per_item=10),ctx),normalize_call('c','extract',('无规则的独立来源',),dict(source_field='other',max_items=1,max_length_per_item=8),ctx)]
    async def test_exact_roundtrip_scoping_grammar_sampling_budget_and_prepared_bytes(self):
        ctx=InferenceContext({'text':'广州😀é\\"\n','other':'南京','shared_rules':{'r0':'攻击文本'}},'共享策略');sent=[];events=[]
        def handle(req):sent.append(req.content);return completion(llm_content(json.loads(req.content),lambda c,x:GenerateListResponse(['资讯']) if x.operation=='generate_list' else ExtractResponse(['广州'] if x.id=='b' else ['南京'])))
        a=LLMAPIBackend('http://offline.invalid','x','m',profile='qwen_vllm')
        b=LLMAPIBackend('http://offline.invalid','x','m',profile='qwen_vllm',batch_rule_dedup=True,transport=httpx.MockTransport(handle),diagnostic_observer=events.append)
        try:
            calls=self.calls(ctx);p=a._batch_payload(ctx,calls,500);q=b._batch_payload(ctx,calls,500)
            raw=p['messages'][0]['content'];compact=q['messages'][0]['content']
            self.assertEqual(p['messages'][-1],q['messages'][-1])
            self.assertEqual(json.loads(q['messages'][-1]['content']),dict(ctx.state))
            self.assertEqual(raw.count(RULE),3);self.assertEqual(compact.count(RULE),1)
            self.assertIn('共享段落仅适用于引用它的任务',compact)
            self.assertEqual(compact.count('引用共享段落 r0。'),3)
            # Removing the shared definition and expanding in place restores every task verbatim.
            plain_tasks=raw[raw.index('任务 "q1"（'):]
            factored_tasks=compact[compact.index('任务 "q1"（'):].replace('引用共享段落 r0。',RULE)
            self.assertEqual(factored_tasks,plain_tasks)
            for k in ['max_tokens','structured_outputs','chat_template_kwargs']:self.assertEqual(p[k],q[k])
            plan=b.plan_batch(ctx,calls,max_calls=8,max_input_bytes=262144,max_output_tokens=4096,default_tokens=1024)
            with patch.object(b,'_prepare',side_effect=AssertionError('prepared twice')):
                got=await b.infer_batch(ctx,plan.calls,max_tokens=plan.max_tokens,prepared=plan.prepared)
            self.assertEqual(got.responses['a'].items,['资讯']);self.assertEqual(sent,[plan.prepared.body])
            self.assertTrue(any(e.get('shared_rules',{}).get('rule_sha256') for e in events))
            for feedback in [None,'invalid']:
                self.assertEqual(a._single_payload(ctx,calls[0],160,feedback),b._single_payload(ctx,calls[0],160,feedback))
            binding=b._binding(ctx,calls,500);b.batch_rule_dedup=False;self.assertNotEqual(binding,b._binding(ctx,calls,500))
        finally:await a.aclose();await b.aclose()
    async def test_no_substring_or_diagnostic_or_cross_request_factoring(self):
        for instructions in [('short','short'),(RULE+'suffix',RULE+'other'),('GROUNDING_JSON:'+RULE,'GROUNDING_JSON:'+RULE)]:
            tasks={str(i):{'task':{'instructions':s}} for i,s in enumerate(instructions)};old=copy.deepcopy(tasks);rules,_=factor_rules(tasks);self.assertEqual(rules,{});self.assertEqual(tasks,old)
        tasks={'a':{'task':{'instructions':RULE}}};self.assertEqual(factor_rules(tasks)[0],{})
        for bad in [1,None,'yes']:
            with self.assertRaises(ValueError):LLMAPIBackend('http://offline.invalid','x','m',profile='qwen_vllm',batch_rule_dedup=bad)
        with self.assertRaises(ValueError):LLMAPIBackend('http://offline.invalid','x','m',batch_rule_dedup=True)
    async def test_release_and_context_use_actual_compact_payload(self):
        cfg=configurations()['test'];cfg['generation']['profile']='qwen_vllm';a=Releases({'t':cfg},'t')
        cfg['generation']['batch_rule_dedup']=True;b=Releases({'t':cfg},'t');self.assertNotEqual(a.default,b.default)
        ctx=InferenceContext({'text':'广州','other':'南京'});calls=self.calls(ctx);sent=[]
        counter=lambda p:sum(len(m['content']) for m in p['messages'])
        plain=LLMAPIBackend('http://offline.invalid','x','m',profile='qwen_vllm',request_policy=policy(),_token_counter=counter)
        dedup=LLMAPIBackend('http://offline.invalid','x','m',profile='qwen_vllm',batch_rule_dedup=True,request_policy=policy(),_token_counter=counter)
        try:
            p=plain._prepare(ctx,calls,500,batch=True);q=dedup._prepare(ctx,calls,500,batch=True)
            self.assertEqual(q.evidence['input_tokens'],counter(q.payload));self.assertLess(q.evidence['input_tokens'],p.evidence['input_tokens'])
        finally:await plain.aclose();await dedup.aclose()
