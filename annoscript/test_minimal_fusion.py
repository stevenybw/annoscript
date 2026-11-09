"""Minimal fusion regression: synthetic fixtures only; no live model calls."""
import json
import unittest
from copy import deepcopy
import httpx

from annoscript.backends import InferenceContext
from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from annoscript.backends.base import normalize_call
from annoscript._test_evaluation_http import llm_calls, llm_content
from annoscript.test_llm_batch import completion
from annoscript.anno import GenerateListResponse, ExtractResponse
from annoscript.server.releases import Releases
from annoscript.test_jobs_server import configurations



class LayoutBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_inline_single_batch_and_repair_preserve_sources_and_160_budget(self):
        ctx=InferenceContext({'text':'广州😀é\\"\n'},'固定策略')
        calls=[normalize_call('name','generate_list',('名称',),dict(min_items=0,max_items=1,min_length_per_item=1,max_length_per_item=80,max_num_tokens=160),ctx),
               normalize_call('e','extract',('片段',),dict(source_field='text',max_items=2,max_length_per_item=20),ctx)]
        for profile in ('legacy','qwen_vllm'):
            sent=[];events=[]
            def handle(req):
                body=json.loads(req.content);sent.append(body)
                return completion(llm_content(body,lambda c,x: GenerateListResponse(['名称']) if x.operation=='generate_list' else ExtractResponse(['广州'])))
            b=LLMAPIBackend('http://offline.invalid','fixture','fixture',profile=profile,
                           transport=httpx.MockTransport(handle),diagnostic_observer=events.append)
            try:
                for c in calls: await b.infer(ctx,c)
                await b.infer(ctx,calls[0],repair_feedback='invalid',max_tokens_override=160)
                await b.infer(ctx,calls[1],repair_feedback='invalid')
                plan=b.plan_batch(ctx,calls,max_calls=8,max_input_bytes=262144,max_output_tokens=4096,default_tokens=1024)
                await b.infer_batch(ctx,plan.calls,max_tokens=plan.max_tokens,prepared=plan.prepared)
                self.assertEqual(sent[0]['max_tokens'],160);self.assertEqual(sent[2]['max_tokens'],160)
                self.assertEqual(plan.max_tokens,64+160+32+(2*2*20+32+8*2)+32)
                for body in (sent[1],sent[3],sent[4]):
                    payload=json.loads(body['messages'][-1]['content'])
                    self.assertNotIn('source_ref',json.dumps(payload))
                    self.assertEqual(json.dumps(body['messages'],ensure_ascii=False).count('广州'),1)
                    self.assertEqual(llm_calls(body)[1][-1][1].extraction_source,ctx.state['text'])
                self.assertTrue(all(e['source_layout']=='source_field_v1' for e in events if 'source_layout' in e))
            finally:await b.aclose()

    async def test_obsolete_layout_rejected_and_release_records_new_binding(self):
        for value in ('inline','state_refs_v1',False,'bad'):
            with self.assertRaisesRegex(TypeError, 'source_layout'):
                LLMAPIBackend('http://offline.invalid','fixture','m',source_layout=value)
            cfg=configurations()['test'];cfg['generation']['source_layout']=value
            with self.assertRaisesRegex(ValueError, 'source_layout was removed'):
                Releases({'test':cfg},'test')
        release=Releases(configurations(),'test')
        self.assertEqual(release.manifests[release.default]['engineering']['qwen_source_layout'],'source_field_v1')

if __name__=='__main__':unittest.main()

class WorkerFusionTests(unittest.TestCase):
    pass
