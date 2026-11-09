"""P0 wire, preflight, immutable release, and local error regressions; no model calls."""
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch, Mock
import httpx
from annoscript.backends import InferenceContext, InferenceError, BackendFatalError
from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from annoscript.backends.base import normalize_call
from annoscript.backends._generation_policy import SAMPLING, PinnedTokenCounter
from annoscript.server.releases import manifest


def policy(window=8192):
    return {'sampling':dict(SAMPLING),'default_max_tokens':1024,'context_window':window,
            'bounded_outputs':True,'tokenizer':{'revision':'fixture-revision','files':
            {k:'0'*64 for k in ('tokenizer.json','tokenizer_config.json','chat_template.jinja')}}}


def call(ctx, op='generate', id='x', **kwargs):
    defaults = {'max_length':512} if op=='generate' else {'max_items':8,'max_length_per_item':58}
    return normalize_call(id,op,('任务',),{**defaults, **kwargs},ctx)


class GenerationPolicyTests(unittest.IsolatedAsyncioTestCase):
    def backend(self, handler, *, counter=lambda payload:10, window=8192, diagnostics=None):
        return LLMAPIBackend('https://test.invalid/v1','fixture','qwen',profile='qwen_vllm',
            extract_mode='substring_chars',request_policy=policy(window),_token_counter=counter,
            transport=httpx.MockTransport(handler),diagnostic_observer=diagnostics)

    async def test_single_and_repair_sampling_budget_and_bounds(self):
        ctx=InferenceContext('甲乙','共享策略')
        for op, text in [('generate','甲'),('extract','{"items":["甲"]}'),('generate_list','{"items":["甲"]}')]:
            for budget in ((None,) if op=="extract" else (None,160,600)):
                for repair in (None,'bad format'):
                    sent=[]
                    def handle(req):
                        sent.append(json.loads(req.content))
                        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':text}}]})
                    b=self.backend(handle)
                    kwargs={} if op=='extract' else {'max_num_tokens':budget}
                    kwargs.update({'max_length':2} if op=='generate' else {'max_items':2,'max_length_per_item':2})
                    try:
                        await b.infer(ctx,call(ctx,op,**kwargs),repair_feedback=repair,max_tokens_override=1024 if repair else None)
                    finally: await b.aclose()
                    self.assertEqual(len(sent),1);p=sent[0]
                    self.assertEqual(p['max_tokens'],budget or (4 if op=='generate' else 56))
                    self.assertEqual({k:p[k] for k in SAMPLING},SAMPLING)
                    self.assertEqual(p['chat_template_kwargs'],{'enable_thinking':False})
                    self.assertIn('grammar',p['structured_outputs'])

    async def test_context_exact_boundary_and_one_over_zero_http(self):
        ctx=InferenceContext('甲');sent=[];events=[]
        def handle(req):
            sent.append(req);return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'甲'}}]})
        for input_tokens in (7168,7169):
            b=self.backend(handle,counter=lambda p:input_tokens,diagnostics=events.append)
            try:
                if input_tokens==7168: await b.infer(ctx,call(ctx))
                else:
                    with self.assertRaises(InferenceError) as cm: await b.infer(ctx,call(ctx),repair_feedback='bad')
                    self.assertEqual(cm.exception.code,'context_length_exceeded')
                    self.assertEqual(cm.exception.metadata['dispatch_status'],'not_sent')
            finally: await b.aclose()
        self.assertEqual(len(sent),1)
        local=[e for e in events if e['event']=='local_rejection']
        self.assertEqual(len(local),1);self.assertNotIn('request_id',local[0]);self.assertTrue(local[0]['repair'])

    async def test_batch_prefix_default_and_infer_preflight(self):
        ctx=InferenceContext('甲乙');sent=[]
        b=self.backend(lambda req:sent.append(req),counter=lambda p:7900)
        calls=[call(ctx,'extract',str(i),max_items=2,max_length_per_item=2) for i in range(4)]
        try:
            # Two calls fit: 7900 + 240. Three do not: 7900 + 328.
            plan=b.plan_batch(ctx,calls,max_calls=8,max_input_bytes=1000000,max_output_tokens=4096,default_tokens=999)
            self.assertEqual(len(plan.calls),2);self.assertEqual(plan.max_tokens,240)
            with self.assertRaises(InferenceError): await b.infer_batch(ctx,calls,max_tokens=4096)
            self.assertFalse(sent)
            p=b._batch_payload(ctx,plan.calls,plan.max_tokens)
            self.assertEqual({k:p[k] for k in SAMPLING},SAMPLING)
        finally: await b.aclose()

    async def test_unsatisfiable_local_and_planner_fallback(self):
        ctx=InferenceContext('');sent=[];events=[]
        b=self.backend(lambda req:sent.append(req),diagnostics=events.append)
        impossible=call(ctx,'extract',min_items=1)
        try:
            self.assertIsNone(b.plan_batch(ctx,[impossible]*2,max_calls=8,max_input_bytes=100000,max_output_tokens=4096,default_tokens=1024))
            with self.assertRaises(InferenceError) as cm: await b.infer(ctx,impossible)
            self.assertEqual(cm.exception.code,'unsatisfiable_constraints');self.assertFalse(sent)
            self.assertEqual(events[0]['dispatch_status'],'not_sent')
        finally: await b.aclose()

    async def test_provider_ignores_bounds_still_fails(self):
        ctx=InferenceContext('甲乙丙')
        for op, value, kwargs in [('generate','',{'min_length':1,'max_length':2}),
                                 ('generate_list','{"items":["甲","乙"]}',{'max_items':1}),
                                 ('extract','{"items":["不在原文"]}',{'max_length_per_item':2})]:
            b=self.backend(lambda req:httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':value}}]}))
            try:
                with self.assertRaises(InferenceError): await b.infer(ctx,call(ctx,op,**kwargs))
            finally: await b.aclose()

    async def test_diagnostic_failure_never_sends_or_retries(self):
        ctx=InferenceContext('甲');sent=[]
        def broken(event):raise OSError('fixture')
        b=self.backend(lambda req:sent.append(req),counter=lambda p:8192,diagnostics=broken)
        try:
            with self.assertRaises(BackendFatalError):await b.infer(ctx,call(ctx))
            self.assertFalse(sent)
        finally:await b.aclose()

    def test_counter_requires_ids_not_transformers_default_mapping(self):
        tok=Mock();tok.apply_chat_template.return_value=[1,2,3]
        with patch.dict(os.environ,ANNO_GENERATE_TOKENIZER_DIR='fixture'), patch('annoscript.backends._generation_policy._load_tokenizer',return_value=tok):
            counter=PinnedTokenCounter(policy()['tokenizer'])
            self.assertEqual(counter({'messages':[]}),3)
            self.assertIs(tok.apply_chat_template.call_args.kwargs['return_dict'],False)
            tok.apply_chat_template.return_value={'input_ids':[1,2,3], 'attention_mask':[1,1,1]}
            with self.assertRaises(BackendFatalError):counter({'messages':[]})

    def test_config_digest_validation_and_release_binding(self):
        cfg={'decision':{'base_url':'https://d.invalid','model':'d','revision':'d'},
             'generation':{'base_url':'https://q.invalid','model':'q','revision':'fixture-revision','profile':'qwen_vllm',
                           'structured_output_revision':'fixture-decoder','request_policy':policy()}}
        self.assertEqual(manifest(cfg)['generation']['request_policy'],policy())
        for mutate in (lambda x:x['generation'].update(revision='other'),
                       lambda x:x['generation'].update(assumed_avg_num_tokens_per_char=0),
                       lambda x:x['generation']['request_policy']['sampling'].update(temperature=1)):
            altered=copy.deepcopy(cfg);mutate(altered)
            with self.assertRaises(ValueError):manifest(altered)
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,ANNO_GENERATE_TOKENIZER_DIR=root):
            with self.assertRaises(BackendFatalError):PinnedTokenCounter(policy()['tokenizer'])


# Exercise the same pause, accounting and cancellation integration fixtures through P0.
from annoscript.test_qwen_pilot_regression import ProtocolTests
from annoscript.test_diagnostics import WorkerTests as DiagnosticWorkerTests
from annoscript.backends import DecisionAPIBackend
from annoscript.server.worker import Worker
import asyncio

class P0ProtocolWorkerTests(ProtocolTests):
    def run_http(self, handler, *, observer=None):
        def factory(_):
            return (DecisionAPIBackend('https://offline.invalid','fixture','model',transport=httpx.MockTransport(handler)),
                    LLMAPIBackend('https://offline.invalid','fixture','model',profile='qwen_vllm',extract_mode='substring_chars',request_policy=policy(),_token_counter=lambda p:10,transport=httpx.MockTransport(handler)))
        return Worker(self.store,self.releases,backend_factory=factory,concurrency=4,request_observer=observer)

class P0DiagnosticWorkerTests(DiagnosticWorkerTests):
    def worker(self, handler, **kw):
        def factory(_):
            return (DecisionAPIBackend('https://offline.invalid','fixture','model',transport=httpx.MockTransport(handler)),
                    LLMAPIBackend('https://offline.invalid','fixture','model',profile='qwen_vllm',extract_mode='substring_chars',request_policy=policy(),_token_counter=lambda p:10,transport=httpx.MockTransport(handler)))
        return Worker(self.store,self.releases,backend_factory=factory,**kw)

    def test_local_context_failure_continues_next_batch_without_retry(self):
        from annoscript.test_jobs_server import payload
        sent=[];events=[]
        def handle(req):
            sent.append(req);return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'yes'}}]})
        def count(body):return 8192 if body['messages'][-1]['content']=='too long' else 10
        def factory(_):
            return (DecisionAPIBackend('https://offline.invalid','fixture','model',transport=httpx.MockTransport(handle)),
                    LLMAPIBackend('https://offline.invalid','fixture','model',profile='qwen_vllm',request_policy=policy(),_token_counter=count,transport=httpx.MockTransport(handle)))
        worker=Worker(self.store,self.releases,backend_factory=factory,diagnostic_observer=events.append)
        for batch in range(2):
            states=tuple('too long' if batch==0 and i==0 else str(i) for i in range(20))
            job=self.store.submit('alice','p0-local-'+str(batch),payload(states),self.releases)
            for _ in range(20):asyncio.run(worker.execute(self.store.claim('w',self.releases)))
            status=self.store.status('alice',job['job_id'])
            self.assertEqual(status['state'],'completed');self.assertEqual(status['counts']['error'],1 if batch==0 else 0)
            self.assertIsNone(self.store.claim('w',self.releases))
        self.assertEqual(len(sent),39)
        local=[e for e in events if e['event']=='local_rejection'];self.assertEqual(len(local),1)
        self.assertIn('job_id',local[0]);self.assertIn('index',local[0])

if __name__=="__main__":unittest.main()
