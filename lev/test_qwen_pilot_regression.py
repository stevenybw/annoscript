"""Offline regressions for Qwen layout, paused dispatch and uncertain outcomes."""
import asyncio
import json
import threading
import unittest
from unittest.mock import patch
import httpx
from lev.backends import InferenceContext, LLMAPIBackend, DecisionAPIBackend, BackendFatalError, InferenceError
from lev.backends.base import normalize_call
from lev.server.worker import Worker, classification
from lev.server.metrics import RequestMeter
from lev.server.releases import Releases
from lev.test_jobs_server import StoreTests, configurations, payload


def steps(lev):
    first = lev.generate('first', max_length=512).text
    return lev.generate(first, max_length=512).text


def completion(text='next'):
    return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{'content':text}}],
                                   'usage':{'prompt_tokens':7,'completion_tokens':2}})


class LayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_batch_repair_and_empty_policy_keep_contents_and_budgets(self):
        for policy in (None, '', '共同规则'):
            context = InferenceContext({'text':'广州'}, policy)
            calls = [normalize_call('a','extract',('地点',),{'source':'广州', 'max_items': 8, 'max_length_per_item': 58},context),
                     normalize_call('b','generate_list',('维度',),{'max_num_tokens':32, 'max_items': 8, 'max_length_per_item': 58},context)]
            for mode in ('single','repair','batch'):
                bodies = {}
                for profile in ('legacy','qwen_vllm'):
                    def handle(req):
                        body=json.loads(req.content);bodies[profile]=body
                        return completion(json.dumps({'answers':{'a':{'type':'extract','items':['广州']},'b':{'type':'generate_list','items':['空间']}}} if mode=='batch' else {'items':['广州']}))
                    backend=LLMAPIBackend('https://offline.invalid/v1','fixture','model',profile=profile,
                                          transport=httpx.MockTransport(handle))
                    try:
                        if mode=='batch':await backend.infer_batch(context,calls,max_tokens=2176)
                        else:await backend.infer(context,calls[0],**({'repair_feedback':'bad answer','max_tokens_override':128} if mode=='repair' else {}))
                    finally:await backend.aclose()
                legacy,qwen=bodies['legacy'],bodies['qwen_vllm']
                self.assertEqual([m['role'] for m in qwen['messages']],['system','user'])
                # The engineering source-reference envelope adds only its
                # resolution contract; business policy and logical arguments
                # remain unchanged across profiles.
                contract=legacy['messages'][0]['content']
                qcontract=qwen['messages'][0]['content']
                if policy is not None:
                    self.assertTrue(qcontract.endswith('\n\n'+policy))
                    qcontract=qcontract[:-(len(policy)+2)]
                if mode=='repair':
                    prefix,feedback=contract.split(' A previous batch answer',1)
                    self.assertTrue(qcontract.startswith(prefix))
                    self.assertTrue(qcontract.endswith(' A previous batch answer'+feedback))
                else:self.assertTrue(qcontract.startswith(contract))
                self.assertIn('source_ref',qcontract)
                from lev._test_evaluation_http import llm_calls
                a,b=llm_calls(legacy),llm_calls(qwen)
                self.assertEqual(a[0],b[0])
                self.assertEqual([(c.state,x.operation,x.arguments) for c,x in a[1]],
                                 [(c.state,x.operation,x.arguments) for c,x in b[1]])
                self.assertEqual(qwen.get('max_tokens'),legacy.get('max_tokens'))
                self.assertEqual(len(legacy['messages']),2 if policy is None else 3)


class ProtocolTests(StoreTests):
    # Inherited store tests remain a separate baseline; pytest collection here
    # also exercises them against the new policy defaults.
    def config(self, **policy):
        self.releases=Releases(configurations(**policy),'test')

    def run_http(self, handler, *, observer=None):
        def factory(_):
            return (DecisionAPIBackend('https://offline.invalid','fixture','model',transport=httpx.MockTransport(handler)),
                    LLMAPIBackend('https://offline.invalid/v1','fixture','model',profile='qwen_vllm',transport=httpx.MockTransport(handler)))
        return Worker(self.store,self.releases,backend_factory=factory,concurrency=4,request_observer=observer)

    def test_protocol_auth_context_and_retry_classification(self):
        for code,metadata,expected in [
            ('backend_protocol_error',{'http_status':400},'pause'),
            ('authentication_failed',{'http_status':401},'pause'),
            ('model_http_error',{'http_status':400},'result'),
            ('model_timeout',{},'pause'),('transport_error',{},'pause'),
            ('backend_unavailable',{},'retry'),
            ('model_http_error',{'http_status':429},'retry'),
            ('model_http_error',{'http_status':503},'retry')]:
            self.assertEqual(classification(code,metadata,retry_unknown_outcomes=False),expected)
        self.assertEqual(classification('model_timeout',{}),'retry')

    def test_four_inflight_workers_drain_after_protocol_pause_without_new_dispatch(self):
        self.config(retry_unknown_outcomes=False)
        job=self.store.submit('alice','protocol',payload(tuple(str(i) for i in range(8)),program=steps),self.releases)
        claims=[self.claim() for _ in range(4)];audit=[];calls=[]
        async def run():
            started=asyncio.Event();release=asyncio.Event()
            async def handler(req):
                body=json.loads(req.content);state=json.loads(body['messages'][-1]['content'])['state'];calls.append(state)
                if len(calls)==4:started.set()
                await started.wait()
                if state=='0':return httpx.Response(400,json={'error':{'message':'System message must be at the beginning.'}})
                await release.wait()
                return completion()
            worker=self.run_http(handler,observer=audit.append)
            tasks=[asyncio.create_task(worker.execute(c)) for c in claims]
            async with asyncio.timeout(10):
                while self.store.status('alice',job['job_id'])['state']!='paused':await asyncio.sleep(.005)
                self.assertIsNone(self.claim())
                self.assertTrue(all(self.store.renew(c) for c in claims[1:]))
                self.assertTrue(all(not self.store.can_dispatch(c) for c in claims[1:]))
                release.set();await asyncio.gather(*tasks)
        asyncio.run(run())
        self.assertEqual(len(calls),4)
        status=self.store.status('alice',job['job_id'])
        self.assertEqual(status['pause_reason']['code'],'backend_protocol_error')
        self.assertEqual(status['counts']['completed'],0)
        with self.store._tx() as db:
            rows=db.execute('SELECT http_status,seconds,usage FROM http_requests').fetchall()
        self.assertEqual(sorted(r['http_status'] for r in rows),[200,200,200,400])
        self.assertTrue(all(r['seconds'] is not None for r in rows))
        self.assertEqual(len([x for x in audit if x['event']=='outcome']),4)
        self.assertTrue(all(x['storage_committed'] for x in audit if x['event']=='outcome'))

    def test_context_error_is_local_and_unknown_timeout_pauses_once(self):
        for name,unknown in [('context',False),('timeout',True)]:
            self.config(retry_unknown_outcomes=False)
            job=self.submit((name,),key=name);calls=[]
            async def handler(req):
                calls.append(req)
                if unknown:raise httpx.ReadTimeout('fixture')
                return httpx.Response(400,json={'error':{'message':'maximum context length is 8192 tokens'}})
            asyncio.run(self.run_http(handler).execute(self.claim()))
            self.assertEqual(len(calls),1)
            status=self.store.status('alice',job['job_id'])
            self.assertEqual(status['state'],'paused' if unknown else 'completed')
            if unknown:self.assertIsNone(self.claim())
            else:self.assertEqual(self.store.results('alice',job['job_id'])['results'][0]['error']['code'],'model_http_error')

    def test_pause_gate_at_meter_boundary_preserves_existing_response(self):
        job=self.submit(('one','two'));a,b=self.claim(),self.claim()
        self.store.request_started(b,'inflight','generation',1)
        self.store.attempt_failed(a,{'code':'backend_protocol_error','message':'fixture'},pause=True)
        self.assertFalse(self.store.request_started(b,'blocked','generation',1))
        self.assertTrue(self.store.request_finished(b,'inflight',seconds=1,http_status=200,usage={'prompt_tokens':7}))
        async def run():
            with self.assertRaises(BackendFatalError) as e:await RequestMeter(self.store,b,'generation').start(1)
            self.assertEqual(e.exception.code,'job_paused')
        asyncio.run(run())

    def test_observer_failure_never_retries_model(self):
        for phase,expected in [('intent',0),('outcome',1)]:
            self.config(retry_unknown_outcomes=False)
            job=self.submit((phase,),key=phase);calls=[]
            def observe(event):
                if event['event']==phase:raise OSError('fixture storage unavailable')
            def handler(req):calls.append(req);return completion()
            asyncio.run(self.run_http(handler,observer=observe).execute(self.claim()))
            self.assertEqual(len(calls),expected)
            self.assertEqual(self.store.status('alice',job['job_id'])['pause_reason']['code'],'metrics_failed')
            self.assertIsNone(self.claim())

    def test_cancel_during_commit_still_audits_observed_outcome_once(self):
        self.submit(('a',));claim=self.claim();audit=[];meter=RequestMeter(self.store,claim,'generation',observer=audit.append)
        entered=threading.Event();release=threading.Event();original=self.store.request_finished
        def finish(*args,**kwargs):entered.set();release.wait(5);return original(*args,**kwargs)
        async def run():
            identity=await meter.start(1)
            with patch.object(self.store,'request_finished',side_effect=finish):
                task=asyncio.create_task(meter.finish(identity,.1,200,{'prompt_tokens':5,'completion_tokens':1}))
                async with asyncio.timeout(5):
                    while not entered.is_set():await asyncio.sleep(.005)
                task.cancel();release.set()
                with self.assertRaises(asyncio.CancelledError):await task
        asyncio.run(run())
        outcomes=[e for e in audit if e['event']=='outcome']
        self.assertEqual(len(outcomes),1);self.assertTrue(outcomes[0]['storage_committed'])
        self.assertEqual(outcomes[0]['usage']['prompt_tokens'],5)
        self.assertIsNotNone(outcomes[0]['finished_unix'])

    def test_policy_boolean_digest_default_and_abandoned_attempt(self):
        default=self.releases.default
        self.config(retry_unknown_outcomes=True);self.assertEqual(default,self.releases.default)
        self.config(retry_unknown_outcomes=False);self.assertNotEqual(default,self.releases.default)
        for invalid in (0,1,'false',None):
            with self.assertRaises(ValueError):Releases(configurations(retry_unknown_outcomes=invalid),'test')
        job=self.submit(('a',));self.claim();self.clock.now+=61
        self.assertIsNone(self.claim());self.assertEqual(self.store.status('alice',job['job_id'])['state'],'paused')
