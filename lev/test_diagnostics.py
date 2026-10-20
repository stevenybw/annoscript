"""Opt-in raw diagnostics preserve wire behavior, failure isolation and usage."""
import asyncio,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import httpx
from lev.backends import LLMAPIBackend,DecisionAPIBackend,InferenceContext,InferenceError
from lev.backends.base import normalize_call
from lev.server.store import Store
from lev.server.releases import Releases
from lev.server.worker import Worker
from lev.test_jobs_server import configurations,payload
from lev.test_llm_batch import mixed_generation,answer,completion
from lev._test_evaluation_http import llm_content,llm_calls

class WireTests(unittest.IsolatedAsyncioTestCase):
 async def test_raw_before_validation_and_same_requests_and_results(self):
  for content,status,finish,usage in [('{"items":["广州"]}',200,'stop',{'prompt_tokens':5,'completion_tokens':2}),('{',200,'stop',None),('{"items":["北京"]}',200,'stop',{}),('{"items":[]}',200,'length',{}),('{"error":{"message":"context length"}}',400,None,None)]:
   bodies=[];outcomes=[];events=[]
   for enabled in (False,True):
    def handler(req):
     bodies.append(req.content)
     body={'choices':[{'message':{'content':content},'finish_reason':finish}]}
     if usage is not None:body['usage']=usage
     return httpx.Response(status,json=body) if status==200 else httpx.Response(status,content=content)
    b=LLMAPIBackend('https://offline.invalid','secret-fixture','qwen',profile='qwen_vllm',transport=httpx.MockTransport(handler),diagnostic_observer=events.append if enabled else None)
    context=InferenceContext('广州');call=normalize_call('topic','extract',('地名',),{'max_items': 8, 'max_length_per_item': 58},context)
    try:
     result=await b.infer(context,call);outcomes.append(result.response.items)
    except InferenceError as e:outcomes.append(e.code)
    finally:await b.aclose()
   self.assertEqual(bodies[0],bodies[1]);self.assertEqual(outcomes[0],outcomes[1])
   wire_events=[e for e in events if e['event']!='advisory_lengths']
   self.assertEqual([e['event'] for e in wire_events],['request','response','outcome'])
   self.assertEqual(len({e['request_id'] for e in events}),1)
   self.assertEqual(events[0]['body'],bodies[1]);self.assertTrue(events[1]['body'])
   self.assertNotIn('secret-fixture',repr(events));self.assertEqual(wire_events[-1]['http_status'],status)

class WorkerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.store=Store(Path(self.temp.name)/'tasks.sqlite');self.releases=Releases(configurations(retry_unknown_outcomes=False),'test')
 def worker(self,handler,**kw):
  def backend(_):
   transport=httpx.MockTransport(handler)
   return DecisionAPIBackend('https://offline.invalid','secret','m',transport=transport),LLMAPIBackend('https://offline.invalid','secret','m',profile='qwen_vllm',transport=transport)
  return Worker(self.store,self.releases,backend_factory=backend,**kw)
 def submit(self,key,**kw):return self.store.submit('alice',key,payload(**kw),self.releases)
 def test_batch_repair_raw_parent_usage_and_runtime_mapping(self):
  calls=[];audit=[];raw=[]
  def handler(req):
   body=json.loads(req.content);calls.append(body);value=llm_content(body,answer)
   if llm_calls(body)[0]:
    v=json.loads(value);del v['answers'][next(iter(v['answers']))];value=json.dumps(v)
   return completion(value)
  job=self.submit('batch',states=('广州与北京',),program=mixed_generation)
  asyncio.run(self.worker(handler,request_observer=audit.append,diagnostic_observer=raw.append).execute(self.store.claim('w',self.releases)))
  self.assertEqual(self.store.status('alice',job['job_id'])['counts']['success'],1)
  rr=[e for e in raw if e['event']=='request'];self.assertEqual(len(rr),2)
  self.assertEqual(rr[1]['repair_parent_request_ids'],[rr[0]['request_id']]);self.assertTrue(rr[1]['repair'])
  self.assertEqual({e['request_id'] for e in audit},{e['request_id'] for e in rr})
  self.assertTrue(any(e['event']=='trace' and e['plan'] for e in raw))
 def test_diagnostic_failure_pauses_without_model_retry_and_keeps_usage(self):
  for phase,expected in [('request',0),('response',1),('outcome',1)]:
   job=self.submit(phase,states=('one',));calls=[];audit=[]
   def observe(event):
    if event['event']==phase:raise OSError('disk fixture')
   def handler(req):calls.append(req);return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'yes'}}],'usage':{'prompt_tokens':5,'completion_tokens':1}})
   asyncio.run(self.worker(handler,request_observer=audit.append,diagnostic_observer=observe).execute(self.store.claim('w',self.releases)))
   self.assertEqual(len(calls),expected)
   self.assertEqual(self.store.status('alice',job['job_id'])['pause_reason']['code'],'diagnostics_failed')
   if expected:self.assertEqual([x for x in audit if x['event']=='outcome'][0]['usage']['prompt_tokens'],5)
 def test_item_failure_completes_and_next_batch_can_be_claimed(self):
  count=[]
  for batch in range(2):
   job=self.submit(str(batch),states=tuple(str(i) for i in range(20)))
   def handler(req):count.append(req);return httpx.Response(200,content=b'{')
   worker=self.worker(handler,diagnostic_observer=lambda e:None)
   for _ in range(20):asyncio.run(worker.execute(self.store.claim('w',self.releases)))
   self.assertEqual(self.store.status('alice',job['job_id'])['state'],'completed')
   self.assertEqual(self.store.status('alice',job['job_id'])['counts']['error'],20)
  self.assertEqual(len(count),40)
 def test_cancel_during_commit_keeps_raw_response_and_matching_usage(self):
  self.submit('cancel',states=('one',));entered=threading.Event();release=threading.Event();original=self.store.request_finished;raw=[];audit=[]
  def finish(*a,**kw):entered.set();release.wait(5);return original(*a,**kw)
  def handler(req):return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'yes'}}],'usage':{'prompt_tokens':9,'completion_tokens':1}})
  async def run():
   with patch.object(self.store,'request_finished',side_effect=finish):
    task=asyncio.create_task(self.worker(handler,request_observer=audit.append,diagnostic_observer=raw.append).execute(self.store.claim('w',self.releases)))
    async with asyncio.timeout(5):
     while not entered.is_set():await asyncio.sleep(.005)
    task.cancel();release.set()
    with self.assertRaises(asyncio.CancelledError):await task
  asyncio.run(run())
  self.assertEqual(len([e for e in raw if e['event']=='response']),1)
  self.assertEqual([e for e in audit if e['event']=='outcome'][0]['usage']['prompt_tokens'],9)
  self.assertEqual([e for e in raw if e['event']=='outcome'][0]['http_status'],200)
