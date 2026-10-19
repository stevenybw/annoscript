"""Engineering P0 regressions: no live models or business-system access."""
import asyncio
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import patch

import httpx

from lev._data import thaw
from lev._test_evaluation_http import llm_calls, llm_content
from lev.backends import InferenceContext, LLMAPIBackend, BackendFatalError, InferenceError
from lev.backends.base import normalize_call
from lev.compiler import compile_program
from lev.ir import Program
from lev.plan_lowering import lower_to_plan
from lev.server._plan_cache import PlanCache
from lev.server.worker import ObservedBackend, Worker
from lev.test_generation_policy import policy
from lev.test_llm_batch import answer, completion


def identity_program(lev):
    return lev.state


class PlanCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_miss_cancel_and_release_isolation(self):
        ir = compile_program(identity_program).to_dict()
        cache = PlanCache()
        compiled = []
        def build(encoded):
            time.sleep(.025)
            compiled.append(encoded)
            return lower_to_plan(Program.from_json(encoded.decode()))
        try:
            tasks = [asyncio.create_task(cache.get("a", ir, build)) for _ in range(4)]
            await asyncio.sleep(.005)
            tasks[0].cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.assertIsInstance(results[0], asyncio.CancelledError)
            self.assertEqual(len(compiled), 1)
            self.assertTrue(all(v[0] is results[1][0] for v in results[1:]))
            self.assertEqual((await cache.get("a", ir, build))[1]["status"], "hit")
            await cache.get("b", ir, build)
            changed = json.loads(json.dumps(ir)); changed["source"]["line"] += 1
            await cache.get("a", changed, build)
            self.assertEqual(len(compiled), 3)
            self.assertEqual(cache.stats["waits"], 3)
        finally:
            await cache.aclose()

    async def test_lru_oversize_and_failure_not_cached(self):
        ir = compile_program(identity_program).to_dict()
        build = lambda b: lower_to_plan(Program.from_json(b.decode()))
        for options in ({"max_entries": 1}, {"max_bytes": 1}):
            cache = PlanCache(**options)
            try:
                await cache.get("a", ir, build)
                await cache.get("b", ir, build)
                _, event = await cache.get("a", ir, build)
                self.assertEqual(event["status"], "miss")
                self.assertLessEqual(len(cache.entries), options.get("max_entries", 32))
                self.assertLessEqual(cache.bytes, options.get("max_bytes", 64*1024*1024))
                def fail(b): raise ValueError("bad IR")
                for _ in range(2):
                    with self.assertRaises(ValueError): await cache.get("failure", ir, fail)
                self.assertEqual(cache.stats["failures"], 2)
            finally:
                await cache.aclose()


class PreparedRequestTests(unittest.IsolatedAsyncioTestCase):
    def backend(self, events=None, profile="qwen_vllm", window=8192, counter=None):
        self.sent = []
        def handle(request):
            self.sent.append(request)
            return completion(llm_content(json.loads(request.content), answer))
        return LLMAPIBackend("https://offline.invalid", "fixture", "fixture", profile=profile,
            request_policy=policy(window) if profile == "qwen_vllm" else None,
            _token_counter=counter or (lambda p: 10), transport=httpx.MockTransport(handle),
            diagnostic_observer=events.append if events is not None else None)

    def calls(self, ctx):
        return [normalize_call("a", "generate", ("标题",), {"max_length": 8}, ctx),
                normalize_call("b", "extract", ("地点",),
                               {"source": "广州与北京", "max_items": 2, "max_length_per_item": 8}, ctx)]

    def plan(self, b, ctx, calls):
        return b.plan_batch(ctx, calls, max_calls=8, max_input_bytes=262144,
                            max_output_tokens=4096, default_tokens=1024)

    async def test_selected_payload_sent_without_rebuild_count_or_encoding(self):
        events=[]; b=self.backend(events); ctx=InferenceContext({"text":"广州与北京"}, "业务策略")
        calls=self.calls(ctx)
        try:
            p=self.plan(b,ctx,calls)
            before=dict(b.preparation_stats)
            self.assertIs(self.plan(b,ctx,calls),p)
            with patch.object(b,"_prepare",side_effect=AssertionError("prepared again")):
                out=await b.infer_batch(ctx,calls,max_tokens=p.max_tokens,prepared=p.prepared)
            self.assertEqual(out.responses['b'].items,['广州'])
            self.assertEqual(b.preparation_stats['token_counts'],before['token_counts'])
            self.assertEqual(b.preparation_stats['encodes'],before['encodes'])
            self.assertEqual(self.sent[0].content,p.prepared.body)
            self.assertEqual(next(e['body'] for e in events if e['event']=='request'),self.sent[0].content)
            self.assertTrue(thaw(p.prepared.payload)['messages'][0]['content'].endswith('\n\n业务策略'))
            with self.assertRaises(TypeError): p.prepared.payload['model']='changed'
        finally: await b.aclose()

    async def test_stale_budget_context_calls_and_config_are_reprepared(self):
        b=self.backend();ctx=InferenceContext({"text":"广州与北京"});calls=self.calls(ctx)
        try:
            p=self.plan(b,ctx,calls)
            tests=[(ctx,calls,p.max_tokens+1),
                   (InferenceContext({"text":"独立来源"}),calls,p.max_tokens),
                   (ctx,[replace(calls[0],arguments={**calls[0].arguments,"instructions":"不同标题"}),calls[1]],p.max_tokens)]
            for c,cs,budget in tests:
                before=b.preparation_stats['payload_builds']
                await b.infer_batch(c,cs,max_tokens=budget,prepared=p.prepared)
                self.assertEqual(b.preparation_stats['payload_builds'],before+1)
                self.assertEqual(json.loads(self.sent[-1].content)['max_tokens'],budget)
            b.model="different"
            self.assertIsNot(self.plan(b,ctx,calls),p)
        finally: await b.aclose()

    async def test_source_refs_single_repair_batch_and_other_profiles(self):
        texts=['广州与北京','甲"乙\\丙\n😀e\u0301','', '\\u1234']
        for profile in ['qwen_vllm','legacy']:
            for text in texts:
                for state in [text,{"text":text,"other":"不能抽取"}]:
                    b=self.backend(profile=profile);ctx=InferenceContext(state)
                    c=normalize_call('e','extract',('原文',),{'source':text,'max_items':2,'max_length_per_item':8},ctx)
                    try:
                        for feedback in [None,'bad structure']:
                            p=b._prepare(ctx,[c],80,repair_feedback=feedback)
                            wire=json.loads(thaw(p.payload)['messages'][-1]['content'])
                            self.assertEqual(wire['state'],state)
                            self.assertEqual('source_ref' in wire['task'],profile=='qwen_vllm')
                            self.assertEqual(llm_calls(thaw(p.payload))[1][0][1].arguments['source'],text)
                        other=normalize_call('other','extract',('原文',),{'source':'独立来源','max_items':1,'max_length_per_item':8},ctx)
                        p=self.plan(b,ctx,[c,other,replace(c,id='again')])
                        reconstructed=llm_calls(thaw(p.prepared.payload))[1]
                        self.assertEqual([v.arguments['source'] for _,v in reconstructed],[text,'独立来源',text])
                        wire=json.loads(thaw(p.prepared.payload)['messages'][-1]['content'])
                        self.assertEqual(len(wire['sources']),2)
                    finally: await b.aclose()

    async def test_prepared_pause_blocks_http_and_local_rejection_not_sent(self):
        b=self.backend();ctx=InferenceContext('广州与北京');calls=self.calls(ctx)
        p=self.plan(b,ctx,calls)
        allowed=True
        observed=ObservedBackend(b,[],lambda:True,dispatch_allowed=lambda:allowed)
        allowed=False
        try:
            with self.assertRaises(BackendFatalError):
                await observed.infer_batch(ctx,calls,max_tokens=p.max_tokens,prepared=p.prepared)
            self.assertEqual(self.sent,[])
        finally:await b.aclose()
        b=self.backend(counter=lambda p:8192)
        try:
            with self.assertRaises(InferenceError) as e:await b.infer(ctx,calls[0])
            self.assertEqual(e.exception.metadata['dispatch_status'],'not_sent')
            self.assertEqual(self.sent,[])
        finally:await b.aclose()


class WorkerRefillTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_refills_before_poll_timeout(self):
        stop=asyncio.Event();times=[]
        class Store:
            n=0
            def maintain(self):pass
            def claim(self,*args):
                if self.n>=2:return None
                self.n+=1;return self.n
        class TestWorker(Worker):
            async def execute(self,claim):
                times.append(time.monotonic())
                await asyncio.sleep(.01)
                if claim==2:stop.set()
        w=TestWorker(Store(),None,concurrency=1,poll_interval=1)
        await asyncio.wait_for(w.run(stop=stop),timeout=.5)
        self.assertEqual(len(times),2)
        self.assertLess(times[1]-times[0],.2)


if __name__ == '__main__': unittest.main()
