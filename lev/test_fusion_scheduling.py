"""Shared admission and nullable semantics; synthetic/local HTTP only."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from lev._dispatch_pool import RequestSlots
from lev.backends import BatchResult, InferenceOutput, DecisionAPIBackend, LLMAPIBackend
from lev.compiler import compile_program
from lev.lev import ChoiceResponse, GenerateResponse
from lev.plan_lowering import lower_to_plan
from lev.runtime import LevRuntime, RunInput
from lev.server.worker import Worker
from lev.server.store import Store
from lev.server.releases import Releases
from lev.test_jobs_server import configurations
from lev.remote import ir_sha256


def join_ready(lev):
    first = lev.choice("ready immediately", {"yes": "yes", "no": "no"}).choice
    text = lev.generate("synthetic", max_length=10).text
    second = lev.choice(text, {"yes": "yes", "no": "no"}).choice
    return [first, second]


class SlotTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_cancel_duplicate_wait_and_idempotent_release(self):
        p = RequestSlots(generation=1, decision=1)
        woke = []
        r = p.acquire("llm", "a", lambda: woke.append("a"))
        self.assertIsNone(p.acquire("llm", "b", lambda: woke.append("b")))
        self.assertIsNone(p.acquire("llm", "c", lambda: woke.append("c")))
        self.assertIsNone(p.acquire("llm", "b", lambda: woke.append("duplicate")))
        r(); r()
        self.assertEqual(woke, ["b"])
        self.assertIsNone(p.acquire("llm", "c", lambda: None))
        p.cancel("b")
        c = p.acquire("llm", "c", lambda: None); c()
        self.assertEqual(p.active, {"llm": 0, "decision": 0})
        self.assertFalse(p.waiting("b"))

    async def test_waiting_batches_are_selected_at_admission(self):
        pool = RequestSlots(generation=8, decision=1)
        release = pool.acquire("decision", "external", lambda: None)
        batches = []
        generated = asyncio.Event()
        class Backend:
            async def infer(self, ctx, call):
                generated.set()
                return InferenceOutput(GenerateResponse("newly ready"))
            async def infer_batch(self, ctx, calls):
                batches.append([c.arguments["instructions"] for c in calls])
                return BatchResult({c.id: ChoiceResponse("yes", 1., {"yes": 1., "no": 0.}) for c in calls})
            async def aclose(self): pass
        b = Backend()
        async with LevRuntime(b, b, request_slots=pool, trace=True, max_decision_inflight=1) as runtime:
            stream = await runtime.submit(lower_to_plan(compile_program(join_ready)), [RunInput(0, "fixture")])
            await generated.wait()
            for _ in range(8): await asyncio.sleep(0)
            self.assertFalse(batches)
            release()
            result = [r async for r in stream]
            self.assertEqual(result[0].value, ["yes", "yes"])
            self.assertEqual(batches, [["ready immediately", "newly ready"]])
            self.assertTrue(any(e['event'] == 'slot_wait' for e in runtime.trace))
        self.assertEqual(sum(pool.active.values()), 0)

    async def test_cancel_waiter_does_not_send_or_leak(self):
        pool = RequestSlots(generation=1, decision=1)
        a = pool.acquire('decision', 'held', lambda: None)
        b = pool.acquire('llm', 'held', lambda: None)
        class Never:
            async def infer(self, *a, **k): raise AssertionError('sent')
            async def infer_batch(self, *a, **k): raise AssertionError('sent')
            async def aclose(self): pass
        async with LevRuntime(Never(), Never(), request_slots=pool) as rt:
            stream = await rt.submit(lower_to_plan(compile_program(join_ready)), [RunInput(0, 'fixture')])
            for _ in range(8): await asyncio.sleep(0)
            await stream.aclose()
        a(); b()
        self.assertFalse(any(pool.waiters.values()))
        self.assertEqual(sum(pool.active.values()), 0)


class SharedWorkerTests(unittest.IsolatedAsyncioTestCase):
    pass


from lev.test_generation_policy import P0ProtocolWorkerTests, P0DiagnosticWorkerTests

class SharedProtocolTests(P0ProtocolWorkerTests):
    def run_http(self, handler, *, observer=None):
        w = super().run_http(handler, observer=observer)
        w.concurrency = 8
        w.request_slots = RequestSlots(generation=8, decision=4)
        return w

class SharedDiagnosticTests(P0DiagnosticWorkerTests):
    def worker(self, handler, **kw):
        w = super().worker(handler, **kw)
        w.concurrency = 8
        w.request_slots = RequestSlots(generation=8, decision=4)
        return w


if __name__ == '__main__': unittest.main()
