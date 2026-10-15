"""The concrete synchronous client and asynchronous runtime share ownership safely."""

import asyncio
import unittest
from unittest.mock import patch
from threading import Event

from lev.client import LevClient, precondition, require
from lev.runtime import LevRuntime
from lev.backends import InferenceOutput, BackendFatalError
from lev.lev import GenerateResponse
from lev.test_runtime import FakeBackend, mixed


def echo(lev):
    return lev.generate(lev.state).text


def constant(lev):
    return [None, "中文"]


def check(state):
    require(state != "invalid", message="invalid record")


@precondition(check)
def checked(lev):
    return lev.generate(lev.state).text


class ClientTests(unittest.TestCase):
    def client(self, backend=None, **options):
        backend = backend or FakeBackend()
        return LevClient(LevRuntime(backend, backend, **options))

    def test_results_preconditions_order_indices_and_repeated_submissions(self):
        with self.client() as client:
            for ordered in (False, True):
                with client.submit(checked, ["a", "invalid", "b"], ordered=ordered) as stream:
                    result = list(stream)
                by_index = {r.index: r for r in result}
                self.assertEqual(by_index[0].value, "a")
                self.assertEqual(by_index[1].error.code, "invalid_input")
                self.assertEqual(by_index[2].value, "b")
                if ordered:
                    self.assertEqual([r.index for r in result], [0, 1, 2])
            self.assertEqual(list(client.submit(constant, [])), [])
            self.assertEqual(list(client.submit(constant, ["x"]))[0].value, [None, "中文"])
            self.assertEqual(list(client.submit(echo, [float("nan")]))[0].error.code, "invalid_input")
        self.assertFalse(client._thread.is_alive())
        self.assertTrue(client.runtime.closed)
        client.close()

    def test_checker_defect_prevents_all_model_calls(self):
        seen = []
        def broken(state):
            seen.append(state)
            if state == "bad":
                raise RuntimeError("checker defect")
        with self.client() as client, patch("lev._client_common._PRECONDITIONS", {echo: (broken,)}):
            with self.assertRaisesRegex(RuntimeError, "checker defect"):
                client.submit(echo, ["first", "bad"])
            self.assertIsNone(client._thread)
            self.assertEqual(client.runtime.llm_backend.calls, [])
        self.assertEqual(seen, ["first", "bad"])

    def test_per_input_failure_isolated_and_fatal_error_closes_stream(self):
        with self.client(FakeBackend(fail=("bad",))) as client:
            result = list(client.submit(echo, ["bad", "ok"], ordered=True))
            self.assertEqual([r.status for r in result], ["error", "success"])
        class Fatal(FakeBackend):
            async def infer(self, context, call):
                raise BackendFatalError("auth failed")
        with self.client(Fatal()) as client:
            stream = client.submit(echo, ["a", "b"])
            with self.assertRaisesRegex(BackendFatalError, "auth failed"):
                list(stream)
            self.assertTrue(stream.closed)

    def test_close_stops_queued_work_without_waiting_for_model(self):
        class Controlled(FakeBackend):
            async def infer(self, context, call):
                self.count += 1
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
        backend = Controlled()
        backend.started, backend.cancelled, backend.count = Event(), Event(), 0
        with self.client(backend, max_llm_inflight=1) as client:
            stream = client.submit(echo, ["a", "b", "c"])
            self.assertTrue(backend.started.wait(2))
            stream.close()
            self.assertTrue(backend.cancelled.wait(2))
            self.assertEqual(backend.count, 1)
            self.assertFalse(client.runtime.closed)
            self.assertEqual(list(stream), [])

    def test_client_does_not_execute_program_function(self):
        with self.client() as client, patch("lev.lev.Lev.__init__", side_effect=AssertionError("must not construct Lev")):
            result = list(client.submit(mixed, ["state"]))
            self.assertEqual(result[0].value, [0.8, "a", 0])

    def test_runtime_has_one_client_owner(self):
        with self.client() as client:
            with self.assertRaisesRegex(ValueError, "unstarted"):
                LevClient(client.runtime)

    def test_results_are_delivered_out_of_order_then_can_be_ordered(self):
        class Ordered(FakeBackend):
            async def infer(self, context, call):
                if call.arguments["instructions"] == "slow":
                    if self.release is None:
                        self.release = asyncio.Event()
                    await self.release.wait()
                    await asyncio.sleep(0)
                else:
                    if self.release is None:
                        self.release = asyncio.Event()
                    self.release.set()
                return InferenceOutput(GenerateResponse(call.arguments["instructions"]))
        for ordered in (False, True):
            backend = Ordered()
            backend.release = None
            with self.client(backend) as client:
                result = list(client.submit(echo, ["slow", "fast"], ordered=ordered))
                self.assertEqual([r.index for r in result], [0, 1] if ordered else [1, 0])


if __name__ == "__main__":
    unittest.main()
