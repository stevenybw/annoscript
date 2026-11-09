"""Synthetic end-to-end checks; no external services or application fixtures."""
import asyncio
from collections import Counter
import json
from pathlib import Path
import unittest

from annoscript._test_artifacts import assert_saved_artifact
from annoscript._test_evaluation_http import http_backends
from annoscript._test_synthetic import CannedBackend, cases
from annoscript.client import AnnoClient
from annoscript.compiler import compile_program
from annoscript.plan_lowering import lower_to_plan
from annoscript.runtime import AnnoRuntime, RunInput
from annoscript_examples.request import annotate


class SyntheticExampleTests(unittest.TestCase):
    def test_saved_ir_and_plan(self):
        directory = Path(__file__).parents[1] / "annoscript_examples"
        program = compile_program(annotate)
        assert_saved_artifact(self, directory / "request.json", program)
        assert_saved_artifact(self, directory / "request.execution_plan.json", lower_to_plan(program))

    def test_all_modes_backends_and_branch_isolation(self):
        scenarios, inputs, expected, failures = cases()
        for mode in ("strict", "eager", "speculative"):
                with self.subTest(mode=mode):
                    backend = CannedBackend(scenarios, failures, speculative=True)
                    with AnnoClient(AnnoRuntime(backend, backend), mode=mode) as client:
                        results = list(client.submit(annotate, inputs, ordered=True))
                    self.assertEqual([r.index for r in results], list(range(len(inputs))))
                    for result, want in zip(results, expected):
                        self.assertEqual(result.status, want["status"], result.error)
                        if result.status == "success":
                            self.assertEqual(result.value, want["value"])
                        else:
                            self.assertEqual(result.error.code, want["code"])

    def test_real_wire_adapters_and_output_mapping(self):
        async def run(mode, profile):
            scenarios, _, _, _ = cases()
            requests = []
            decision, generation = http_backends(scenarios, requests=requests, profile=profile)
            async with AnnoRuntime(decision, generation) as runtime:
                stream = await runtime.submit(lower_to_plan(compile_program(annotate)),
                                               [RunInput(i, state) for i, (state, _) in enumerate(scenarios[:3])],
                                               mode=mode)
                rows = [r async for r in stream]
            self.assertEqual({r.index: r.value for r in rows}, {i: v for i, (_, v) in enumerate(scenarios[:3])})
            self.assertTrue(all(r.status == "success" for r in rows))
            self.assertTrue(any("questions" in r for r in requests))
            self.assertTrue(any("messages" in r for r in requests))
        for mode in ("strict", "eager", "speculative"):
            for profile in ("legacy", "qwen_vllm"):
                with self.subTest(mode=mode, profile=profile):
                    asyncio.run(run(mode, profile))


if __name__ == "__main__":
    unittest.main()
