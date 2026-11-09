"""Offline wire, scheduling, bounded repair and car acceptance for LLM batches."""

import asyncio
import json
import unittest

import httpx

from annoscript._test_evaluation_http import llm_calls, llm_content, llm_wire_id
from annoscript.backends import InferenceContext, InferenceError, BackendFatalError
from annoscript._test_evaluation_http import FixtureLLMBackend as LLMAPIBackend
from annoscript.backends.base import normalize_call, encode_payload
from annoscript.backends._bounded_grammar import grammar
def batch_grammar(calls): return grammar(calls, batch=True)
from annoscript.compiler import compile_program
from annoscript.anno import GenerateResponse, GenerateListResponse, ExtractResponse
from annoscript.plan_lowering import lower_to_plan
from annoscript.runtime import AnnoRuntime, RunInput
from annoscript.test_runtime import FakeBackend


def mixed_generation(anno):
    title = anno.generate("标题", max_length=8, max_num_tokens=16).text
    labels = anno.generate_list("维度", max_items=2, max_num_tokens=32, max_length_per_item=58).items
    places = anno.extract("地点", min_items=1, max_items=8, max_length_per_item=58).items
    return [title, labels, places]


def independent_generation(anno):
    a = anno.generate("first", max_length=512).text
    b = anno.generate("second", max_length=512).text
    c = anno.generate("third", max_length=512).text
    return [a, b, c]


def dependent_generation(anno):
    a = anno.generate("first", max_length=512).text
    b = anno.generate("second", max_length=512).text
    return [a, b, anno.generate("dependent:" + a, max_length=512).text]


def conditional_generation(anno):
    gate = anno.generate("no", max_length=512).text
    if gate == "yes":
        return anno.generate("inactive", max_length=512).text
    return gate


def generation_loop(anno):
    result = []
    for item in anno.state:
        result = result + [anno.generate(item, max_length=512).text]
    return result


def reused_generation(anno):
    a = anno.generate("same", max_length=512).text
    b = anno.generate("same", max_length=512).text
    return [a, b, anno.generate("different", max_length=512).text]


def answer(context, call):
    if call.operation == "generate":
        return GenerateResponse(call.arguments["instructions"])
    if call.operation == "generate_list":
        return GenerateListResponse(["条目"])
    return ExtractResponse(["广州"])


def completion(content, *, finish="stop", refusal=None):
    return httpx.Response(200, json={"choices": [{"finish_reason": finish,
        "message": {"content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 5}})


def wire(handler=None, *, profile="legacy"):
    requests, sizes = [], []
    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        sizes.append(len(request.content))
        if handler is not None:
            response = handler(body)
            if hasattr(response, "__await__"):
                response = await response
            if response is not None:
                return response
        return completion(llm_content(body, answer))
    backend = LLMAPIBackend("https://offline.invalid/v1", "fixture", "fixture", profile=profile,
                            transport=httpx.MockTransport(handle))
    backend.test_request_sizes = sizes
    return backend, requests


async def execute(function=independent_generation, state="state", *, handler=None, profile="legacy",
                  mode="eager", inputs=None, **options):
    backend, requests = wire(handler, profile=profile)
    async with AnnoRuntime(FakeBackend(), backend, trace=True, **options) as runtime:
        plan = lower_to_plan(compile_program(function))
        stream = await runtime.submit(plan, inputs or [RunInput(0, state)], mode=mode)
        results = [r async for r in stream]
    return results, requests, runtime.trace


class BatchBackendTests(unittest.IsolatedAsyncioTestCase):
    def calls(self):
        context = InferenceContext({"text": "背景", "places": "广州与北京"}, "共同规则")
        calls = [normalize_call("a", "generate", ("标题",), {"max_num_tokens": 16, 'max_length': 512}, context),
                 normalize_call("b", "generate_list", ("维度",), {"max_num_tokens": 32, 'max_items': 8, 'max_length_per_item': 58}, context),
                 normalize_call("c", "extract", ("地点",), {"source_field": "places", 'max_items': 8, 'max_length_per_item': 58}, context),
                 normalize_call("d", "extract", ("城市",), {"source_field": "places", 'max_items': 8, 'max_length_per_item': 58}, context)]
        return context, calls

    async def test_mixed_profiles_source_dedup_reordering_and_exact_bytes(self):
        context, calls = self.calls()
        for profile in ("legacy", "qwen_vllm"):
            def reverse(body):
                value = json.loads(llm_content(body, answer))
                value["answers"] = dict(reversed(list(value["answers"].items())))
                return completion(json.dumps(value, ensure_ascii=False))
            backend, requests = wire(reverse, profile=profile)
            try:
                plan = backend.plan_batch(context, calls, max_calls=8, max_input_bytes=262144,
                                          max_output_tokens=16384, default_tokens=1024)
                self.assertEqual(plan.max_tokens, 64 + 16 + 32 + 1024 * 2 + 32 * 4)
                self.assertEqual(requests, [])
                result = await backend.infer_batch(context, plan.calls, max_tokens=plan.max_tokens)
                self.assertEqual(result.responses["a"].text, "标题")
                self.assertEqual(result.responses["c"].items, ["广州"])
                self.assertEqual(result.errors, {})
                body = requests[0]
                self.assertEqual(len(encode_payload(body)), plan.input_bytes)
                self.assertEqual(backend.test_request_sizes, [plan.input_bytes])
                payload = json.loads(body["messages"][-1]["content"])
                self.assertEqual(payload, {"text": "背景", "places": "广州与北京"})
                self.assertNotIn("sources", payload)
                self.assertIn('输入的顶层字段 "places"', body["messages"][0]["content"])
                if profile == "qwen_vllm":
                    self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
                    self.assertTrue(body["messages"][0]["content"].startswith("共同规则\n"))
                    self.assertEqual(body["structured_outputs"], {"grammar": batch_grammar(calls)})
                    self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                else:
                    self.assertTrue(body["messages"][0]["content"].startswith("共同规则\n"))
                    self.assertEqual(body["response_format"], {"type": "json_object"})
                    self.assertEqual(body["thinking"], {"type": "disabled"})
            finally:
                await backend.aclose()

    async def test_planning_boundaries_do_not_truncate_or_expand_individual_calls(self):
        context, calls = self.calls()
        backend, requests = wire()
        limits = dict(max_calls=8, max_input_bytes=262144, max_output_tokens=16384, default_tokens=1024)
        two = backend.plan_batch(context, calls[:2], **limits)
        for change in ({"max_calls": 2}, {"max_output_tokens": two.max_tokens}, {"max_input_bytes": two.input_bytes}):
            plan = backend.plan_batch(context, calls, **(limits | change))
            self.assertEqual(plan.calls, tuple(calls[:2]))
        for change in ({"max_calls": 1}, {"max_output_tokens": two.max_tokens - 1}, {"max_input_bytes": two.input_bytes - 1}):
            self.assertIsNone(backend.plan_batch(context, calls[:2], **(limits | change)))
        self.assertEqual(requests, [])
        await backend.aclose()

    async def test_batch_thinking_override_can_be_omitted(self):
        context, calls = self.calls()
        for profile in ("legacy", "qwen_vllm"):
            backend, requests = wire(profile=profile)
            backend.disable_thinking = False
            try:
                await backend.infer_batch(context, calls, max_tokens=4096)
                self.assertNotIn("thinking", requests[0])
                self.assertNotIn("chat_template_kwargs", requests[0])
            finally:
                await backend.aclose()

    async def test_per_item_errors_preserve_good_results(self):
        context, calls = self.calls()
        def invalid(body):
            value = json.loads(llm_content(body, answer))
            del value["answers"]["q2"]
            value["answers"]["q3"] = ["深圳"]
            value["answers"]["q4"] = "wrong value type"
            return completion(json.dumps(value))
        backend, _ = wire(invalid)
        try:
            result = await backend.infer_batch(context, calls, max_tokens=2048)
            self.assertEqual(set(result.responses), {"a"})
            self.assertEqual(set(result.errors), {"b", "c", "d"})
            self.assertTrue(all(e.call_id == key for key, e in result.errors.items()))
        finally:
            await backend.aclose()

    async def test_unreliable_mapping_truncation_refusal_and_http_classification(self):
        context, calls = self.calls()
        cases = [
            (completion('{"answers":{'), "invalid_batch_output"),
            (completion('{"answers":{"q1":[],"q1":[]}}'), "invalid_batch_output"),
            (completion('{"answers":{"unknown":{}}}'), "invalid_batch_output"),
            (completion('{"answers":{}}', finish="length"), "incomplete_batch_generation"),
            (completion(None, refusal="no"), "model_refusal"),
            (completion(None, finish="content_filter"), "model_refusal"),
            (httpx.Response(200, json={"choices": [{"finish_reason": "content_filter"}]}), "model_refusal"),
            (httpx.Response(429, json={}), "model_http_error"),
            (httpx.Response(200, text="not JSON"), "invalid_response"),
        ]
        for response, code in cases:
            backend, requests = wire(lambda _: response)
            try:
                with self.subTest(code=code), self.assertRaises(InferenceError) as caught:
                    await backend.infer_batch(context, calls, max_tokens=1024)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(len(requests), 1)
            finally:
                await backend.aclose()

    async def test_unicode_and_all_per_item_contracts(self):
        source = ' 中文😀 "引号" \\路径\n换行\t '
        context = InferenceContext(source)
        calls = [normalize_call("a", "generate", ("text",), {"min_length": 1, "max_length": len(source)}, context),
                 normalize_call("b", "generate_list", ("list",), {"min_items": 1, "max_items": 1, "max_length_per_item": len(source)}, context),
                 normalize_call("c", "extract", ("source",), {"min_items": 1, "max_length_per_item": len(source), 'max_items': 8}, context)]
        valid = {"q1": source, "q2": [source], "q3": [source]}
        for ascii_only in (False, True):
            backend, _ = wire(lambda _: completion(json.dumps({"answers": valid}, ensure_ascii=ascii_only)))
            try:
                result = await backend.infer_batch(context, calls, max_tokens=1024)
                self.assertEqual(result.responses["a"].text, source)
                self.assertEqual(result.responses["c"].items, [source])
            finally:
                await backend.aclose()
        for key, identity, value in (("q1", "a", None), ("q1", "a", ""),
                                     ("q2", "b", []), ("q2", "b", [source, source]),
                                     ("q3", "c", [""])):
            invalid = json.loads(json.dumps(valid))
            invalid[key] = value
            backend, _ = wire(lambda _: completion(json.dumps({"answers": invalid})))
            try:
                result = await backend.infer_batch(context, calls, max_tokens=1024)
                self.assertEqual(set(result.errors), {identity})
            finally:
                await backend.aclose()


class BatchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_mixed_batch_and_disable_restores_original_wire(self):
        for profile in ("legacy", "qwen_vllm"):
            results, requests, trace = await execute(mixed_generation, state="广州与北京", profile=profile)
            self.assertEqual(results[0].value, ["标题", ["条目"], ["广州"]])
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["max_tokens"], 64 + 16 + 32 + 1024 + 32 * 3)
            self.assertEqual(sum(e.get("usage", {}).get("prompt_tokens", 0) for e in trace), 13)
            results, requests, _ = await execute(mixed_generation, state="广州与北京", profile=profile, max_llm_batch_calls=1)
            self.assertEqual(len(requests), 3)
            self.assertEqual([r.get("max_tokens") for r in requests], [16, 32, 1024])
            self.assertTrue(all(not llm_calls(r)[0] for r in requests))

    async def test_count_byte_budget_and_oversized_single_fallback(self):
        _, reference, _ = await execute(max_llm_batch_calls=2)
        two_call_bytes = len(encode_payload(reference[0]))
        for options in ({"max_llm_batch_calls": 2}, {"max_llm_batch_output_tokens": 64 + 2 * (1024 + 32)},
                        {"max_llm_batch_input_bytes": two_call_bytes}):
            results, requests, _ = await execute(**options)
            self.assertEqual(results[0].value, ["first", "second", "third"])
            self.assertEqual([len(llm_calls(r)[1]) for r in requests], [2, 1])
        for options in ({"max_llm_batch_input_bytes": 1}, {"max_llm_batch_output_tokens": 1}):
            results, requests, _ = await execute(**options)
            self.assertEqual(results[0].status, "success")
            self.assertEqual(len(requests), 3)
            self.assertTrue(all(r["max_tokens"] == 1024 for r in requests))

    async def test_dependencies_strict_activation_reuse_and_cross_input_isolation(self):
        _, requests, _ = await execute(dependent_generation)
        self.assertEqual([len(llm_calls(r)[1]) for r in requests], [2, 1])
        self.assertEqual(llm_calls(requests[-1])[1][0][1].arguments["instructions"], "dependent:first")
        _, requests, _ = await execute(mode="strict")
        self.assertEqual([llm_calls(r)[1][0][1].arguments["instructions"] for r in requests], ["first", "second", "third"])
        _, requests, _ = await execute(conditional_generation)
        self.assertEqual(len(requests), 1)
        results, requests, _ = await execute(reused_generation)
        self.assertEqual(results[0].value, ["same", "same", "different"])
        self.assertEqual(len(llm_calls(requests[0])[1]), 2)
        results, requests, _ = await execute(inputs=[RunInput(0, "same"), RunInput(1, "same")])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual([len(llm_calls(r)[1]) for r in requests], [3, 3])

    async def test_loops_local_quantum_and_physical_concurrency_bound(self):
        active = peak = 0
        async def handle(body):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
        results, requests, _ = await execute(generation_loop, [str(i) for i in range(19)], handler=handle,
            max_llm_batch_calls=3, loop_window=5, local_task_budget=20, max_llm_inflight=2)
        self.assertEqual(results[0].value, [str(i) for i in range(19)])
        self.assertLessEqual(peak, 2)
        sizes = [len(llm_calls(r)[1]) for r in requests]
        self.assertEqual(sum(sizes), 19)
        self.assertLessEqual(max(sizes), 3)
        self.assertGreater(max(sizes), 1)

    async def test_partial_repair_keeps_good_answers_and_records_usage_once(self):
        def fail_one(body):
            if llm_calls(body)[0]:
                value = json.loads(llm_content(body, answer))
                del value["answers"][next(iter(value["answers"]))]
                return completion(json.dumps(value))
        results, requests, trace = await execute(handler=fail_one)
        self.assertEqual(results[0].value, ["first", "second", "third"])
        self.assertEqual([len(llm_calls(r)[1]) for r in requests], [3, 1])
        self.assertEqual(llm_calls(requests[1])[1][0][1].arguments["instructions"], "first")
        self.assertEqual(requests[1]["max_tokens"], 1024)
        self.assertIn("未通过格式或约束检查", requests[1]["messages"][0]["content"])
        submitted = [e for e in trace if e["event"] == "http_submitted"]
        self.assertEqual(submitted[1]["repair_parent"], submitted[0]["batch"])
        self.assertEqual(submitted[1]["repair_attempt"], 1)
        self.assertEqual(sum(e.get("usage", {}).get("prompt_tokens", 0) for e in trace), 26)

    async def test_unreliable_batch_repairs_every_item_once_and_failure_is_terminal(self):
        for bad in (completion("{"), completion('{"answers":{}}', finish="length")):
            def handle(body):
                return bad if llm_calls(body)[0] else None
            results, requests, _ = await execute(handler=handle)
            self.assertEqual(results[0].status, "success")
            self.assertEqual([len(llm_calls(r)[1]) for r in requests], [3, 1, 1, 1])
        def fail_again(body):
            return completion("{") if llm_calls(body)[0] else completion("", finish="length")
        results, requests, _ = await execute(handler=fail_again, max_llm_inflight=1)
        self.assertEqual(results[0].status, "error")
        self.assertEqual(len(requests), 2)
        results, requests, _ = await execute(handler=lambda _: completion("{"), llm_batch_repair_attempts=0)
        self.assertEqual(results[0].status, "error")
        self.assertEqual(len(requests), 1)

    async def test_no_content_repair_on_network_errors_or_refusal(self):
        for response in (httpx.Response(429, json={}), completion(None, refusal="no"),
                         httpx.Response(200, json={"choices": [{"finish_reason": "content_filter"}]})):
            results, requests, _ = await execute(handler=lambda _: response)
            self.assertEqual(results[0].status, "error")
            self.assertEqual(len(requests), 1)
        for error in (httpx.ReadTimeout("offline"), httpx.ReadError("offline")):
            def fail(body):
                raise error
            results, requests, _ = await execute(handler=fail)
            self.assertEqual(results[0].status, "error")
            self.assertEqual(len(requests), 1)
        def disconnected(body):
            raise httpx.ConnectError("offline")
        backend, requests = wire(disconnected)
        async with AnnoRuntime(FakeBackend(), backend) as runtime:
            stream = await runtime.submit(lower_to_plan(compile_program(independent_generation)), [RunInput(0, "state")])
            with self.assertRaises(BackendFatalError):
                [r async for r in stream]
        self.assertEqual(len(requests), 1)

    async def test_repair_preserves_explicit_budget_task_state_and_source(self):
        for operation in ("generate", "generate_list", "extract"):
            def invalid(body):
                batched, calls = llm_calls(body)
                if batched:
                    value = json.loads(llm_content(body, answer))
                    call = next(c for _, c in calls if c.operation == operation)
                    value["answers"][llm_wire_id(body, call.id)] = {"type": "wrong"}
                    return completion(json.dumps(value))
            results, requests, trace = await execute(mixed_generation, state="广州与北京", handler=invalid, max_llm_inflight=1)
            self.assertEqual(results[0].status, "success")
            self.assertEqual(len(requests), 2)
            context, repair = llm_calls(requests[1])[1][0]
            original = next(c for _, c in llm_calls(requests[0])[1] if c.operation == operation)
            self.assertEqual(repair.arguments, original.arguments)
            self.assertEqual(context.state, "广州与北京")
            self.assertEqual(requests[1]["max_tokens"], {"generate": 16, "generate_list": 32, "extract": 1024}[operation])
            self.assertNotIn("content", str(trace))

    async def test_strict_failure_order_and_fairness_across_runs(self):
        results, requests, _ = await execute(mode="strict", handler=lambda _: completion("", finish="length"))
        self.assertEqual(results[0].status, "error")
        self.assertEqual(len(requests), 1)
        _, requests, _ = await execute(inputs=[RunInput(0, "A"), RunInput(1, "B")],
                                        max_llm_batch_calls=2, max_llm_inflight=1)
        self.assertEqual([llm_calls(r)[1][0][0].state for r in requests], ["A", "B", "A", "B"])

    async def test_repairs_precede_ready_calls_and_other_runs(self):
        def invalid(body):
            batched, calls = llm_calls(body)
            if batched and calls[0][0].state == "A":
                value = json.loads(llm_content(body, answer))
                del value["answers"][llm_wire_id(body, calls[1][1].id)]
                return completion(json.dumps(value))
        results, requests, _ = await execute(dependent_generation, state="A", handler=invalid, max_llm_inflight=1)
        self.assertEqual(results[0].status, "success")
        self.assertEqual([llm_calls(r)[1][-1][1].arguments["instructions"] for r in requests],
                         ["second", "second", "dependent:first"])
        results, requests, _ = await execute(inputs=[RunInput(0, "A"), RunInput(1, "B")],
                                             handler=invalid, max_llm_inflight=1)
        self.assertTrue(all(r.status == "success" for r in results))
        self.assertEqual([llm_calls(r)[1][0][0].state for r in requests], ["A", "A", "B"])


    async def test_cancellation_during_repair_stops_further_dispatch(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def handle(body):
            if llm_calls(body)[0]:
                return completion("{")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        backend, requests = wire(handle)
        async with AnnoRuntime(FakeBackend(), backend, max_llm_inflight=1) as runtime:
            stream = await runtime.submit(lower_to_plan(compile_program(independent_generation)), [RunInput(0, "state")])
            await asyncio.wait_for(started.wait(), 2)
            await stream.aclose()
            await asyncio.wait_for(cancelled.wait(), 2)
        self.assertEqual(len(requests), 2)

    async def test_configuration_validation(self):
        for kwargs in ({"max_llm_batch_calls": 0}, {"max_llm_batch_input_bytes": True},
                       {"max_llm_batch_output_tokens": -1}, {"llm_batch_default_tokens": 0},
                       {"llm_batch_repair_attempts": 2}, {"llm_batch_repair_attempts": False}):
            with self.assertRaises(ValueError):
                AnnoRuntime(FakeBackend(), FakeBackend(), **kwargs)


if __name__ == "__main__":
    unittest.main()
