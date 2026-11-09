"""Decision-first speculation with selected-path generation and bounded work."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest

import httpx


from lev import test_plan as shapes
from lev import test_plan_constraints as checks
from lev import plan as p
from lev._test_evaluation_http import FixtureLLMBackend, http_backends, llm_calls
from lev.backends import DecisionAPIBackend
from lev.compiler import compile_program
from lev.plan_lowering import lower_to_plan
from lev.plan_formatter import analyze_plan, format_plan_html, format_plan
from lev.runtime import LevRuntime, RunInput
from lev.test_runtime import execute, changed_criteria
from lev.test_generation_policy import policy
from lev._dispatch_pool import RequestSlots
from lev.lev import constrain


def arms(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "yes":
        return lev.generate("wanted", min_length=1, max_length=20, max_num_tokens=64).text
    else:
        return lev.generate("unused", min_length=1, max_length=20, max_num_tokens=64).text


def batch_arms(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "yes":
        first = lev.generate("wanted", min_length=1, max_length=20, max_num_tokens=64).text
        second = lev.generate("companion", min_length=1, max_length=20, max_num_tokens=64).text
        return [first, second]
    return lev.generate("unused", max_length=20).text


def alternating(lev):
    outer = lev.choice("outer", {"yes": "yes", "no": "no"}).choice
    first = lev.generate(outer, max_length=20).text
    inner = lev.choice(first, {"yes": "yes", "no": "no"}).choice
    if inner == "yes":
        return lev.generate("wanted", max_length=20).text
    return lev.generate("unused", max_length=20).text


def qwen_gate(lev):
    first = lev.generate("predicate", max_length=20).text
    if first == "yes":
        return lev.generate("wanted", max_length=20).text
    return lev.generate("unused", max_length=20).text


def delayed_branch(lev):
    first = lev.generate("independent", max_length=20).text
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "yes":
        instructions = "wanted"
    else:
        instructions = "unused"
    return [first, lev.generate(instructions, max_length=20).text]


def unused_decision(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "no":
        return lev.noul("invalid").noul
    return lev.generate("wanted", max_length=20).text


def unused_arguments(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "no":
        return lev.generate(lev.state["missing"], max_length=-1).text
    return lev.generate("wanted", max_length=20).text


def unused_transport(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "no":
        return lev.generate("unused", max_length=20).text
    return "done"


def nested_and_chain(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "yes":
        if lev.noul("inside").noul > 0.5:
            first = lev.generate("first", max_length=20).text
            return lev.generate(first + " next", max_length=20).text
    return "none"


def gated_loop(lev):
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    result = []
    if selected == "yes":
        for item in lev.state:
            result = result + [lev.generate(item, max_length=20).text]
    return result


def unused_check_and_store(lev):
    data = {"value": "original"}
    selected = lev.choice("gate", {"yes": "yes", "no": "no"}).choice
    if selected == "no":
        data["value"] = "changed"
        constrain(False)
        lev.generate("unused", max_length=20)
    return data


async def run(function, state, decision, generation, *, optimize=True, **options):
    async with LevRuntime(decision, generation, trace=True, **options) as runtime:
        plan = lower_to_plan(compile_program(function, optimize=optimize))
        result, = await asyncio.wait_for(collect(runtime, plan, state), 15)
    return result, runtime.trace


async def collect(runtime, plan, state):
    return [r async for r in await runtime.submit(plan, [RunInput(0, state)], mode="speculative")]


def wire_backends(handler):
    transport = httpx.MockTransport(handler)
    return (DecisionAPIBackend("http://offline.invalid/decision", "fixture", "fixture", transport=transport),
            FixtureLLMBackend("http://offline.invalid/v1", "fixture", "fixture",
                              profile="qwen_vllm", transport=transport))


def decision_answer(body, *, selected="yes", invalid=False):
    answers = {}
    for identity, question in body["questions"].items():
        if question["type"] == "choice":
            value = selected
            answers[identity] = {"type": "choice", "choice": value, "confidence": 1,
                                 "probabilities": {k: float(k == value) for k in question["criteria"]}}
        else:
            answers[identity] = {"type": "noul", "noul": "bad" if invalid else 0.8}
    return httpx.Response(200, json={"answers": answers})


def completion(body, answer):
    batched, calls = llm_calls(body)
    answers = {call.arguments.get("field_name", f"q{index}"): answer(call)
               for index, (_, call) in enumerate(calls, 1)}
    content = json.dumps({"answers": answers}) if batched else next(iter(answers.values()))
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})


class SpeculationTests(unittest.IsolatedAsyncioTestCase):


    async def test_existing_memory_loop_and_return_semantics(self):
        cases = [
            (shapes.independent, "s"), (shapes.duplicates, "s"),
            (shapes.branch_duplicates, []), (shapes.branch_duplicates, [1]),
            (shapes.loop_invariant, [True, False, True]),
            (shapes.independent_loop, ["a", "b", "c"]),
            (shapes.last_value, []), (shapes.last_value, ["a", "b"]),
            (shapes.conditional_last, ["a", "", "b"]),
            (shapes.nested_loop, [["a", "b"], [], ["c"]]),
            (shapes.local_container, ["a", "b"]), (shapes.memory_snapshots, "s"),
            (shapes.iteration_alias, ["a", "b", "c"]),
            (shapes.nested_memory, [["a", "b"], [], ["c"]]),
            (shapes.iterator_twice, ["a", "b"]),
            (shapes.early_return, []), (shapes.early_return, [True]),
            (shapes.nested_return, [False]), (shapes.nested_return, [True]),
            (shapes.partial_nested_return, []), (shapes.partial_nested_return, [False]),
            (checks.checked_memory, "s"), (checks.checked_rebinding, "s"),
            (checks.checked_shared, "same"), (checks.checked_memory_loop, ["a", "b"]),
            (changed_criteria, "s"),
        ]
        for function, state in cases:
            with self.subTest(function=function.__name__, state=state):
                expected, _, _ = await execute(function, state, mode="eager", loop_window=2)
                result, _, _ = await asyncio.wait_for(
                    execute(function, state, mode="speculative", loop_window=2), 5)
                self.assertEqual(asdict(result), asdict(expected))

    async def test_only_demanded_invalid_answers_are_repaired(self):
        for fail_wanted in (False, True):
            requests, occurrences = [], {}

            def handler(request):
                body = json.loads(request.content)
                requests.append(body)
                if "questions" in body:
                    return decision_answer(body)
                def answer(call):
                    name = call.arguments["instructions"]
                    occurrences[name] = occurrences.get(name, 0) + 1
                    return "" if name == "unused" or fail_wanted and name == "wanted" and occurrences[name] == 1 else name
                return completion(body, answer)

            result, trace = await run(batch_arms, "text", *wire_backends(handler))
            self.assertEqual(result.value, ["wanted", "companion"], result.error)
            repairs = [e for e in trace if e.get("repair_attempt")]
            self.assertEqual(len(repairs), int(fail_wanted))
            self.assertNotIn("unused", occurrences)
            self.assertEqual(occurrences["companion"], 1)

    async def test_jev_partial_errors_and_argument_errors(self):
        for function, state in ((unused_decision, "text"), (unused_arguments, {})):
            def handler(request):
                body = json.loads(request.content)
                if "questions" in body:
                    return decision_answer(body, invalid=True)
                return completion(body, lambda c: c.arguments["instructions"])
            result, trace = await run(function, state, *wire_backends(handler))
            self.assertEqual(result.value, "wanted", result.error)
            self.assertFalse(any(e.get("repair_attempt") for e in trace))
            if function is unused_decision:
                self.assertTrue(any(e["event"] == "speculation_error_deferred" for e in trace))

    async def test_checks_and_stores_commit_only_on_the_selected_path(self):
        def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                return decision_answer(body)
            return completion(body, lambda _: "unused")
        result, _ = await run(unused_check_and_store, "text", *wire_backends(handler))
        self.assertEqual(result.value, {"value": "original"}, result.error)

    async def test_cancelled_selected_work_keeps_shared_slot_until_completion(self):
        started, release = asyncio.Event(), asyncio.Event()
        pool = RequestSlots(generation=1, decision=1)
        async def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                return decision_answer(body, selected="no")
            started.set()
            await release.wait()
            return completion(body, lambda _: "unused")
        decision, generation = wire_backends(handler)
        runtime = LevRuntime(decision, generation, request_slots=pool, trace=True)
        try:
            stream = await runtime.submit(lower_to_plan(compile_program(unused_transport)), [RunInput(0, "text")], mode="speculative")
            await asyncio.wait_for(started.wait(), 5)
            await stream.aclose()
            self.assertEqual(pool.active["llm"], 1)
            closing = asyncio.create_task(runtime.aclose())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertEqual(pool.active["llm"], 1)
            release.set()
            await asyncio.wait_for(closing, 5)
            self.assertEqual(pool.active, {"decision": 0, "llm": 0})
        finally:
            release.set()
            await runtime.aclose()

    async def test_unselected_transport_is_never_called(self):
        for selected in ("yes", "no"):
            generated = []
            async def handler(request):
                body = json.loads(request.content)
                if "questions" in body:
                    return decision_answer(body, selected=selected)
                generated.append(body)
                return httpx.Response(503)
            result, _ = await run(unused_transport, "text", *wire_backends(handler))
            self.assertEqual(result.status, "success" if selected == "yes" else "error")
            if selected == "yes":
                self.assertEqual(result.value, "done")
            self.assertEqual(len(generated), int(selected == "no"))

    async def test_context_limits_split_or_reject_using_actual_payload(self):
        for input_tokens, count, status in ((800, 2, "success"), (961, 0, "error")):
            requests = []
            def handler(request):
                body = json.loads(request.content)
                if "questions" in body:
                    return decision_answer(body)
                requests.append(body)
                return completion(body, lambda c: c.arguments["instructions"])
            transport = httpx.MockTransport(handler)
            decision = DecisionAPIBackend("http://offline.invalid", "fixture", "fixture", transport=transport)
            generation = FixtureLLMBackend("http://offline.invalid", "fixture", "fixture", profile="qwen_vllm",
                request_policy=policy(1024), _token_counter=lambda _: input_tokens, transport=transport)
            result, _ = await run(batch_arms, "input", decision, generation)
            self.assertEqual(result.status, status, result.error)
            self.assertEqual(len(requests), count)
            if status == "error":
                self.assertEqual(result.error.code, "context_length_exceeded")
            for body in requests:
                self.assertEqual(body["messages"][1]["content"], "input")
                self.assertEqual(body["max_tokens"], 64)

    async def test_fatal_service_error_is_not_suppressed(self):
        async def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                return decision_answer(body, selected="no")
            return httpx.Response(401)
        from lev.backends import BackendFatalError
        with self.assertRaises(BackendFatalError):
            await run(unused_transport, "text", *wire_backends(handler))

    async def test_real_data_dependency_waits_for_response(self):
        generated = []
        def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                return decision_answer(body)
            def answer(call):
                name = call.arguments["instructions"]
                generated.append(name)
                return "response" if name == "first" else name
            return completion(body, answer)
        result, _ = await run(nested_and_chain, "text", *wire_backends(handler))
        self.assertEqual(result.value, "response next", result.error)
        self.assertEqual(generated, ["first", "response next"])

    async def test_decision_phase_waits_for_every_inflight_batch(self):
        entered, release = asyncio.Event(), asyncio.Event()
        decisions, generated = [], []
        async def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                self.assertFalse(generated)
                decisions.append(body)
                if len(decisions) == 2:
                    entered.set()
                    await release.wait()
                return decision_answer(body)
            self.assertTrue(release.is_set())
            generated.append(body)
            return completion(body, lambda c: "response" if c.arguments["instructions"] == "first" else c.arguments["instructions"])
        decision, generation = wire_backends(handler)
        task = asyncio.create_task(run(nested_and_chain, "text", decision, generation,
                                       max_batch_questions=1, max_decision_inflight=2))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            self.assertFalse(generated)
        finally:
            release.set()
            result, trace = await asyncio.wait_for(task, 5)
        self.assertEqual(result.value, "response next", result.error)
        self.assertEqual(len(decisions), 2)
        submitted = [e for e in trace if e["event"] == "http_submitted" and e["backend"] == "decision"]
        self.assertEqual([len(e["speculative_calls"]) for e in submitted], [0, 1])

    async def test_alternating_dependencies_repeat_decision_then_generation(self):
        order = []
        def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                order.append("decision")
                return decision_answer(body)
            order.append("llm")
            return completion(body, lambda c: c.arguments["instructions"])
        result, _ = await run(alternating, "text", *wire_backends(handler), local_task_budget=1)
        self.assertEqual(result.value, "wanted", result.error)
        self.assertEqual(order, ["decision", "llm", "decision", "llm"])

    async def test_generation_waits_for_unknown_qwen_gate(self):
        for selected in ("yes", "no"):
            generated = []
            def handler(request):
                body = json.loads(request.content)
                self.assertNotIn("questions", body)
                self.assertEqual(len(llm_calls(body)[1]), 1)
                def answer(call):
                    name = call.arguments["instructions"]
                    generated.append(name)
                    return selected if name == "predicate" else name
                return completion(body, answer)
            result, _ = await run(qwen_gate, "text", *wire_backends(handler), local_task_budget=1)
            expected = "wanted" if selected == "yes" else "unused"
            self.assertEqual(result.value, expected, result.error)
            self.assertEqual(generated, ["predicate", expected])

    async def test_private_branch_and_merge_do_not_wait_for_formal_branch(self):
        for selected in ("yes", "no"):
            batches = []
            def handler(request):
                body = json.loads(request.content)
                if "questions" in body:
                    return decision_answer(body, selected=selected)
                batches.append([c.arguments["instructions"] for _, c in llm_calls(body)[1]])
                return completion(body, lambda c: c.arguments["instructions"])
            plan = lower_to_plan(compile_program(delayed_branch))
            first = next(t for t in plan.tasks if isinstance(t, p.Infer))
            branch = next(t for t in plan.tasks if isinstance(t, p.Branch))
            # A conservative formal control edge delays Branch and Merge until
            # the first Qwen response; its pure predicate is already knowable.
            plan = replace(plan, dependencies=plan.dependencies + (
                p.Dependency(first.id, branch.id, "control", p.InstanceMap()),))
            async with LevRuntime(*wire_backends(handler), trace=True, local_task_budget=1) as runtime:
                result, = await asyncio.wait_for(collect(runtime, plan, "text"), 5)
            expected = ["independent", "wanted" if selected == "yes" else "unused"]
            self.assertEqual(result.value, expected, result.error)
            self.assertEqual(batches, [expected])

    async def test_slow_input_decisions_do_not_block_other_input_generation(self):
        slow_started, fast_generated = asyncio.Event(), asyncio.Event()
        order = []
        async def handler(request):
            body = json.loads(request.content)
            if "questions" in body:
                if body["state"] == "slow":
                    slow_started.set()
                    await fast_generated.wait()
                else:
                    await slow_started.wait()
                order.append((body["state"], "decision"))
                return decision_answer(body)
            state = llm_calls(body)[1][0][0].state
            order.append((state, "llm"))
            if state == "fast":
                fast_generated.set()
            return completion(body, lambda c: c.arguments["instructions"])
        async with LevRuntime(*wire_backends(handler), max_active_runs=2) as runtime:
            stream = await runtime.submit(lower_to_plan(compile_program(arms)),
                [RunInput(0, "slow"), RunInput(1, "fast")], mode="speculative")
            async def consume():
                return [r async for r in stream]
            results = await asyncio.wait_for(consume(), 5)
        self.assertEqual([r.value for r in results], ["wanted", "wanted"])
        self.assertLess(order.index(("fast", "llm")), order.index(("slow", "decision")))

    async def test_gated_loop_is_selected_and_bounded(self):
        for selected in ("yes", "no"):
            generated, sizes = [], []
            def handler(request):
                body = json.loads(request.content)
                if "questions" in body:
                    return decision_answer(body, selected=selected)
                sizes.append(len(llm_calls(body)[1]))
                def answer(call):
                    name = call.arguments["instructions"]
                    generated.append(name)
                    return name
                return completion(body, answer)
            state = ["a", "b", "c", "d", "e"]
            result, _ = await run(gated_loop, state, *wire_backends(handler), loop_window=2, local_task_budget=1)
            self.assertEqual(result.value, state if selected == "yes" else [], result.error)
            self.assertEqual(generated, result.value)
            self.assertTrue(all(n <= 2 for n in sizes))


    def test_viewer_retains_formal_edges_and_lists_candidates(self):
        plan = lower_to_plan(compile_program(arms))
        before = plan.to_dict()
        eager = analyze_plan(plan)
        report = analyze_plan(plan, mode="speculative")
        self.assertEqual(report["edges"], eager["edges"])
        self.assertEqual(len(report["speculative_candidates"]), 3)
        self.assertIn("推测候选", format_plan(plan, mode="speculative"))
        self.assertIn('value="speculative"', format_plan_html(plan, mode="speculative"))
        self.assertEqual(plan.to_dict(), before)
