"""Lease-owning worker with isolated Runtime instances for individual attempts."""

import asyncio
import os
import random
import time
import uuid

from lev.backends.base import HTTPBackend
from lev.server.metrics import RequestMeter
from lev.server._plan_cache import PlanCache
from lev._dispatch_pool import RequestSlots

from lev._client_common import Result, ResultError
from lev.backends import BackendFatalError
from lev.ir import Program
from lev.job_protocol import encode, retry_after_seconds
from lev.passes import optimize_program
from lev.plan_lowering import lower_to_plan
from lev.runtime import LevRuntime, RunInput


def backends(manifest):
    from lev.backends.decision import DecisionAPIBackend
    from lev.backends.llm import LLMAPIBackend
    try:
        decision_token = os.environ["LEV_DECISION_AUTH_TOKEN"]
        generation_token = os.environ["LEV_GENERATE_AUTH_TOKEN"]
    except KeyError:
        raise BackendFatalError("model credentials are unavailable", code="configuration_failed") from None
    timeout = manifest["policy"]["model_timeout"]
    d, g = manifest["decision"], manifest["generation"]
    return (DecisionAPIBackend(d["base_url"], decision_token, d["model"], timeout=timeout),
            LLMAPIBackend(g["base_url"], generation_token, g["model"], timeout=timeout,
                          profile=g["profile"], extract_mode=g.get("extract_mode", "validated"),
                          batch_rule_dedup=g.get("batch_rule_dedup", False),
                          disable_thinking=manifest["disable_thinking"], request_policy=g.get("request_policy"),
                          assumed_avg_num_tokens_per_char=g.get("assumed_avg_num_tokens_per_char", 2)))


class ObservedBackend:
    def __init__(self, backend, failures, owned, *, dispatch_allowed=None, on_failure=None):
        self.backend, self.failures, self.owned = backend, failures, owned
        self.dispatch_allowed = dispatch_allowed
        self.on_failure = on_failure
        if isinstance(backend, HTTPBackend):
            backend._dispatch_guard = self._check_dispatch

    def _check_dispatch(self):
        if not self.owned():
            raise asyncio.CancelledError()
        if self.dispatch_allowed is not None and not self.dispatch_allowed():
            raise BackendFatalError("job is paused; no new model request was sent", code="job_paused")

    @property
    def supports_llm_batch(self):
        return getattr(self.backend, "supports_llm_batch", False) is True

    def plan_batch(self, *args, **kwargs):
        return self.backend.plan_batch(*args, **kwargs)

    @property
    def supports_prepared_batch(self):
        return getattr(self.backend, "supports_prepared_batch", False) is True

    def resolve_token_budget(self, call, fallback):
        resolve = getattr(self.backend, "resolve_token_budget", None)
        return resolve(call, fallback) if resolve else (call.arguments.get("max_num_tokens") or fallback)

    async def infer(self, *args, **kwargs):
        self._check_dispatch()
        try:
            return await self.backend.infer(*args, **kwargs)
        except Exception as exc:
            self.failures.append(exc)
            if self.on_failure is not None:
                self.on_failure(exc)
            raise

    async def infer_batch(self, *args, **kwargs):
        self._check_dispatch()
        try:
            return await self.backend.infer_batch(*args, **kwargs)
        except Exception as exc:
            self.failures.append(exc)
            if self.on_failure is not None:
                self.on_failure(exc)
            raise

    async def aclose(self):
        await self.backend.aclose()


def classification(code, metadata, *, retry_unknown_outcomes=True):
    status = metadata.get("http_status")
    if code in ("authentication_failed", "configuration_failed", "backend_protocol_error",
                "job_paused", "metrics_failed", "diagnostics_failed") or status in (401, 403):
        return "pause"
    if not retry_unknown_outcomes and status is None and code in ("model_timeout", "transport_error"):
        return "pause"
    if (code in ("backend_unavailable", "model_timeout", "transport_error")
            or status in (408, 429) or (type(status) is int and 500 <= status <= 599)):
        return "retry"
    if code == "runtime_failed":
        return "fatal"
    return "result"


class Worker:
    def __init__(self, store, releases, *, backend_factory=backends, concurrency=8,
                 poll_interval=0.2, random_source=random.random, request_observer=None,
                 diagnostic_observer=None, model_request_limits=None):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("worker concurrency must be positive")
        self.store, self.releases, self.backend_factory = store, releases, backend_factory
        self.concurrency, self.poll_interval, self.random = concurrency, poll_interval, random_source
        self.request_observer = request_observer
        self.diagnostic_observer = diagnostic_observer
        self.id = uuid.uuid4().hex
        self.plan_cache = PlanCache()
        if model_request_limits is not None and (type(model_request_limits) is not dict
                or set(model_request_limits) != {"generation", "decision"}):
            raise ValueError("model_request_limits requires generation and decision")
        self.request_slots = RequestSlots(**model_request_limits) if model_request_limits is not None else None

    async def execute(self, claim):
        manifest = self.releases.manifests[claim["release_id"]]
        failures = []
        started = time.monotonic()

        async def run():
            def diagnose(stage, event):
                if self.diagnostic_observer is not None:
                    try:
                        self.diagnostic_observer({"job_id": claim["job_id"], "index": claim["index"],
                            "attempt": claim["attempt"], "stage": stage, **event})
                    except Exception:
                        raise BackendFatalError("diagnostic persistence failed", code="diagnostics_failed") from None
            def failed(exc):
                code, metadata = getattr(exc, "code", "execution_failed"), getattr(exc, "metadata", {})
                if classification(code, metadata, retry_unknown_outcomes=manifest["policy"].get("retry_unknown_outcomes", True)) == "pause":
                    self.store.pause_dispatch(claim, {"code": code, "message": str(exc)})
            if self.request_slots is not None:
                diagnose("scheduling", {"event": "resource_limits", "active_inputs": self.concurrency,
                    "limits": self.request_slots.limits, "algorithm": "ready_at_admission_fifo_v1"})
            def compile_plan(encoded):
                program = Program.from_json(encoded.decode())
                if manifest["server_optimize"]:
                    program = optimize_program(program)
                return lower_to_plan(program)
            plan, preparation = await self.plan_cache.get(claim["release_id"], claim["ir"], compile_plan)
            diagnose("preparation", preparation)
            try:
                d, g = self.backend_factory(manifest)
            except ValueError:
                raise BackendFatalError("invalid backend configuration", code="configuration_failed") from None
            supported = isinstance(d, HTTPBackend) and isinstance(g, HTTPBackend) and d is not g
            if not self.store.meter_attempt(claim, supported):
                raise asyncio.CancelledError()
            if isinstance(d, HTTPBackend):
                d._request_meter = RequestMeter(self.store, claim, "decision", observer=self.request_observer)
                if self.diagnostic_observer is not None:
                    d.diagnostic_observer = lambda event: diagnose("decision", event)
            if isinstance(g, HTTPBackend) and d is not g:
                g._request_meter = RequestMeter(self.store, claim, "generation", observer=self.request_observer)
                if self.diagnostic_observer is not None:
                    g.diagnostic_observer = lambda event: diagnose("generation", event)
            decision = ObservedBackend(d, failures, lambda: self.store.renew(claim),
                                       dispatch_allowed=lambda: self.store.can_dispatch(claim),
                                       on_failure=failed if self.request_slots is not None else None)
            generation = decision if d is g else ObservedBackend(g, failures, lambda: self.store.renew(claim),
                                                                 dispatch_allowed=lambda: self.store.can_dispatch(claim),
                                                                 on_failure=failed if self.request_slots is not None else None)
            runtime = LevRuntime(decision, generation, trace=self.diagnostic_observer is not None,
                                 request_slots=self.request_slots, **manifest["runtime"])
            try:
                async with runtime:
                    async with await runtime.submit(plan, [RunInput(claim["index"], claim["state"])], mode=claim["mode"]) as stream:
                        result = await anext(stream)
                        # Verify one and only one outcome for this isolated input.
                        try:
                            await anext(stream)
                        except StopAsyncIteration:
                            return result
                        raise RuntimeError("multiple results for an input attempt")
            finally:
                if self.diagnostic_observer is not None:
                    diagnose("runtime", {"event": "trace", "trace": runtime.trace,
                                         "plan": plan.to_dict()})

        task = asyncio.create_task(run())

        async def renew():
            try:
                while True:
                    await asyncio.sleep(self.store.lease_seconds / 4)
                    if not self.store.renew(claim):
                        task.cancel()
                        return
            except Exception:
                task.cancel()

        heartbeat = asyncio.create_task(renew())
        try:
            try:
                async with asyncio.timeout(manifest["policy"]["input_timeout"]):
                    result = await task
                if result.status == "success":
                    self.store.finish(claim, result, elapsed_seconds=time.monotonic() - started)
                    return
                code, message = result.error.code, result.error.message
                original = next((e for e in failures if getattr(e, "code", None) == code), None)
                metadata = getattr(original, "metadata", {}) if original is not None else {}
            except TimeoutError:
                self.store.finish(claim, Result(claim["index"], "error", error=ResultError("execution_timeout", "input execution time limit exceeded")), elapsed_seconds=time.monotonic() - started)
                return
            except BackendFatalError as exc:
                code, message, metadata = exc.code, str(exc), exc.metadata
            except asyncio.CancelledError:
                # Cancelled/expired ownership cannot commit anything. Process
                # shutdown leaves a lease that another worker can reclaim.
                raise
            except Exception:
                self.store.fail_job(claim, {"code": "execution_failed", "message": "job lowering or runtime failed"}, elapsed_seconds=time.monotonic() - started)
                return
            error = {"code": code, "message": message}
            kind = classification(code, metadata,
                                  retry_unknown_outcomes=manifest["policy"].get("retry_unknown_outcomes", True))
            if kind in ("pause", "retry"):
                delay = max(self.random() * min(30.0, 2 ** (claim["round_attempt"] - 1)),
                            retry_after_seconds(metadata.get("retry_after"), self.store.clock()))
                self.store.attempt_failed(claim, error, pause=kind == "pause", delay=delay, metadata=metadata, elapsed_seconds=time.monotonic() - started)
            elif kind == "fatal":
                self.store.fail_job(claim, error, elapsed_seconds=time.monotonic() - started)
            else:
                self.store.finish(claim, Result(claim["index"], "error", error=ResultError(**error)), elapsed_seconds=time.monotonic() - started)
        finally:
            heartbeat.cancel()
            task.cancel()
            await asyncio.gather(heartbeat, task, return_exceptions=True)

    async def run(self, *, stop=None):
        active = set()
        stop_waiter = asyncio.create_task(stop.wait()) if stop is not None else None
        try:
            while stop is None or not stop.is_set():
                self.store.maintain()
                for task in list(active):
                    if task.done():
                        active.remove(task)
                        if not task.cancelled():
                            task.result()
                while len(active) < self.concurrency:
                    claim = self.store.claim(self.id, self.releases)
                    if claim is None:
                        break
                    active.add(asyncio.create_task(self.execute(claim)))
                waiting = active | ({stop_waiter} if stop_waiter is not None else set())
                if waiting:
                    await asyncio.wait(waiting, timeout=self.poll_interval, return_when=asyncio.FIRST_COMPLETED)
                else:
                    await asyncio.sleep(self.poll_interval)
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            if stop_waiter is not None:
                stop_waiter.cancel()
                await asyncio.gather(stop_waiter, return_exceptions=True)
            await self.aclose()

    async def aclose(self):
        await self.plan_cache.aclose()
