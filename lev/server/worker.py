"""Lease-owning worker with isolated Runtime instances for individual attempts."""

import asyncio
import os
import random
import time
import uuid

from lev.backends.base import HTTPBackend
from lev.server.metrics import RequestMeter

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
                          profile=g["profile"], disable_thinking=manifest["disable_thinking"]))


class ObservedBackend:
    def __init__(self, backend, failures, owned):
        self.backend, self.failures, self.owned = backend, failures, owned

    async def infer(self, *args):
        if not self.owned():
            raise asyncio.CancelledError()
        try:
            return await self.backend.infer(*args)
        except Exception as exc:
            self.failures.append(exc)
            raise

    async def infer_batch(self, *args):
        if not self.owned():
            raise asyncio.CancelledError()
        try:
            return await self.backend.infer_batch(*args)
        except Exception as exc:
            self.failures.append(exc)
            raise

    async def aclose(self):
        await self.backend.aclose()


def classification(code, metadata):
    status = metadata.get("http_status")
    if code in ("authentication_failed", "configuration_failed") or status in (401, 403):
        return "pause"
    if (code in ("backend_unavailable", "model_timeout", "transport_error")
            or status in (408, 429) or (type(status) is int and 500 <= status <= 599)):
        return "retry"
    if code == "runtime_failed":
        return "fatal"
    return "result"


class Worker:
    def __init__(self, store, releases, *, backend_factory=backends, concurrency=8,
                 poll_interval=0.2, random_source=random.random):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("worker concurrency must be positive")
        self.store, self.releases, self.backend_factory = store, releases, backend_factory
        self.concurrency, self.poll_interval, self.random = concurrency, poll_interval, random_source
        self.id = uuid.uuid4().hex

    async def execute(self, claim):
        manifest = self.releases.manifests[claim["release_id"]]
        failures = []
        started = time.monotonic()

        async def run():
            program = Program.from_json(encode(claim["ir"]).decode())
            if manifest["server_optimize"]:
                program = optimize_program(program)
            plan = lower_to_plan(program)
            try:
                d, g = self.backend_factory(manifest)
            except ValueError:
                raise BackendFatalError("invalid backend configuration", code="configuration_failed") from None
            supported = isinstance(d, HTTPBackend) and isinstance(g, HTTPBackend) and d is not g
            if not self.store.meter_attempt(claim, supported):
                raise asyncio.CancelledError()
            if isinstance(d, HTTPBackend):
                d._request_meter = RequestMeter(self.store, claim, "decision")
            if isinstance(g, HTTPBackend) and d is not g:
                g._request_meter = RequestMeter(self.store, claim, "generation")
            decision = ObservedBackend(d, failures, lambda: self.store.renew(claim))
            generation = decision if d is g else ObservedBackend(g, failures, lambda: self.store.renew(claim))
            async with LevRuntime(decision, generation, **manifest["runtime"]) as runtime:
                async with await runtime.submit(plan, [RunInput(claim["index"], claim["state"])], mode=claim["mode"]) as stream:
                    result = await anext(stream)
                    # Verify one and only one outcome for this isolated input.
                    try:
                        await anext(stream)
                    except StopAsyncIteration:
                        return result
                    raise RuntimeError("multiple results for an input attempt")

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
            kind = classification(code, metadata)
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
                await asyncio.sleep(self.poll_interval)
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
