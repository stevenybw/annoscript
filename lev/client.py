"""Local or session-owned remote client, with compatible shared contract exports."""

from __future__ import annotations

from collections.abc import Callable
from typing import List

from lev.lev import Lev
from lev._client_common import (
    JSONValue, State, R, InputCheck, InputValidationError,
    Result, ResultError, ResultStream, require, precondition, system_prompt,
    get_program_config, validate_inputs, prepare_inputs,
)


class LevClient:
    """Synchronous local Runtime owner or simplified remote job client.

    Construction performs no network work. A private event-loop thread starts
    on the first valid submission and is shared by this client's submissions.
    The local client owns its runtime. ``remote()`` owns a remote job session.
    Use a context manager or close() to release ownership explicitly.
    """

    @property
    def closed(self):
        return self._remote.closed if hasattr(self, "_remote") else self._closed

    @closed.setter
    def closed(self, value):
        self._closed = value

    def __init__(self, runtime, *, mode="eager"):
        from lev.runtime import LevRuntime
        if not isinstance(runtime, LevRuntime):
            raise TypeError("runtime must be a LevRuntime")
        if runtime._loop is not None or runtime.closed or getattr(runtime, "_client_owner", None) is not None:
            raise ValueError("client requires an unstarted runtime")
        if mode not in ("eager", "strict", "speculative"):
            raise ValueError("mode must be eager, strict or speculative")
        self.runtime, self.mode = runtime, mode
        runtime._client_owner = self
        self._loop = self._thread = None
        self._streams = set()
        self.closed = False

    @classmethod
    def remote(cls, base_url, auth_token=None, *, mode="eager", optimize=True,
               timeout=30.0, transport=None):
        """Use session-owned durable jobs with in-memory reconnect state.

        Closing this client cancels unfinished remote jobs. A new process does
        not recover this session; use LevJobClient for explicit durable recovery.
        """
        import weakref
        from lev._remote_jobs import RemoteAdapter, _finalize
        client = cls.__new__(cls)
        client._remote = RemoteAdapter(base_url, auth_token, mode=mode, optimize=optimize,
                                       timeout=timeout, transport=transport)
        client.mode, client.closed = mode, False
        client._finalizer = weakref.finalize(client, _finalize, client._remote.session)
        return client

    @classmethod
    def from_env(cls, *, env_file=None, mode="eager", disable_thinking=True,
                 generation_profile=None, decision_request_adapter=None, generation_request_adapter=None,
                 decision_request_adapter_options=None,
                 batch_rule_dedup=False, assumed_avg_num_tokens_per_char=2,
                 timeout=60.0, **runtime_options):
        if "source_layout" in runtime_options:
            raise TypeError("source_layout was removed; extract now uses source_field")
        from lev.backends.config import backends_from_env
        from lev.runtime import LevRuntime
        decision, llm = backends_from_env(env_file=env_file, disable_thinking=disable_thinking,
                                         generation_profile=generation_profile, batch_rule_dedup=batch_rule_dedup, timeout=timeout,
                                         decision_request_adapter=decision_request_adapter,
                                         decision_request_adapter_options=decision_request_adapter_options,
                                         generation_request_adapter=generation_request_adapter,
                                         assumed_avg_num_tokens_per_char=assumed_avg_num_tokens_per_char)
        return cls(LevRuntime(decision, llm, **runtime_options), mode=mode)

    def _start(self):
        if self._loop is None:
            import asyncio
            from threading import Thread, Event
            self._loop = asyncio.new_event_loop()
            started = Event()
            def serve():
                asyncio.set_event_loop(self._loop)
                self._loop.call_soon(started.set)
                self._loop.run_forever()
                self._loop.close()
            self._thread = Thread(target=serve, name="lev-runtime", daemon=True)
            self._thread.start()
            started.wait()

    def _call(self, coroutine):
        import asyncio
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("client is closed")
        return self

    def __exit__(self, *args):
        if args and args[0] is not None:
            try:
                self.close()
            except BaseException:
                import logging
                logging.getLogger(__name__).warning("Client cleanup failed while handling another exception")
        else:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "_remote"):
            try:
                self._remote.close()
            finally:
                self._finalizer.detach()
            return
        try:
            for stream in list(self._streams):
                stream.close()
            if self._loop is not None:
                self._call(self.runtime.aclose())
            else:
                import asyncio
                asyncio.run(self.runtime.aclose())
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()

    _get_program_config = staticmethod(get_program_config)
    _validate_inputs = staticmethod(validate_inputs)

    def submit(
        self,
        program: Callable[[Lev], R],
        inputs: List[State],
        *,
        ordered: bool = False,
    ) -> ResultStream[R]:
        """Apply a single-input eDSL function to a list of input states.

        For each input, supply ``program`` with an independent ``Lev`` bound to
        that state and the execution configuration. All Lev operations in the
        program must target this supplied instance; programs must not construct
        additional Lev instances or switch to derived contexts. Model execution
        and constraint handling belong to the eDSL runtime.

        Include ``_get_program_config(program)`` as remote program configuration.
        The runtime supplies its ``system_prompt`` to each injected Lev and uses
        it for LLM calls, including those in helpers. Decision requests use their
        own instructions and requirements. Keep the prompt separate from input
        states and per-call instructions. The prompt text is transmitted;
        the decorator declaration itself is not part of remote executable code.

        ``inputs`` must be a list of strings, JSON arrays, or JSON objects.
        An empty list produces an empty result stream. Input and output JSON
        values must use string object keys and finite numbers.

        Run all registered ``precondition`` checks locally with
        ``_validate_inputs`` before any remote submission. Return failed inputs
        as error results with code ``invalid_input`` and their original indices;
        do not create a Lev context or remote task for those inputs. Submit only
        inputs that passed, and merge local errors into the same ResultStream,
        respecting ``ordered``. An unexpected checker exception aborts the whole
        submission before any remote work. Local check functions, rules, and
        decorator declarations must not be included in the remote program.

        Preserve each program return value in ``Result.value``, without encoding
        it as a JSON string or flattening lists. Result indices are the zero-based
        positions in ``inputs``; retries retain the same index. Normal completion
        produces one success or error result for every input, without duplicate
        delivery within the stream. Early closure or a submission-level failure
        may leave inputs without delivered results.

        By default deliver results as they become available. ``ordered=True``
        delivers them in input order. Scheduling and concurrency are left to
        concrete implementations.

        Per-input execution failures are error results, so other inputs may
        continue. Invalid submission configuration, program errors preventing
        submission, authentication failures, and unrecoverable transport failures
        raise exceptions.
        Once a stream is returned, fatal iteration errors close it automatically.

        Use the returned stream as a context manager to release resources even
        when the consumer stops early or raises an exception.
        """
        if hasattr(self, "_remote"):
            if self.closed:
                raise RuntimeError("client is closed")
            return self._remote.submit(program, inputs, ordered=ordered)
        from lev.compiler import compile_program
        from lev.plan_lowering import lower_to_plan
        from lev.runtime import RunInput
        if self.closed:
            raise RuntimeError("client is closed")
        prepared, errors = prepare_inputs(program, inputs)
        valid = [RunInput(index, state) for index, state in prepared]
        plan = lower_to_plan(compile_program(program))
        self._start()
        source = self._call(self.runtime.submit(plan, valid, mode=self.mode))
        stream = _ClientStream(self, source, errors, ordered)
        self._streams.add(stream)
        return stream


class _ClientStream(ResultStream):
    def __init__(self, client, source, errors, ordered):
        super().__init__()
        self.client, self.source, self.ordered = client, source, ordered
        self.local = iter(Result(i, "error", error=e) for i, e in sorted(errors.items()))
        self.buffer, self.index = {}, 0

    def _pull(self):
        try:
            return next(self.local)
        except StopIteration:
            try:
                return self.client._call(self.source.__anext__())
            except StopAsyncIteration:
                raise StopIteration from None

    def _next_result(self):
        if not self.ordered:
            return self._pull()
        while self.index not in self.buffer:
            result = self._pull()
            self.buffer[result.index] = result
        result = self.buffer.pop(self.index)
        self.index += 1
        return result

    def _close(self):
        try:
            self.client._call(self.source.aclose())
        finally:
            self.buffer.clear()
            self.client._streams.discard(self)
