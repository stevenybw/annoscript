"""Batch submission and result-stream contracts for Lev eDSL programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import FunctionType, TracebackType
from typing import Generic, List, Literal, TypeAlias, TypeVar
from weakref import WeakKeyDictionary

from lev.lev import Lev


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
State: TypeAlias = str | list[JSONValue] | dict[str, JSONValue]
R = TypeVar("R", bound=JSONValue)
InputCheck: TypeAlias = Callable[[State], None]

# Keep local checks outside the program function's attributes and execution body.
_PRECONDITIONS: WeakKeyDictionary[
    Callable[[Lev], JSONValue], tuple[InputCheck, ...]
] = WeakKeyDictionary()
# Only this registry's text values are exported as remote program configuration.
_SYSTEM_PROMPTS: WeakKeyDictionary[Callable[[Lev], JSONValue], str] = WeakKeyDictionary()


class InputValidationError(ValueError):
    """An input failed a client-local precondition, before remote submission."""


def require(condition: bool, *, message: str) -> None:
    """Check an input condition locally; raise InputValidationError on failure.

    This ordinary Python function remains active under optimization. It never
    invokes inference, repairs inputs, or propagates constraints to the eDSL.
    """
    if not condition:
        raise InputValidationError(message)


def precondition(
    check: InputCheck,
) -> Callable[[Callable[[Lev], R]], Callable[[Lev], R]]:
    """Register a client-local input check and return the original program.

    Checks are synchronous, deterministic, read-only Python functions. Return
    None for success and raise InputValidationError for an invalid input. Other
    exceptions indicate a checker defect and abort submission. Checks must not
    invoke models or make network requests.

    Registration does not run checks or wrap the program. Direct program calls
    do not run checks; LevClient implementations run them before any remote
    submission. Stacked decorators are checked in source order (top to bottom),
    stopping at the first failed check for each input.

    The registry is local only. Remote program packaging must exclude these
    decorator declarations, check functions, and their validation dependencies.
    """
    if not callable(check):
        raise TypeError("precondition requires a callable input check")

    def register(program: Callable[[Lev], R]) -> Callable[[Lev], R]:
        if not isinstance(program, FunctionType):
            raise TypeError("precondition must decorate a Python function")
        _PRECONDITIONS[program] = (check,) + _PRECONDITIONS.get(program, ())
        return program

    return register


def system_prompt(
    prompt: str,
) -> Callable[[Callable[[Lev], R]], Callable[[Lev], R]]:
    """Declare shared model instructions and return the original program.

    The client exports this text as program configuration. The runtime binds it
    to every record's injected Lev and applies it to every model operation,
    including calls inside helper functions. Per-call instructions describe the
    specific task; API guarantees and hard constraints remain independently
    enforced. Mapping the shared instructions to a model is the runtime's job.

    Registration neither wraps nor executes the program and creates no context.
    Direct Python calls do not inject the prompt. Each program may declare one
    system prompt; any repeated declaration raises ValueError. Empty strings
    are allowed and preserved. Prompt text is sent as configuration, without
    executing the decorator remotely or exporting local precondition checks.
    """
    if not isinstance(prompt, str):
        raise TypeError("system_prompt requires a string")

    def register(program: Callable[[Lev], R]) -> Callable[[Lev], R]:
        if not isinstance(program, FunctionType):
            raise TypeError("system_prompt must decorate a Python function")
        if program in _SYSTEM_PROMPTS:
            raise ValueError("a program may declare only one system_prompt")
        _SYSTEM_PROMPTS[program] = prompt
        return program

    return register


@dataclass(frozen=True)
class ResultError:
    """A per-input failure, including local validation or unsatisfied constraints."""

    code: str
    message: str


@dataclass(frozen=True)
class Result(Generic[R]):
    """One terminal outcome, identified by its zero-based input index.

    Success preserves the program's return value, including ``None``. An error
    carries no value. A list-valued result stays one result, not multiple items.
    """

    index: int
    status: Literal["success", "error"]
    value: R | None = None
    error: ResultError | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a nonnegative integer")
        if self.status == "success":
            if self.error is not None:
                raise ValueError("a successful result cannot contain an error")
        elif self.status == "error":
            if not isinstance(self.error, ResultError) or self.value is not None:
                raise ValueError("an error result requires ResultError and no value")
        else:
            raise ValueError("status must be 'success' or 'error'")


class ResultStream(Iterator[Result[R]], ABC):
    """A single-pass result iterator with explicit resource ownership.

    Use ``with`` when stopping early: breaking a plain ``for`` loop does not
    close the stream. Exhaustion and iteration errors automatically close it.
    Concrete transports implement ``_next_result`` and ``_close`` and call
    ``super().__init__()`` when overriding construction.

    Streams are intended for consumption and closure by a single thread.
    """

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> ResultStream[R]:
        return self

    def __next__(self) -> Result[R]:
        if self._closed:
            raise StopIteration
        try:
            return self._next_result()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Stop receiving results and release resources; repeated calls are harmless.

        Closing discards undelivered results. No further work may be submitted.
        The transport should request cancellation of pending
        work, without promising that already-running remote work can be stopped.
        The stream stays closed even if transport cleanup raises an exception.
        """
        if not self._closed:
            self._closed = True
            self._close()

    def __enter__(self) -> ResultStream[R]:
        if self._closed:
            raise RuntimeError("cannot enter a closed result stream")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    @abstractmethod
    def _next_result(self) -> Result[R]:
        """Return the next result, or raise StopIteration when fully exhausted.

        Return per-input failures as error results. Raise submission-level or
        transport failures as exceptions; the stream will then close itself.
        """
        raise NotImplementedError

    @abstractmethod
    def _close(self) -> None:
        """Release local resources and request cancellation, at most once.

        This hook must not submit new work or wait for unfinished work to
        complete. It is also called after normal exhaustion or iteration errors.
        """
        raise NotImplementedError


class LevClient:
    """Synchronous owner of a previously unstarted LevRuntime.

    Construction performs no network work. A private event-loop thread starts
    on the first valid submission and is shared by this client's submissions.
    The client takes ownership of its runtime; use a context manager or close().
    """

    def __init__(self, runtime, *, mode="eager"):
        from lev.runtime import LevRuntime
        if not isinstance(runtime, LevRuntime):
            raise TypeError("runtime must be a LevRuntime")
        if runtime._loop is not None or runtime.closed or getattr(runtime, "_client_owner", None) is not None:
            raise ValueError("client requires an unstarted runtime")
        if mode not in ("eager", "strict"):
            raise ValueError("mode must be eager or strict")
        self.runtime, self.mode = runtime, mode
        runtime._client_owner = self
        self._loop = self._thread = None
        self._streams = set()
        self.closed = False

    @classmethod
    def from_env(cls, *, env_file=None, mode="eager", disable_thinking=True,
                 timeout=60.0, **runtime_options):
        from lev.backends.config import backends_from_env
        from lev.runtime import LevRuntime
        decision, llm = backends_from_env(env_file=env_file, disable_thinking=disable_thinking, timeout=timeout)
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
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
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

    @staticmethod
    def _get_program_config(program: Callable[[Lev], R]) -> dict[str, JSONValue]:
        """Return fresh, JSON-compatible execution settings for remote submission.

        Export system-prompt text only, never local check functions or rules.
        Programs without a declaration have no system_prompt configuration key.
        Reading configuration does not execute the program or its preconditions.
        """
        prompt = _SYSTEM_PROMPTS.get(program)
        return {} if prompt is None else {"system_prompt": prompt}

    @staticmethod
    def _validate_inputs(
        program: Callable[[Lev], R], inputs: List[State]
    ) -> dict[int, ResultError]:
        """Run the whole batch's preconditions locally, without running program.

        Call before any remote submission. Return invalid-input errors keyed by
        original list index; the remaining indices passed all registered checks.
        Only InputValidationError becomes an item error. Unexpected exceptions
        and non-None check returns abort validation and must prevent submission.

        This helper validates the outer list and program-specific preconditions.
        Concrete clients also validate the generic JSON input contract.
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list")
        checks = _PRECONDITIONS.get(program, ())
        errors = {}
        for index, state in enumerate(inputs):
            try:
                for check in checks:
                    if check(state) is not None:
                        raise TypeError("precondition checks must return None on success")
            except InputValidationError as error:
                errors[index] = ResultError(code="invalid_input", message=str(error))
        return errors

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
        it for all model calls, including those in helpers. Keep it separate from
        input states and per-call instructions. The prompt text is transmitted;
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
        from lev._data import snapshot
        from lev.compiler import compile_program
        from lev.plan_lowering import lower_to_plan
        from lev.runtime import RunInput
        if self.closed:
            raise RuntimeError("client is closed")
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list")
        errors, valid = {}, []
        for index, state in enumerate(inputs):
            try:
                valid.append(RunInput(index, snapshot(state, state=True)))
            except (ValueError, RecursionError):
                errors[index] = ResultError("invalid_input", "input must contain finite JSON data and have string, array or object shape")
        checked = self._validate_inputs(program, [item.state for item in valid])
        errors.update({valid[i].index: error for i, error in checked.items()})
        valid = [item for i, item in enumerate(valid) if i not in checked]
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
