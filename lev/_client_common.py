"""Python 3.10+ contracts shared by local and remote clients; no execution imports."""

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
    to every record's injected Lev and applies it to LLM operations, including
    calls inside helper functions. Decision requests use their own instructions
    and requirements. API guarantees and hard constraints remain independently
    enforced. Mapping shared instructions to a model is the runtime's job.

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
        except StopIteration:
            self.close()
            raise
        except BaseException:
            try:
                self.close()
            except BaseException:
                import logging
                logging.getLogger(__name__).warning("Result cleanup failed while handling another exception")
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
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                import logging
                logging.getLogger(__name__).warning("Result cleanup failed while handling another exception")
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


def get_program_config(program: Callable[[Lev], R]) -> dict[str, JSONValue]:
    """Return fresh, JSON-compatible execution settings for remote submission.

    Export system-prompt text only, never local check functions or rules.
    Programs without a declaration have no system_prompt configuration key.
    Reading configuration does not execute the program or its preconditions.
    """
    prompt = _SYSTEM_PROMPTS.get(program)
    return {} if prompt is None else {"system_prompt": prompt}


def validate_inputs(
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


def prepare_inputs(program, inputs):
    """Snapshot JSON inputs, then check all valid shapes before any submission.

    Return (indexed states, local errors), retaining the original input indices.
    Unexpected checker exceptions propagate before either client dispatches work.
    """
    from lev._data import snapshot

    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    errors, valid = {}, []
    for index, state in enumerate(inputs):
        try:
            valid.append((index, snapshot(state, state=True)))
        except (ValueError, RecursionError):
            errors[index] = ResultError(
                "invalid_input",
                "input must contain finite JSON data and have string, array or object shape",
            )
    checked = validate_inputs(program, [state for _, state in valid])
    errors.update({valid[i][0]: error for i, error in checked.items()})
    return [item for i, item in enumerate(valid) if i not in checked], errors
