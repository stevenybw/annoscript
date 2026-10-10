"""Batch submission and result-stream contracts for Lev eDSL programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, List, Literal, TypeAlias, TypeVar

from lev.lev import Lev


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
State: TypeAlias = str | list[JSONValue] | dict[str, JSONValue]
R = TypeVar("R", bound=JSONValue)


@dataclass(frozen=True)
class ResultError:
    """A per-input execution failure, including an unsatisfied constraint."""

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


class LevClient(ABC):
    """Client configuration and batch submission contract for a Lev service.

    ``base_url`` identifies the service's API root; ``submit_path`` is appended
    to that root, preserving any base path. ``api_key=None`` means no
    authentication header; otherwise use Bearer authentication. ``timeout`` is
    the HTTP operation timeout in seconds and must be finite and positive.

    Construction only stores configuration. Concrete implementations handle
    validation, transport, and scheduling. Function packaging and dependency
    submission are left to future implementations.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        submit_path: str = "/submit",
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.submit_path = submit_path

    @abstractmethod
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

        ``inputs`` must be a list of strings, JSON arrays, or JSON objects.
        An empty list produces an empty result stream. Input and output JSON
        values must use string object keys and finite numbers.

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
        raise NotImplementedError
