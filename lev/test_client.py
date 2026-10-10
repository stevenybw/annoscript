"""Behavioral checks for result-stream ownership and terminal outcomes."""

from collections.abc import Iterable
import unittest

from lev.client import Result, ResultError, ResultStream


class ScriptedStream(ResultStream[str]):
    def __init__(
        self,
        events: Iterable[Result[str] | BaseException] = (),
        *,
        cleanup_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.events = iter(events)
        self.cleanup_error = cleanup_error
        self.pulls = 0
        self.cleanups = 0

    def _next_result(self) -> Result[str]:
        self.pulls += 1
        event = next(self.events)
        if isinstance(event, BaseException):
            raise event
        return event

    def _close(self) -> None:
        self.cleanups += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class ResultStreamTests(unittest.TestCase):
    def test_exhaustion_closes_nonempty_and_empty_streams(self):
        result = Result(index=0, status="success", value="done")
        for events in ([], [result]):
            with self.subTest(events=events):
                stream = ScriptedStream(events)
                self.assertIs(iter(stream), stream)
                self.assertEqual(list(stream), events)
                self.assertTrue(stream.closed)
                self.assertEqual(stream.cleanups, 1)

    def test_explicit_close_is_idempotent_and_prevents_further_pulls(self):
        stream = ScriptedStream([Result(index=0, status="success", value="unused")])
        stream.close()
        stream.close()
        self.assertEqual(list(stream), [])
        self.assertTrue(stream.closed)
        self.assertEqual(stream.pulls, 0)
        self.assertEqual(stream.cleanups, 1)

    def test_context_closes_after_early_break_without_consuming_remainder(self):
        first = Result(index=0, status="success", value="first")
        second = Result(index=1, status="success", value="second")
        source = iter([first, second])
        stream = ScriptedStream(source)
        with stream as entered:
            self.assertIs(entered, stream)
            for result in entered:
                self.assertIs(result, first)
                break
            self.assertFalse(stream.closed)
        self.assertTrue(stream.closed)
        self.assertEqual(stream.cleanups, 1)
        self.assertEqual(stream.pulls, 1)
        self.assertIs(next(source), second)
        self.assertEqual(list(stream), [])

    def test_context_exits_without_double_cleanup_after_exhaustion(self):
        stream = ScriptedStream()
        with stream:
            self.assertEqual(list(stream), [])
        self.assertEqual(stream.cleanups, 1)

    def test_consumer_exception_propagates_and_closes(self):
        error = ValueError("consumer failed")
        stream = ScriptedStream()
        with self.assertRaises(ValueError) as caught:
            with stream:
                raise error
        self.assertIs(caught.exception, error)
        self.assertTrue(stream.closed)
        self.assertEqual(stream.cleanups, 1)

    def test_iteration_failure_and_interrupt_propagate_and_close(self):
        for error in (ConnectionError("connection lost"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                stream = ScriptedStream([error])
                with self.assertRaises(type(error)) as caught:
                    next(stream)
                self.assertIs(caught.exception, error)
                self.assertTrue(stream.closed)
                self.assertEqual(stream.cleanups, 1)
                self.assertEqual(list(stream), [])
                self.assertEqual(stream.pulls, 1)

    def test_per_input_error_does_not_stop_following_results(self):
        failed = Result[str](
            index=0,
            status="error",
            error=ResultError(code="constraint_unsatisfied", message="No valid label"),
        )
        succeeded = Result(index=1, status="success", value="done")
        stream = ScriptedStream([failed, succeeded])
        with stream:
            self.assertIs(next(stream), failed)
            self.assertFalse(stream.closed)
            self.assertIs(next(stream), succeeded)
        self.assertEqual(stream.cleanups, 1)

    def test_cleanup_failure_still_leaves_stream_closed(self):
        error = OSError("cleanup failed")
        stream = ScriptedStream(cleanup_error=error)
        with self.assertRaises(OSError) as caught:
            stream.close()
        self.assertIs(caught.exception, error)
        self.assertTrue(stream.closed)
        stream.close()
        self.assertEqual(list(stream), [])
        self.assertEqual(stream.cleanups, 1)

    def test_closed_stream_cannot_be_reentered(self):
        stream = ScriptedStream()
        stream.close()
        with self.assertRaises(RuntimeError):
            with stream:
                self.fail("A closed stream must not enter the context body")
        self.assertEqual(stream.cleanups, 1)


class ResultTests(unittest.TestCase):
    def test_success_preserves_none_and_list_values(self):
        null_result = Result(index=0, status="success", value=None)
        self.assertEqual(null_result.status, "success")
        self.assertIsNone(null_result.error)
        value = ["a", "b"]
        list_result = Result(index=1, status="success", value=value)
        self.assertIs(list_result.value, value)

    def test_rejects_invalid_indices_and_contradictory_outcomes(self):
        error = ResultError(code="failed", message="Failed")
        cases = [
            {"index": -1, "status": "success"},
            {"index": True, "status": "success"},
            {"index": 0, "status": "pending"},
            {"index": 0, "status": "success", "error": error},
            {"index": 0, "status": "error"},
            {"index": 0, "status": "error", "error": error, "value": "unexpected"},
        ]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    Result(**options)


if __name__ == "__main__":
    unittest.main()
