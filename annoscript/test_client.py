"""Behavioral checks for result-stream ownership and terminal outcomes."""

from collections.abc import Iterable
import json
import unittest

from annoscript.client import (
    InputValidationError, AnnoClient, Result, ResultError, ResultStream,
    precondition, require, system_prompt,
)


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


class SystemPromptTests(unittest.TestCase):
    def test_registration_preserves_function_and_prompt_text(self):
        calls = []

        def program(anno):
            calls.append(anno)
            return "done"

        original_code = program.__code__
        original_attributes = vars(program).copy()
        prompt = '  公共规则\n保留 "原文"。  '
        decorated = system_prompt(prompt)(program)
        self.assertIs(decorated, program)
        self.assertIs(decorated.__code__, original_code)
        self.assertEqual(vars(decorated), original_attributes)
        self.assertEqual(AnnoClient._get_program_config(decorated), {"system_prompt": prompt})
        self.assertEqual(calls, [])
        self.assertEqual(decorated(None), "done")
        self.assertEqual(calls, [None])

    def test_duplicate_declarations_fail_without_replacing_original(self):
        for prompt in ("rules", ""):
            with self.subTest(prompt=prompt):
                @system_prompt(prompt)
                def program(anno):
                    return "unused"

                for repeated in (prompt, "replacement"):
                    with self.assertRaisesRegex(ValueError, "only one"):
                        system_prompt(repeated)(program)
                self.assertEqual(AnnoClient._get_program_config(program), {"system_prompt": prompt})

    def test_both_decorator_orders_export_prompt_and_keep_checks_local(self):
        for prompt_outermost in (True, False):
            with self.subTest(prompt_outermost=prompt_outermost):
                checked = []

                def check(state):
                    checked.append(state)
                    require(isinstance(state, str), message="expected text")

                def program(anno):
                    self.fail("Metadata and validation must not execute the program")

                if prompt_outermost:
                    decorated = system_prompt("rules")(precondition(check)(program))
                else:
                    decorated = precondition(check)(system_prompt("rules")(program))
                config = AnnoClient._get_program_config(decorated)
                self.assertEqual(json.loads(json.dumps(config)), {"system_prompt": "rules"})
                self.assertEqual(checked, [])
                self.assertIs(decorated, program)
                errors = AnnoClient._validate_inputs(decorated, ["valid", {}])
                self.assertEqual(checked, ["valid", {}])
                self.assertEqual(set(errors), {1})
                self.assertEqual(errors[1].code, "invalid_input")

    def test_configuration_is_isolated_between_programs_and_reads(self):
        @system_prompt("first rules")
        def first(anno):
            return "unused"

        @system_prompt("second rules")
        def second(anno):
            return "unused"

        def undecorated(anno):
            return "unused"

        config = AnnoClient._get_program_config(first)
        config["system_prompt"] = "changed"
        self.assertEqual(AnnoClient._get_program_config(first), {"system_prompt": "first rules"})
        self.assertEqual(AnnoClient._get_program_config(second), {"system_prompt": "second rules"})
        self.assertEqual(AnnoClient._get_program_config(undecorated), {})

    def test_prompt_must_be_text(self):
        for prompt in (None, 1, [], {}):
            with self.subTest(prompt=prompt):
                with self.assertRaisesRegex(TypeError, "requires a string"):
                    system_prompt(prompt)


class PreconditionTests(unittest.TestCase):
    def test_require_reports_invalid_input(self):
        self.assertIsNone(require(True, message="unused"))
        with self.assertRaisesRegex(InputValidationError, "text is required"):
            require(False, message="text is required")

    def test_registration_preserves_function_and_does_not_run_checks(self):
        checked = []

        def check(state):
            checked.append(state)

        def program(anno):
            return "done"

        original_attributes = vars(program).copy()
        original_code = program.__code__
        decorated = precondition(check)(program)
        self.assertIs(decorated, program)
        self.assertIs(decorated.__code__, original_code)
        self.assertEqual(vars(decorated), original_attributes)
        self.assertEqual(decorated(None), "done")
        self.assertEqual(checked, [])
        self.assertEqual(AnnoClient._validate_inputs(decorated, ["record"]), {})
        self.assertEqual(checked, ["record"])

    def test_local_batch_check_preserves_indices_and_never_runs_program(self):
        checked = []

        def check(state):
            checked.append(state)
            require(isinstance(state, str), message="expected text")

        @precondition(check)
        def program(anno):
            self.fail("Preconditions must not execute the program")

        inputs = [{}, "valid", []]
        errors = AnnoClient._validate_inputs(program, inputs)
        self.assertEqual(checked, inputs)
        self.assertEqual(errors, {
            0: ResultError(code="invalid_input", message="expected text"),
            2: ResultError(code="invalid_input", message="expected text"),
        })

    def test_stacked_checks_run_in_source_order_and_stop_on_invalid_input(self):
        checked = []

        def check_object(state):
            checked.append("object")
            require(isinstance(state, dict), message="expected object")

        def check_text(state):
            checked.append("text")
            require(isinstance(state.get("text"), str), message="expected text")

        @precondition(check_object)
        @precondition(check_text)
        def program(anno):
            return "unused"

        errors = AnnoClient._validate_inputs(program, ["wrong", {}, {"text": "valid"}])
        self.assertEqual(set(errors), {0, 1})
        self.assertEqual(checked, ["object", "object", "text", "object", "text"])

    def test_unexpected_checker_errors_abort_validation(self):
        for error in (ValueError("checker bug"), KeyError("missing")):
            with self.subTest(error=type(error).__name__):
                checked = []

                def check(state):
                    checked.append(state)
                    raise error

                @precondition(check)
                def program(anno):
                    return "unused"

                with self.assertRaises(type(error)) as caught:
                    AnnoClient._validate_inputs(program, ["first", "second"])
                self.assertIs(caught.exception, error)
                self.assertEqual(checked, ["first"])

    def test_boolean_checker_return_is_a_defect_not_a_validation_result(self):
        @precondition(lambda state: False)
        def program(anno):
            return "unused"

        with self.assertRaisesRegex(TypeError, "return None"):
            AnnoClient._validate_inputs(program, ["record"])

    def test_empty_batch_and_program_without_preconditions(self):
        def check(state):
            self.fail("An empty batch must not run checks")

        def program(anno):
            self.fail("Validation must not run the program")

        self.assertEqual(AnnoClient._validate_inputs(program, ["record"]), {})
        precondition(check)(program)
        self.assertEqual(AnnoClient._validate_inputs(program, []), {})


if __name__ == "__main__":
    unittest.main()
