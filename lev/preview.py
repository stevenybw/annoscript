"""Preview resolved logical calls through the real builders, without model execution.

python -m lev.preview lev_examples/request_preview.decision.json -o /tmp/decision.json
No Python program is imported, no credential file is read, and no HTTP is sent.
"""

import argparse
import json
from pathlib import Path
import sys

from lev._data import snapshot
from lev._request_adapters import DEFAULT_DECISION_ADAPTER, DEFAULT_GENERATION_ADAPTER
from lev.backends import DecisionAPIBackend, LLMAPIBackend, InferenceContext, InferenceError
from lev.backends.base import normalize_call, DECISION_OPERATIONS, LLM_OPERATIONS
from lev.job_protocol import decode
from lev.remote import RemoteProtocolError


def preview_request(spec, *, split_batches=False, max_calls=32):
    """Preview one physical request from explicitly resolved calls and input.

    Generation batch max_tokens is the already selected shared budget. For a
    single call max_tokens, when present, caps its normal effective budget.
    This does not execute IR, guess branch results, or select a batch.
    """
    spec = snapshot(spec)
    required = {"format", "version", "stage", "model", "state", "calls"}
    optional = {"system_prompt", "request_adapter", "request_adapter_options", "options", "mode", "max_tokens", "repair_feedback"}
    if (type(spec) is not dict or not required <= set(spec) or set(spec) - required - optional
            or spec["format"] != "lev-request-spec" or type(spec["version"]) is not int or spec["version"] != 1):
        raise ValueError("expected lev-request-spec version 1 with resolved calls")
    if type(spec["stage"]) is not str or spec["stage"] not in ("decision", "generation"):
        raise ValueError("stage must be decision or generation")
    if "request_adapter_options" in spec and type(spec["request_adapter_options"]) is not dict:
        raise ValueError("request_adapter_options must be an object")
    context = InferenceContext(spec["state"], spec.get("system_prompt"))
    items = spec["calls"]
    if type(items) is not list or not items:
        raise ValueError("calls must be a nonempty list")
    calls, ids = [], set()
    for item in items:
        if (type(item) is not dict or set(item) != {"id", "operation", "arguments"}
                or type(item["id"]) is not str or not item["id"].strip() or item["id"] in ids
                or type(item["operation"]) is not str or type(item["arguments"]) is not dict):
            raise ValueError("calls require distinct nonblank ids, operation and arguments")
        ids.add(item["id"])
        operations = DECISION_OPERATIONS if spec["stage"] == "decision" else LLM_OPERATIONS
        if item["operation"] not in operations:
            raise ValueError("operation does not belong to the selected stage")
        calls.append(normalize_call(item["id"], item["operation"], (), item["arguments"], context))
    options = spec.get("options", {})
    if type(options) is not dict:
        raise ValueError("options must be an object")
    if spec["stage"] == "decision":
        if options or spec.get("mode", "batch") != "batch" or {"max_tokens", "repair_feedback"} & set(spec):
            raise ValueError("decision preview requires batch mode without generation options")
        backend = DecisionAPIBackend("http://offline.invalid/decision", "unused-preview-token", spec["model"],
                                     request_adapter=spec.get("request_adapter", DEFAULT_DECISION_ADAPTER),
                                     request_adapter_options=spec.get("request_adapter_options"))
        if split_batches:
            return backend.preview_batches(context, calls, max_calls=max_calls)
        return backend.preview_batch(context, calls)
    if split_batches or "request_adapter_options" in spec:
        raise ValueError("split_batches and request_adapter_options require decision stage")
    allowed = {"profile", "extract_mode", "disable_thinking", "batch_rule_dedup",
               "request_policy", "assumed_avg_num_tokens_per_char"}
    if set(options) - allowed:
        raise ValueError("unsupported generation preview options")
    mode = spec.get("mode", "single")
    if mode not in ("single", "batch"):
        raise ValueError("generation mode must be single or batch")
    if "repair_feedback" in spec and type(spec["repair_feedback"]) is not str:
        raise ValueError("repair_feedback must be a string")
    if "max_tokens" in spec and (type(spec["max_tokens"]) is not int or spec["max_tokens"] < 1):
        raise ValueError("max_tokens must be a positive integer")
    if mode == "batch" and ("max_tokens" not in spec or "repair_feedback" in spec):
        raise ValueError("batch preview needs max_tokens and cannot include repair_feedback")
    if mode == "single" and len(calls) != 1:
        raise ValueError("single preview requires exactly one call")
    backend = LLMAPIBackend("http://offline.invalid/v1", "unused-preview-token", spec["model"],
                            request_adapter=spec.get("request_adapter", DEFAULT_GENERATION_ADAPTER), **options)
    if mode == "batch":
        return backend.preview_batch(context, calls, max_tokens=spec["max_tokens"])
    return backend.preview(context, calls[0], repair_feedback=spec.get("repair_feedback"),
                           max_tokens_override=spec.get("max_tokens"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lev.preview", description=__doc__)
    parser.add_argument("path", help="resolved lev-request-spec JSON, or - for stdin")
    parser.add_argument("-o", "--output", help="write preview JSON to a different file")
    parser.add_argument("--split-batches", action="store_true", help="preview a Decision physical batch plan")
    parser.add_argument("--max-calls", type=int, default=32, help="Decision questions per physical batch (with --split-batches)")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read() if args.path == "-" else Path(args.path).read_bytes()
        if not args.split_batches and args.max_calls != 32:
            raise ValueError("--max-calls requires --split-batches")
        result = preview_request(decode(raw), split_batches=args.split_batches, max_calls=args.max_calls)
        output = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if args.output and args.output != "-":
            if args.path != "-" and Path(args.path).resolve() == Path(args.output).resolve():
                raise ValueError("output path must differ from the input specification")
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, RemoteProtocolError, InferenceError) as exc:
        print(f"lev.preview: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
