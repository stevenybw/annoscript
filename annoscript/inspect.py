"""Inspect AnnoScript IR or an Execution Plan without importing or running its source.

Usage from the repository root (or with AnnoScript on PYTHONPATH):

    python3 -m annoscript.inspect path/to/program.json
    python3 -m annoscript.inspect path/to/program.json --show-source
    cat path/to/program.json | python3 -m annoscript.inspect -
    python3 -m annoscript.inspect request.execution_plan.json --mode strict
    python3 -m annoscript.inspect request.execution_plan.json --format html -o plan.html

For Python callers, use Program.from_json + annoscript.formatter.format_ir, or
ExecutionPlan.from_json + annoscript.plan_formatter.format_plan / format_plan_html.
Output goes to stdout unless --output is provided. Inspection does not lower
or optimize IR. HTML is self-contained and can be opened offline in a browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from annoscript.formatter import format_ir
from annoscript.ir import IRDecodeError, Program
from annoscript.plan import ExecutionPlan
from annoscript.plan_formatter import format_plan, format_plan_html, format_plan_mermaid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m annoscript.inspect",
        description="Inspect AnnoScript IR or visualize Execution Plan Infer dependencies without execution.",
    )
    parser.add_argument("path", metavar="PATH", help="AnnoScript IR / Execution Plan JSON, or - for stdin")
    parser.add_argument("--show-source", action="store_true", help="show statement IDs, source locations and inline call chains")
    parser.add_argument("--format", choices=("text", "html", "mermaid"), default="text", help="output format (HTML/Mermaid require an Execution Plan)")
    parser.add_argument("--mode", choices=("eager", "strict", "speculative"), default="eager", help="Execution Plan failure mode; HTML includes all three modes")
    parser.add_argument("-o", "--output", metavar="PATH", help="write UTF-8 output to this file instead of stdout")
    args = parser.parse_args(argv)
    try:
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
        # Only sniff the format here. Each strict decoder parses the original
        # text again, so duplicate JSON keys and non-finite values stay errors.
        header = json.loads(text)
        if isinstance(header, dict) and header.get("format") == "lev-execution-plan":
            plan = ExecutionPlan.from_json(text)
            formatter = {"text": format_plan, "html": format_plan_html, "mermaid": format_plan_mermaid}[args.format]
            output = formatter(plan, mode=args.mode, show_source=args.show_source)
        else:
            program = Program.from_json(text)
            if args.format != "text" or args.mode != "eager":
                raise IRDecodeError("--format html/mermaid and --mode strict/speculative require an Execution Plan; lower the IR explicitly first")
            output = format_ir(program, show_source=args.show_source)
        if args.output and args.output != "-":
            if args.path != "-" and Path(args.output).resolve() == Path(args.path).resolve():
                raise ValueError("output path must differ from the input JSON file")
            Path(args.output).write_text(output, encoding="utf-8")
            return 0
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        print(f"annoscript.inspect: {exc}", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except BrokenPipeError:
        # A pager may stop reading early. Closing also prevents a second failed
        # flush during interpreter shutdown.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
