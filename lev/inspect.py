"""Inspect a serialized Lev program, without importing or running its source.

Usage from the repository root (or with Lev on PYTHONPATH):

    python3 -m lev.inspect path/to/program.json
    python3 -m lev.inspect path/to/program.json --show-source
    cat path/to/program.json | python3 -m lev.inspect -

The Python API is Program.from_json(text) followed by format_ir(program).
Output goes to stdout and can be redirected to a file or piped to a pager.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from lev.formatter import format_ir
from lev.ir import IRDecodeError, Program


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m lev.inspect",
        description="Display a Lev IR JSON program without executing it.",
    )
    parser.add_argument("path", metavar="PATH", help="IR JSON file, or - to read stdin")
    parser.add_argument("--show-source", action="store_true", help="show statement IDs, source locations and inline call chains")
    args = parser.parse_args(argv)
    try:
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
        program = Program.from_json(text)
        output = format_ir(program, show_source=args.show_source)
    except (OSError, UnicodeError, IRDecodeError) as exc:
        print(f"lev.inspect: {exc}", file=sys.stderr)
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
