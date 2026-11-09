"""Loopback-only HTTP fixture for cross-version acceptance. Never a production server."""

import asyncio
from copy import deepcopy
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys

from lev._test_synthetic import CannedBackend, cases, key
from lev.ir import Program
from lev.passes import optimize_program
from lev.plan_lowering import lower_to_plan
from lev.remote import ir_sha256
from lev.runtime import LevRuntime, RunInput


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("test server requires Python 3.12")
    scenarios, inputs, expected, failures = cases()
    Path(sys.argv[1]).write_text(json.dumps({"inputs": inputs, "expected": expected}, ensure_ascii=False), encoding="utf-8")
    # Record fixtures before accepting requests; the real Runtime performs all
    # constraint checks and model-response validation during execution.
    fixture_backend = CannedBackend(scenarios, failures, speculative=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def frame(self, value):
            self.wfile.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode())
            self.wfile.flush()

        def do_POST(self):
            if self.path != "/v1/runs:batch":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if payload["protocol_version"] != 1 or ir_sha256(payload["ir"]) != payload["ir_sha256"]:
                self.send_error(400)
                return
            # Deserialization, optimization and lowering occur in this process.
            program = Program.from_json(json.dumps(payload["ir"], ensure_ascii=False, allow_nan=False))
            plan = lower_to_plan(optimize_program(program))
            received = payload["inputs"]
            if [item["index"] for item in received] != list(range(1, len(inputs))):
                self.send_error(400, "client did not preserve validated input indices")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.frame({"type": "accepted", "protocol_version": 1, "ir_sha256": payload["ir_sha256"]})

            async def execute():
                backend = deepcopy(fixture_backend)
                async with LevRuntime(backend, backend) as runtime:
                    stream = await runtime.submit(plan, [RunInput(item["index"], item["state"]) for item in received],
                                                  mode=payload["mode"])
                    async for result in stream:
                        self.frame({"type": "result", **asdict(result)})
                if key(inputs[0]) in {state for state, _, _ in backend.calls}:
                    raise AssertionError("invalid input reached a backend")

            asyncio.run(execute())
            self.frame({"type": "done"})

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "python": "3.12"}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
