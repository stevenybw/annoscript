"""Loopback-only process fixtures for durable-job acceptance, never a deployment."""

import argparse
import json
from pathlib import Path


def server_components(folder):
    from annoscript.server.releases import Releases
    from annoscript.server.store import Store
    from annoscript.test_jobs_server import configurations
    clock = lambda: float(Path(folder, "clock").read_text())
    return Store(Path(folder, "jobs.sqlite"), clock=clock, lease_seconds=0.4), Releases(configurations(), "test")


def serve_api(folder, port):
    import asyncio
    import socket
    import uvicorn
    from annoscript.server.api import create_app
    store, releases = server_components(folder)
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(create_app(store, releases, {"offline-token": "acceptance"}),
                                           log_level="error", access_log=False))
    async def run():
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("test API did not start")
            await asyncio.sleep(0.01)
        print(json.dumps({"url": f"http://127.0.0.1:{sock.getsockname()[1]}"}), flush=True)
        await task
    asyncio.run(run())


def serve_worker(folder):
    import asyncio
    from annoscript._data import thaw
    from annoscript._test_remote_server import cases
    from annoscript._test_synthetic import CannedBackend
    from annoscript.backends import InferenceOutput
    from annoscript.anno import GenerateResponse
    from annoscript.server.worker import Worker
    store, releases = server_components(folder)
    scenarios, _, _, failures = cases()
    class Backend(CannedBackend):
        async def infer(self, context, call):
            state = thaw(context.state)
            if type(state) is str:
                if state == "__hold__":
                    Path(folder, "model-started").touch()
                    while not Path(folder, "release-model").exists():
                        await asyncio.sleep(0.01)
                return InferenceOutput(GenerateResponse(state))
            return await super().infer(context, call)
    def factory(manifest):
        backend = Backend(scenarios, failures, speculative=True)
        return backend, backend
    print(json.dumps({"worker": "ready"}), flush=True)
    asyncio.run(Worker(store, releases, backend_factory=factory, random_source=lambda: 0,
                       poll_interval=0.01).run())


def fixtures(folder):
    from annoscript._test_remote_server import cases
    _, inputs, expected, _ = cases()
    Path(folder, "fixtures.json").write_text(json.dumps({"inputs": inputs[:8], "expected": expected[:8]}, ensure_ascii=False), encoding="utf-8")


def run_client(url, folder, action):
    # Fail fast on accidental import coupling in a real Python 3.10 process.
    import importlib.abc
    import sys
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, *args):
            if fullname.startswith(("annoscript.runtime", "annoscript.backends", "annoscript.plan", "annoscript.server", "fastapi", "uvicorn")):
                raise AssertionError("execution module imported on client: " + fullname)
    sys.meta_path.insert(0, Block())
    import httpx
    from annoscript_examples.request import annotate
    from annoscript.client import AnnoClient
    from annoscript.compiler import compile_program
    from annoscript.jobs import AnnoJobClient
    from annoscript.test_jobs import generate
    if action == "matrix":
        data = json.loads(Path(folder, "fixtures.json").read_text())
        class DropResponses(httpx.BaseTransport):
            def __init__(self):
                self.inner = httpx.HTTPTransport(retries=0)
                self.dropped = set()
            def handle_request(self, request):
                response = self.inner.handle_request(request)
                kind = "accept" if request.url.path == "/v1/jobs" else "page" if request.url.path.endswith("/results") else None
                if kind and kind not in self.dropped:
                    self.dropped.add(kind)
                    response.read()
                    response.close()
                    raise httpx.ReadError("deliberately lost committed HTTP response")
                return response
            def close(self):
                self.inner.close()
        for mode in ("strict", "eager", "speculative"):
            for optimize in (False, True):
                for ordered in (False, True):
                    with AnnoClient.remote(url, "offline-token", mode=mode, optimize=optimize,
                                          transport=DropResponses()) as client:
                        client._remote.session.client.sleep = lambda _: None
                        results = list(client.submit(annotate, data["inputs"], ordered=ordered))
                        assert len(results) == len(data["expected"])
                        if ordered:
                            assert [r.index for r in results] == list(range(len(results)))
                        by_index = {r.index: r for r in results}
                        assert len(by_index) == len(results)
                        for index, expected in enumerate(data["expected"]):
                            result = by_index[index]
                            assert result.status == expected["status"], (index, result)
                            if result.status == "success":
                                assert result.value == expected["value"], (index, result)
                            else:
                                assert result.error.code == expected["code"], (index, result)
                        assert by_index[2].value["tags"] == ["sample"]
        print(json.dumps({"matrix": "12 combinations passed"}), flush=True)
    elif action == "hold-session":
        client = AnnoClient.remote(url, "offline-token")
        stream = client.submit(generate, ["__hold__"])
        print(json.dumps({"job_id": stream.job["job_id"]}), flush=True)
        sys.stdin.readline()  # Parent kills the process; no destructor can run.
        client.close()
    elif action == "submit-durable":
        with AnnoJobClient(url, "offline-token") as client:
            job = client.submit_ir(compile_program(generate), [{"index": 42, "state": "__hold__"}], idempotency_key="durable")
        print(json.dumps(job), flush=True)
    elif action == "read-durable":
        with AnnoJobClient(url, "offline-token") as client:
            job = client.lookup("durable")
            page = client.fetch_results(job)
            assert page["done"] and page["results"][0]["index"] == 42
            assert page["results"][0]["value"] == "__hold__"
            assert page["results"][0]["attempts"] == 2
        print(json.dumps({"recovery": "passed"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("api", "worker", "fixtures", "client"))
    parser.add_argument("folder")
    parser.add_argument("--url")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--action", default="matrix")
    args = parser.parse_args()
    if args.kind == "api":
        serve_api(args.folder, args.port)
    elif args.kind == "worker":
        serve_worker(args.folder)
    elif args.kind == "fixtures":
        fixtures(args.folder)
    else:
        run_client(args.url, args.folder, args.action)


if __name__ == "__main__":
    main()
