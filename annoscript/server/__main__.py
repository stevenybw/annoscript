"""python -m annoscript.server {api,worker} --db jobs.sqlite --config service.json"""

import argparse
import asyncio
import signal

from annoscript.server.releases import load_config
from annoscript.server.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("api", "worker", "releases"))
    parser.add_argument("--db", default="anno-jobs.sqlite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-generation-inflight", type=int)
    parser.add_argument("--max-decision-inflight", type=int)
    args = parser.parse_args()
    releases, tokens = load_config(args.config)
    if args.command == "releases":
        from annoscript.job_protocol import encode
        print(encode({"default_release": releases.default, "releases": releases.manifests}).decode())
        return
    store = Store(args.db)
    if args.command == "api":
        import uvicorn
        from annoscript.server.api import create_app
        uvicorn.run(create_app(store, releases, tokens), host=args.host, port=args.port, access_log=False)
    else:
        from annoscript.server.worker import Worker
        limits = None
        if args.max_generation_inflight is not None or args.max_decision_inflight is not None:
            if args.max_generation_inflight is None or args.max_decision_inflight is None:
                parser.error("both shared model limits must be provided")
            limits = {"generation": args.max_generation_inflight, "decision": args.max_decision_inflight}
        async def serve():
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await Worker(store, releases, concurrency=args.concurrency, model_request_limits=limits).run(stop=stop)
        asyncio.run(serve())


if __name__ == "__main__":
    main()
