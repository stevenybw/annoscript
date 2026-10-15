"""python -m lev.server {api,worker} --db jobs.sqlite --config service.json"""

import argparse
import asyncio
import signal

from lev.server.releases import load_config
from lev.server.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("api", "worker", "releases"))
    parser.add_argument("--db", default="lev-jobs.sqlite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    releases, tokens = load_config(args.config)
    if args.command == "releases":
        from lev.job_protocol import encode
        print(encode({"default_release": releases.default, "releases": releases.manifests}).decode())
        return
    store = Store(args.db)
    if args.command == "api":
        import uvicorn
        from lev.server.api import create_app
        uvicorn.run(create_app(store, releases, tokens), host=args.host, port=args.port, access_log=False)
    else:
        from lev.server.worker import Worker
        async def serve():
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await Worker(store, releases, concurrency=args.concurrency).run(stop=stop)
        asyncio.run(serve())


if __name__ == "__main__":
    main()
