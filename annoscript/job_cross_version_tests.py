"""Mandatory Python 3.10 -> Python 3.12 API + worker durable-job acceptance."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

from annoscript.cross_version_tests import interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--server-python", required=True)
    args = parser.parse_args()
    interpreter(args.client_python, "3.10")
    interpreter(args.server_python, "3.12")
    root = Path(__file__).resolve().parents[1]
    processes = []

    def stop(process, *, kill=False):
        if process.poll() is None:
            process.kill() if kill else process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(5)

    with tempfile.TemporaryDirectory(prefix="anno-jobs-cross-") as folder:
        clock = Path(folder, "clock")
        clock.write_text("1000")

        def set_clock(value):
            temporary = clock.with_suffix(".next")
            temporary.write_text(str(value))
            temporary.replace(clock)

        def start(python, kind, *extra):
            process = subprocess.Popen([python, "-m", "annoscript._test_job_process", kind, folder, *extra],
                cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(process.stdout.readline)
                try:
                    line = future.result(timeout=30)
                    if not line:
                        raise RuntimeError(process.stderr.read())
                    return process, json.loads(line)
                except BaseException:
                    stop(process)
                    raise

        def client(url, action):
            result = subprocess.run([args.client_python, "-m", "annoscript._test_job_process", "client", folder,
                                     "--url", url, "--action", action], cwd=root, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr + result.stdout)
            return json.loads(result.stdout)

        def wait_for(predicate, seconds=20):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                value = predicate()
                if value:
                    return value
                time.sleep(0.02)
            raise AssertionError("cross-process condition timed out")

        try:
            subprocess.run([args.server_python, "-m", "annoscript._test_job_process", "fixtures", folder], cwd=root, check=True, timeout=30)
            api, ready = start(args.server_python, "api")
            worker, _ = start(args.server_python, "worker")
            url = ready["url"]
            assert client(url, "matrix")["matrix"] == "12 combinations passed"

            # Real independent API and worker restarts, retaining the same DB.
            from annoscript.jobs import AnnoJobClient
            with AnnoJobClient(url, "offline-token") as observer:
                durable = client(url, "submit-durable")
                wait_for(lambda: Path(folder, "model-started").exists())
                stop(worker, kill=True)
                stop(api)
                set_clock(1001)  # Expire the abandoned 0.4s worker lease.
                api, restarted = start(args.server_python, "api", "--port", str(urlsplit(url).port))
                assert restarted["url"] == url
                Path(folder, "release-model").touch()
                worker, _ = start(args.server_python, "worker")
                wait_for(lambda: observer.status(durable)["state"] == "completed")
                assert client(url, "read-durable")["recovery"] == "passed"

                # A killed simplified client cannot finalize; the durable owner
                # lease must cancel its outstanding job after virtual time moves.
                Path(folder, "release-model").unlink()
                Path(folder, "model-started").unlink(missing_ok=True)
                owner, owned = start(args.client_python, "client", "--url", url, "--action", "hold-session")
                wait_for(lambda: Path(folder, "model-started").exists())
                stop(owner, kill=True)
                set_clock(1302)
                status = observer.status(owned["job_id"])
                assert status["state"] == "cancelled", status
                page = observer.fetch_results(owned["job_id"])
                assert page["results"][0]["error"]["code"] == "session_expired"
        finally:
            for process in reversed(processes):
                stop(process)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
    print("Python 3.10 durable client -> Python 3.12 API/worker: OK (12 combinations, lost responses, restarts, killed owner)")


if __name__ == "__main__":
    main()
