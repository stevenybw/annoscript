"""Offline cross-process acceptance; both interpreters are mandatory.

python -m lev.cross_version_tests --client-python /path/to/python3.10 \
    --server-python /path/to/python3.12

The loopback HTTP fixture is test-only, not an execution service to deploy.
"""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def interpreter(path, version):
    completed = subprocess.run([path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                               capture_output=True, text=True, check=True, timeout=10)
    if completed.stdout.strip() != version:
        raise ValueError(f"expected Python {version}, got {completed.stdout.strip()} from {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--server-python", required=True)
    args = parser.parse_args()
    interpreter(args.client_python, "3.10")
    interpreter(args.server_python, "3.12")
    root = Path(__file__).resolve().parents[1]
    # Generate only temporary IPC fixtures, never checked-in artifacts.
    with tempfile.TemporaryDirectory(prefix="lev-cross-version-") as folder:
        fixture = Path(folder, "cases.json")
        server = subprocess.Popen([args.server_python, "-m", "lev._test_remote_server", str(fixture)],
                                  cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            # A reader future bounds startup without polling the filesystem.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(server.stdout.readline)
                try:
                    ready = json.loads(future.result(timeout=30))
                except BaseException:
                    server.terminate()
                    server.wait(timeout=10)
                    raise
            if ready.get("python") != "3.12" or not ready.get("url"):
                raise RuntimeError("invalid test-server startup response")
            completed = subprocess.run(
                [args.client_python, "-m", "lev._test_remote_client", ready["url"], str(fixture)],
                cwd=root, timeout=180,
            )
            if completed.returncode:
                raise RuntimeError(f"cross-version client exited with {completed.returncode}")
        finally:
            server.terminate()
            try:
                _, errors = server.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                _, errors = server.communicate()
            if errors:
                raise RuntimeError("test server failed:\n" + errors)
    print("Python 3.10 IR client -> Python 3.12 optimizing Runtime: OK (offline)")


if __name__ == "__main__":
    main()
