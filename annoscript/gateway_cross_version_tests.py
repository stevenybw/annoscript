"""Offline Python 3.10+ Quick Start against a real Python 3.12+ gateway."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from annoscript._test_evaluation_http import http_backends
from annoscript.server.gateway import create_app
from annoscript.server.quota import Quota
from annoscript.test_gateway import loopback, ANSWER
from annoscript_examples.quick_start import SAMPLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-python', required=True)
    parser.add_argument('--installed', action='store_true', help='Use the client interpreter installed package outside the checkout')
    parser.add_argument('--require-base-only', action='store_true')
    args = parser.parse_args()
    requests = []
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    if args.installed:
        env.pop('PYTHONPATH', None)
    else:
        env['PYTHONPATH'] = str(root)
    with TemporaryDirectory() as directory:
        app = create_app(quota=Quota(Path(directory)/'q', 'fixture', minute=100), access_token='offline-server',
            backend_factory=lambda **kw: http_backends([(SAMPLE, ANSWER)], requests=requests))
        with loopback(app) as url:
            # Import a saved program; no eDSL source is synthesized in a REPL.
            code = '''import importlib.util,json,sys
from contextlib import redirect_stdout
from io import StringIO
from annoscript.client import AnnoClient
from annoscript_examples.quick_start import SAMPLE,main
from annoscript_examples.request import annotate
assert sys.version_info >= (3,10)
if BASE_ONLY:
    assert importlib.util.find_spec('fastapi') is None
    assert importlib.util.find_spec('uvicorn') is None
for mode in ('eager','strict','speculative'):
    with AnnoClient.server(URL,'offline-server',mode=mode) as client:
        assert client.runtime.llm_backend.session.http is None
        result=list(client.submit(annotate,[SAMPLE]))[0]
        assert result.status=='success',result.error
        assert result.value==EXPECTED,result.value
original=AnnoClient.server
AnnoClient.server=classmethod(lambda cls: original(URL,'offline-server'))
output=StringIO()
with redirect_stdout(output):
    assert main()==0
assert json.loads(output.getvalue())==EXPECTED
assert 'fastapi' not in sys.modules and 'uvicorn' not in sys.modules
print(json.dumps({'python':sys.version.split()[0],'modes':3,'quick_start':'passed','six_operations':'passed'}))
'''.replace('URL', repr(url)).replace('EXPECTED', repr(ANSWER)).replace('BASE_ONLY', repr(args.require_base_only))
            result = subprocess.run([args.client_python, '-c', code], cwd=directory,
                                    env=env, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            print(result.stdout.strip())
            assert app.state.active == 0
            assert not app.state.editor_sessions.items
            assert requests and any('questions' in request for request in requests)
            assert any('messages' in request for request in requests)
            print('Real gateway and provider HTTP fixtures passed; zero live model calls.')


if __name__ == '__main__':
    main()
