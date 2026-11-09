# Deploy Your Own AnnoScript Server

This advanced setup runs the same model-operation gateway as the Playground.
Python programs, preconditions, the compiler and runtime execute on the user's
computer (Python 3.12+) or in their browser. The gateway accepts only `noul`,
`choice`, `score`, `generate`, `generate_list` and `extract`, including bounded
batches. It never accepts Python source or serialized IR for execution.

For an immediate result without deployment, use the [Quick Start](../README.md#quick-start).
Existing durable-job interfaces remain available separately; this gateway does
not require a job worker or a release registry.

## 1. Configure the gateway

Install Docker with Compose. Download or clone this repository and use
[`deploy/server`](../deploy/server/):

```bash
cd deploy/server
cp .env.example .env
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Generate two different random values for `ANNO_SERVER_TOKEN` (client access) and
`ANNO_QUOTA_HMAC_SECRET` (persistent IP pseudonyms). Edit `.env` locally. Do not
commit it, include it in build contexts, or send provider keys to clients.

Set both model base URLs, model names and provider credentials. Decision uses
AnnoScript's Decision question/answer API. Generation uses the selected
`legacy` or `qwen_vllm` protocol; a generic chat endpoint alone is not a Decision
service. Models may be external services or inference services you already run.
Container `localhost` refers to the gateway container, not a model on the host.
No models or GPUs are bundled in this gateway image.

The default requires `ANNO_SERVER_TOKEN`. Missing required configuration fails
startup before any model request. `ANNO_SERVER_ANONYMOUS=1` explicitly enables
anonymous sessions, as used by the official experience service. Keep authentication
on for your own deployment unless anonymous access is intentional.

## 2. Start one process and preserve storage

```bash
docker compose up -d --build --wait
curl --fail http://127.0.0.1:8000/healthz
docker compose logs --tail=50 gateway
```

The image installs the checked-out `annoscript[server]` source (this release: 0.2.0).
Use the `v0.2.0` tag for a reproducible source selection. It runs one process as a non-root
user with a read-only root filesystem. The named `quota` volume stores SQLite
admission counts; inputs and model responses are not stored there. Preserve
this volume and the HMAC secret across upgrades. Do not run `down -v` during
upgrades or rollback.

Sessions and active request counters are process-local: run exactly **one gateway
process**, with no extra Uvicorn workers or replicas. A restart expires sessions;
clients report that error and do not silently open replacements. Horizontal
scaling requires shared session/concurrency coordination, which is not provided.

For a non-container installation, export the same environment variables, install
`pip install 'annoscript[server]==0.2.0'`, and run:

```bash
python -m annoscript.server gateway --host 127.0.0.1 --port 8000
```

## 3. Add HTTPS

Point your own DNS hostname at your server, allow ports 80/443, and place an HTTPS
reverse proxy in front of `127.0.0.1:8000`. The supplied
[`Caddyfile.example`](../deploy/server/Caddyfile.example) uses automatic TLS;
replace its hostname and run Caddy on the same host. Persist Caddy's certificate
storage. Keep gateway port 8000 bound to loopback and do not expose it directly.

For browser clients, set `ANNO_SERVER_ORIGINS` to exact HTTPS origins. Python
clients do not send an Origin header. The gateway does not trust forwarded IP
headers from arbitrary peers. With a container or proxy, per-IP limits may group
users under the proxy's IP; overall and per-session limits still apply. Configure
trusted forwarding deliberately if your deployment needs individual client IPs.

## 4. Connect the same client

```python
import os
from annoscript.client import AnnoClient
from annoscript_examples.request import annotate
from annoscript_examples.quick_start import SAMPLE

with AnnoClient.server(
    "https://annotations.example.com",
    auth_token=os.environ["MY_ANNO_SERVER_TOKEN"],
) as client:
    with client.submit(annotate, [SAMPLE]) as results:
        for result in results:
            if result.status != "success":
                raise RuntimeError(result.error.message)
            print(result.value)
```

`MY_ANNO_SERVER_TOKEN` is the server access token, never a provider API key.
The client does not read provider credential files or environment variables.
Construction is offline; the first model operation creates one session. Closing
the client closes that session and its HTTP connection. Limits, timeouts and
ambiguous transport failures are explicit errors, without automatic retries or
session renewal. Create a new client explicitly only when you intend a new run.

## Limits and operation protocol

The wire routes remain `/api/playground/session`, `/api/playground/operation`
and `/api/playground/session/close`. The server access token authenticates session
creation. Operations and close use the returned ephemeral session bearer.
Session creation consumes one admission (default: global 1,000/day, 50/IP/day,
5/IP/minute, reset at midnight Asia/Shanghai). Empty sessions do not hold an
execution slot. Default active limit is four sessions; parallel Decision and
Generation requests in the same session share a slot until both finish.

A session lasts 120 seconds and allows 12 requests and 8,192 generation output
tokens reserved from requested budgets. One request accepts at most 32 same-backend
calls, 4,096 output tokens, 16 KB of state and a 64 KB JSON body. Each backend has
one in-flight request per session. Model requests time out after at most 45 seconds.
Rejected or failed model requests do not refund reserved budgets. Quota counters
store HMAC pseudonyms rather than raw IP addresses. The provider still receives
model input; choose your model services according to your data requirements.
