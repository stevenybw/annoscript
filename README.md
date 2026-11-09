# AnnoScript: Programmable AI Annotation

AnnoScript compiles Python annotation programs into a serializable intermediate
representation and an execution plan. Decision operations and generation
operations can use separate model backends while sharing program state.

Quick Start and clients support Python 3.10+. Self-hosted servers and direct
provider execution require Python 3.12+.

## Quick Start

Use Python **3.10+**. Two commands install AnnoScript and run the packaged synthetic
example with real model results, without deploying a server or obtaining model keys:

```bash
pip install annoscript
python -m annoscript_examples.quick_start
```

The example runs Python locally and sends model-operation inputs to the official
anonymous gateway at **https://annoscript.chukonu.ai**, the same gateway used by the
[Playground](https://annoscript.chukonu.ai/#playground). Shared limits apply: 1,000
sessions/day globally, 50/IP/day, 5/IP/minute, four active sessions; each session
lasts 120 seconds with 12 requests and an 8,192 output-token budget. Limits can
make the demo unavailable. Do not send confidential inputs to the hosted demo.

The same packaged program is available through the Python API:

```python
from annoscript.client import AnnoClient
from annoscript_examples.request import annotate
from annoscript_examples.quick_start import SAMPLE

with AnnoClient.server() as client:
    with client.submit(annotate, [SAMPLE]) as results:
        for result in results:
            if result.status != "success":
                raise RuntimeError(result.error.message)
            print(result.value)
```

Programs, local validation, compilation and scheduling execute on your computer.
Only validated model operations go to the server, which handles model calls,
authentication and quotas. The client creates a session at its first model
operation and closes it on exit. Quota, budget and timeout errors do not trigger
automatic retries or session renewal. No provider credentials are read or sent.

## Deploy Your Own AnnoScript Server

For your own models, access policy and storage, follow the advanced
[deployment tutorial](docs/deploy-server.md). It includes Docker Compose,
configuration, persistent quota storage, health checks and HTTPS. It uses the
same gateway architecture; user Python stays on the user's machine or browser.
Connect with `AnnoClient.server("https://annotations.example.com", auth_token=...)`.
The server access token is distinct from model provider keys kept on the server.

Existing local execution, durable-job clients and serialized protocols remain
compatible. Quick Start and `AnnoClient.server()` also support Python 3.10+.
Self-hosted servers and direct provider execution require Python 3.12+.
Optional extras are `annoscript[server]`
for hosting and `annoscript[tokenizer]` for pinned local tokenizer support.

## Offline tools and synthetic example

`annoscript_examples.request.annotate` classifies a synthetic document, conditionally
generates a title, and produces tags and source spans. The example demonstrates
all six operators without requiring an application dataset.

```bash
python -m annoscript.inspect annoscript_examples/request.json --show-source
python -m annoscript.inspect annoscript_examples/request.execution_plan.json
python -m annoscript.inspect annoscript_examples/request.execution_plan.json --format html -o /tmp/lev-plan.html
```

To regenerate the saved example with Python 3.12:

```python
from pathlib import Path
from annoscript.compiler import compile_program
from annoscript.plan_lowering import lower_to_plan
from annoscript_examples.request import annotate

program = compile_program(annotate)
Path("annoscript_examples/request.json").write_text(program.to_json(indent=2) + "\n")
Path("annoscript_examples/request.execution_plan.json").write_text(lower_to_plan(program).to_json(indent=2) + "\n")
```

## Configuration

Install `requirements.txt` for the client and `requirements-server.txt` for
local execution or the durable job server. Copy `.credential.example` to an
ignored local file and supply the desired endpoints and model names.
Environment variables override values from an explicitly supplied environment file.
Canonical `ANNO_*` keys take precedence over legacy `LEV_*` aliases within the same
layer. Process environment values override file values regardless of their prefix;
explicit API arguments take precedence over both. Credentials stay outside packages.

| Variable | Purpose |
| --- | --- |
| `ANNO_DECISION_BASE_URL` | Decision provider URL |
| `ANNO_DECISION_AUTH_TOKEN` | Decision provider credential |
| `ANNO_DECISION_MODEL` | Decision model identifier |
| `ANNO_GENERATE_BASE_URL` | Generation provider URL |
| `ANNO_GENERATE_AUTH_TOKEN` | Generation provider credential |
| `ANNO_GENERATE_MODEL` | Generation model identifier |
| `ANNO_GENERATE_PROFILE` | `legacy` (default) or `qwen_vllm` wire protocol |
| `ANNO_GENERATE_TOKENIZER_DIR` | Caller-provided tokenizer directory when required by policy |

Keep live service settings and credentials outside version control.
The `legacy` profile name describes a supported request protocol.

```bash
python -m annoscript_examples.run --env-file .credential --inputs /tmp/documents.json --mode speculative
```

This command calls the caller's configured services. Inspecting IR, previewing
resolved requests and running the offline tests do not call model services.

## Execution and backends

Programs use `choice`, `noul`, `score`, `generate`, `generate_list` and `extract`.
`constrain` declares hard runtime constraints. `system_prompt` supplies shared
instructions; preconditions validate inputs locally before dispatch.

`AnnoClient` supports strict, eager and speculative execution. The runtime
handles bounded concurrency, batching, cancellation, repair and result ordering.
Examples, field names, nullable output and per-call requirements are included in
the IR and execution-plan bindings. Decision and generation request adapters
are explicitly selectable and can be inspected without dispatch.

```bash
python -m annoscript.preview annoscript_examples/request_preview.decision.json
python -m annoscript.preview annoscript_examples/request_preview.generation.json
```

Remote execution exchanges serialized IR. Durable jobs bind an immutable
release to resumable, idempotent result retrieval. See [remote protocol](remote_protocol.md)
and [durable jobs](job_protocol.md) for request formats and error semantics.

## Offline verification

```bash
ANNO_RUN_LIVE_TESTS=0 python -m annoscript.client_tests -q
ANNO_RUN_LIVE_TESTS=0 python -m unittest discover -q
python -m annoscript.grammar_tests -v
python -m annoscript.cross_version_tests --client-python /path/to/python3.10 --server-python /path/to/python3.12
python -m annoscript.job_cross_version_tests --client-python /path/to/python3.10 --server-python /path/to/python3.12
```

The CPU grammar suite requires Python 3.12 and `requirements-test-grammar.txt`.
Cross-version checks use loopback HTTP and deterministic model doubles. They do
not require provider credentials or download models. Additional pinned decoder
acceptance instructions are in [third_party/xgrammar](third_party/xgrammar/README.md).

See [motivation](motivation.md) and [operator interfaces](design_api.md) for the
design rationale and API contracts.
