# Lev

Lev compiles Python annotation programs into a serializable intermediate
representation and an execution plan. Decision operations and generation
operations can use separate model backends while sharing program state.

The client supports Python 3.10+. Runtime and server execution require Python 3.12+.

## Synthetic example

`lev_examples.request.annotate` classifies a synthetic document, conditionally
generates a title, and produces tags and source spans. The example demonstrates
all six operators without requiring an application dataset.

```bash
python -m lev.inspect lev_examples/request.json --show-source
python -m lev.inspect lev_examples/request.execution_plan.json
python -m lev.inspect lev_examples/request.execution_plan.json --format html -o /tmp/lev-plan.html
```

To regenerate the saved example with Python 3.12:

```python
from pathlib import Path
from lev.compiler import compile_program
from lev.plan_lowering import lower_to_plan
from lev_examples.request import annotate

program = compile_program(annotate)
Path("lev_examples/request.json").write_text(program.to_json(indent=2) + "\n")
Path("lev_examples/request.execution_plan.json").write_text(lower_to_plan(program).to_json(indent=2) + "\n")
```

## Configuration

Install `requirements.txt` for the client and `requirements-server.txt` for
local execution or the durable job server. Copy `.credential.example` to an
ignored local file and supply the desired endpoints and model names.
Environment variables override values from an explicitly supplied environment file.

| Variable | Purpose |
| --- | --- |
| `LEV_DECISION_BASE_URL` | Decision provider URL |
| `LEV_DECISION_AUTH_TOKEN` | Decision provider credential |
| `LEV_DECISION_MODEL` | Decision model identifier |
| `LEV_GENERATE_BASE_URL` | Generation provider URL |
| `LEV_GENERATE_AUTH_TOKEN` | Generation provider credential |
| `LEV_GENERATE_MODEL` | Generation model identifier |
| `LEV_GENERATE_PROFILE` | `legacy` (default) or `qwen_vllm` wire protocol |
| `LEV_GENERATE_TOKENIZER_DIR` | Caller-provided tokenizer directory when required by policy |

Keep live service settings and credentials outside version control.
The `legacy` profile name describes a supported request protocol.

```bash
python -m lev_examples.run --env-file .credential --inputs /tmp/documents.json --mode speculative
```

This command calls the caller's configured services. Inspecting IR, previewing
resolved requests and running the offline tests do not call model services.

## Execution and backends

Programs use `choice`, `noul`, `score`, `generate`, `generate_list` and `extract`.
`constrain` declares hard runtime constraints. `system_prompt` supplies shared
instructions; preconditions validate inputs locally before dispatch.

`LevClient` supports strict, eager and speculative execution. The runtime
handles bounded concurrency, batching, cancellation, repair and result ordering.
Examples, field names, nullable output and per-call requirements are included in
the IR and execution-plan bindings. Decision and generation request adapters
are explicitly selectable and can be inspected without dispatch.

```bash
python -m lev.preview lev_examples/request_preview.decision.json
python -m lev.preview lev_examples/request_preview.generation.json
```

Remote execution exchanges serialized IR. Durable jobs bind an immutable
release to resumable, idempotent result retrieval. See [remote protocol](remote_protocol.md)
and [durable jobs](job_protocol.md) for request formats and error semantics.

## Offline verification

```bash
LEV_RUN_LIVE_TESTS=0 python -m lev.client_tests -q
LEV_RUN_LIVE_TESTS=0 python -m unittest discover -q
python -m lev.grammar_tests -v
python -m lev.cross_version_tests --client-python /path/to/python3.10 --server-python /path/to/python3.12
python -m lev.job_cross_version_tests --client-python /path/to/python3.10 --server-python /path/to/python3.12
```

The CPU grammar suite requires Python 3.12 and `requirements-test-grammar.txt`.
Cross-version checks use loopback HTTP and deterministic model doubles. They do
not require provider credentials or download models. Additional pinned decoder
acceptance instructions are in [third_party/xgrammar](third_party/xgrammar/README.md).

See [motivation](motivation.md) and [operator interfaces](design_api.md) for the
design rationale and API contracts.
