# Operator interfaces

The Python interfaces are defined by `lev.lev.Lev`. Programs receive their
current input as `lev.state`; inputs are read-only within a compiled program.

| Operator | Result field | Purpose |
| --- | --- | --- |
| `choice(instructions, criteria)` | `.choice` | Choose one named option |
| `noul(instructions)` | `.noul` | Return a binary probability |
| `score(instructions, criteria)` | `.score` | Return a rating |
| `generate(instructions, ...)` | `.text` | Generate a string, optionally nullable |
| `generate_list(instructions, ...)` | `.items` | Generate a list of strings |
| `extract(instructions, ...)` | `.items` | Copy strings from the selected source |

All six operators accept keyword-only `examples`, `requirements` and
`field_name`. An example contains its own state and answer; it never expands the
current input's extraction source. Requirements are nonblank strings grouped by
exact equality in generation requests. Field names identify tasks on the wire
and must be distinct within a physical request.

Generation supports advisory length hints and an explicit `max_num_tokens`
budget. Item counts, types, nullable output and source membership are separate
contracts. `generate(nullable=True)` may return JSON null; that is distinct from
the string `"null"`. Extraction source selection uses `source_field` or explicit
`source_fields` according to the existing backend contract.

`constrain(condition, message=...)` is a compiled eDSL intrinsic. It declares a
hard constraint and cannot be called as a standalone Python solver. Constraints
remain active after optimization and apply on the execution path where declared.

`system_prompt` attaches shared instructions. `precondition` attaches local
input validation. Compilation never executes the annotation function or its
preconditions. `LevClient.submit` validates inputs before dispatch and returns
indexed success or error results; remote clients serialize the compiled IR.

See `lev_examples.request` for a complete synthetic program. `lev.preview`
renders already resolved calls using the real request builders, without model
inference or reading a credential file.
