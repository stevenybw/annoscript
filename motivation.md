# Combining decisions and generation

Structured annotation often combines fixed choices, numerical ratings, free
text and exact source spans. These operations need different output contracts
and may benefit from different model backends.

Lev expresses those dependencies as a Python program, compiles the program into
IR, and lowers it into an execution plan. A decision can control a branch while
independent generation tasks run concurrently or share a physical request.

The core library supplies compilation, scheduling, backend protocols, hard
constraints and durable execution. Applications provide their own schemas,
prompts, examples, datasets, credentials and deployment configuration.

The synthetic document example illustrates the distinction: a closed category
and binary decision select a branch; generation writes a title and tags; extraction
copies spans from the current document. An explicit constraint checks that each
returned span occurs in that document.
