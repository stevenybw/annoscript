# Pinned Substring decoder

Apply `substring-nonempty.patch` to the revision in `pin.json`, including that commit’s submodules, and build with apache-tvm-ffi 0.1.10. Install the resulting wheel; a plain PyPI xgrammar release does not implement `allow_empty=false`. The patch defaults to upstream nullable behavior and encodes the nonempty flag as a negative sentinel before chunk lengths, preserving the old default serialized layout. Old compiled grammar caches must not be reused across builds.

Lev `LLMAPIBackend(extract_mode="substring_chars", profile="qwen_vllm")` opts in. Environment configuration accepts `LEV_GENERATE_EXTRACT_MODE`. Job releases additionally require a `generation.structured_output_revision` build digest. The default `validated` mode retains the previous wire format and runtime-only substring validation. Other profiles cannot enable this mode.

Run `python -m lev.substring_tests` with the pinned wheel and `python -m unittest lev.test_substring_backend` for wire tests. The source characters are Unicode code points, serialized as indivisible canonical JSON escape chunks. Empty sources produce only an empty array; business count/length validation remains in runtime. Different extraction sources never share a union grammar.

Qwen tokenizer input normalization may compose combining marks. Exact-output token tests must disable normalization in a separate target encoder and verify a byte-identical decode roundtrip; do not change the serving tokenizer or source text.

## P0 bounded outputs

`generation.request_policy` opts into explicit sampling, default token budgets, pinned tokenizer preflight, and decoded code-point bounds. See `pin.json` for the decoder build. The public task and result schemas are unchanged. Install `requirements-p0-tokenizer.txt` on the Lev worker and set `LEV_GENERATE_TOKENIZER_DIR` to a local copy of the bound tokenizer artifacts. The path is deployment configuration; the release contains their revision and SHA-256 digests. A mismatch fails backend initialization.

The P0 patch adds `Substring(..., max_chunks=N)` with a nonnegative integer bound. It constructs only reachable `(suffix state, consumed chunks)` pairs before expanding whole chunks into bytes. A suffix state's longest string length is not the consumption counter. With `allow_empty=false`, empty chunks are removed, as in the preceding extension; nullable mode counts empty chunks as input blocks. Omitting `max_chunks` preserves the earlier language and serialized representation. `max_chunks=0, allow_empty=false` is the empty language. Parsed grammars, printed EBNF and serialized grammars carry the bound.

In Lev each source code point is one indivisible canonical JSON chunk. Count constraints use bounded repetitions; generated JSON strings count raw Unicode code points, complete BMP escapes, or a complete surrogate pair as one character. Standalone `generate` remains raw text. Rule reuse includes source and all effective bounds. Runtime validation remains mandatory. Empty sources with positive `min_items` fail locally; no HTTP request or retry is made.

The patch also sorts the cache's canonical-state lookup table once and uses binary search instead of scanning the complete mapping for every token-mask state. Canonical IDs and cache keys are unchanged. This avoids the quadratic lookup cost exposed by the bounded product automaton; it does not alter scheduling, prompts, or accepted output strings.

Run `python -m unittest lev.test_generation_policy` for wire and worker integration tests, `python -m lev.p0_grammar_tests` and `python -m lev.substring_tests` in the pinned decoder environment. Performance acceptance must include the first cache-enabled compilation, not just cache-disabled compilation and subsequent hits. Do not deploy a candidate that fails the stated memory or 60-second gate.
