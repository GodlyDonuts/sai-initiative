# Sai Tokenizer Qualification Contract

Status: implemented qualifier. The exact 48K fixed-geometry default is
mechanically qualified, but no tokenizer capacity has won a capability
comparison and training remains unauthorized.

## Decision being measured

Sai compares exactly three tokenizer capacities: 64K, 48K, and 32K. Each
candidate is loaded from an immutable local tree and evaluated on the same
ordered benchmark-disjoint source corpora and the same protected-string suite.
The report separates tokenizer fertility from later parameter reallocation; it
does not claim that a smaller vocabulary improves model capability.

The executable entry point is `sai-qualify-tokenizers`. It accepts exactly one
local tree for each candidate, one or more admitted JSONL corpora, the protected
suite, and fresh report/selection paths. Network loading and remote tokenizer
code are disabled.

## Required source evidence

Every corpus row must satisfy the same `sai-pretraining-document-v1` contract as
the ordered token-stream freezer:

- nonempty dataset, row identity, license, and text;
- one of English, code, math, science, or technical as the primary domain;
- `benchmark_disjoint=true` with a well-formed evidence SHA-256;
- a correct optional document identity; and
- no duplicated normalized document identity across the admitted files.

The report binds each corpus path, byte size, SHA-256, order, domain counts, and
the protected-suite receipt into one corpus identity.

## Candidate admission

Each tokenizer must have the declared exact vocabulary size, unique contiguous
IDs from zero, an EOS token inside a nonempty special-token set, and the same
logical special-token contract as the other candidates. The on-disk tokenizer
must declare either model-level byte fallback or a ByteLevel pre-tokenizer and
ByteLevel decoder. A Python attribute alone is not accepted by the CLI.

Every corpus and protected string is encoded without added special tokens and
decoded without cleanup. Qualification requires:

- byte-exact Unicode round trips;
- zero unknown-token emissions;
- zero empty encodings;
- no out-of-vocabulary token IDs; and
- complete vocabulary and protected-category coverage checks.

The frozen protected suite includes ASCII, English prose and punctuation, code,
indentation, identifiers, URLs and paths, numbers and units, Greek, mathematical
operators, LaTeX, scientific notation, and arbitrary Unicode fallback cases.
Multilingual token efficiency may be deprioritized, but multilingual and
otherwise unusual Unicode remains lossless.

## Measurements and selection

For every candidate, the report records UTF-8 bytes, token counts, tokens per
1,000 UTF-8 bytes, bytes per token, and per-domain fertility. This makes the
comparison independent of character-count conventions.

The CLI emits a 48K selection receipt only when all three candidates qualify.
That fixed default reflects the prospective 100M plan; it is not an empirical
winner claim. The receipt binds the candidate tree identity, corpus identity,
tournament report, fallback behavior, round-trip result, protected fertility,
and special-token preservation.

Both the tournament report and the fixed-default receipt explicitly set
`training_authorized=false`. Building actual candidates, selecting a different
capacity from evidence, freezing the production token stream, and starting any
optimizer remain separate actions. The user's official training order is still
required.

The completed tournament measured an English-labeled corpus only. Its exact
results and the remaining broad-domain/capability selection requirements are
recorded in
[`SAI_TOKENIZER_EVIDENCE_AUDIT.md`](SAI_TOKENIZER_EVIDENCE_AUDIT.md). The 48K
receipt must not be described as an empirical winner receipt.
The current individual-digit pre-tokenization policy is likewise a hypothesis,
not a selected numeric-capability result.
