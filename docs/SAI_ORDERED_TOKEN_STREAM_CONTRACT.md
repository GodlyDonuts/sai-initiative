# Sai Ordered Token-Stream Contract

Status: packer and replay validator implemented. No real corpus has been frozen.

## Why this exists

Architecture comparisons are invalid if models see different document order,
different source bytes, or accidental targets across document boundaries.
`sai-freeze-token-stream` turns an explicit sequence of decontaminated JSONL
sources into a deterministic binary stream shared by every Sai 100M arm.

This is data preparation only. The command writes no checkpoint, performs no
optimizer update, submits no GPU job, and emits `training_authorized=false`.

## Input row

Each nonblank input line has schema `sai-pretraining-document-v1` and contains:

- exact UTF-8 `text`;
- dataset, row ID, license, and one primary domain (`english`, `code`, `math`,
  `science`, or `technical`);
- `benchmark_disjoint=true`; and
- a SHA-256 receipt for the decontamination evidence.

The freezer validates or derives a content/provenance identity, drops duplicate
documents, and excludes malformed or unverified rows. Source file order and
line order are the training-stream order; every source path, byte count, and hash
is bound into the receipt.

## Tokenizer requirements

The tokenizer is loaded only from an existing local directory with
`local_files_only=true`, `trust_remote_code=false`, and a fast offset mapping.
Its exact regular-file tree is hash-bound. Links and special files are rejected.
For every accepted document, token offsets must cover the source contiguously,
token IDs must stay inside the vocabulary, and decoding must reproduce the exact
input string. An explicit EOS token is appended without claiming extra source
bytes.

## Binary format

Every shard has two files:

- `*.tokens.u32le`: fixed-length, little-endian unsigned 32-bit token IDs;
- `*.starts.bitset`: one LSB-first bit per token, set when a new causal segment
  begins.

Position zero is always a segment start. A new document sets another start bit.
If a long document crosses a packed-sequence boundary, position zero in the next
sequence resets the model because no state crosses sequences.

The same bitset deterministically yields both model segment IDs and the causal
loss mask. A target is trainable only when the next token is not a segment start;
the last position is always masked. This prevents a model from being rewarded
for predicting the first token of one document from the final token of another.

## Exact byte prefixes

Fast-tokenizer offsets allocate each source UTF-8 byte to the first emitted token
that completes its covered span. At every requested packed-sequence boundary,
the receipt records the exact cumulative admitted UTF-8 bytes. Sai's iso-data
arms therefore select the same stream prefix and bytes. Its iso-FLOP arms select
the family-specific sequence counts already computed by the exact FLOP planner
and report the corresponding byte exposure honestly.

For the current 100M contract, the required prefix sequence counts are:

- gated GQA: `678678` for iso-FLOP;
- GDN hybrid: `794277` for iso-FLOP;
- KDA/MLA hybrid: `797472` for iso-FLOP; and
- all families: `1048576` for iso-data.

## Replay validation

The validator reopens every source and output shard, verifies hashes and exact
file membership, checks byte geometry and shard order, parses every segment
bitset, rejects nonzero padding, and recomputes the source and ordered-stream
identities. Missing, extra, linked, mutated, reordered, or malformed artifacts
fail closed.

The remaining work is empirical: build the winning tokenizer candidates, freeze
the real admitted corpus, and qualify the resulting stream before an official
training order.
