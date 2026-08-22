# Sai Data Curriculum Contract

Sai treats the data distribution and its presentation order as architecture-level
decisions. A model must first acquire language and symbolic primitives before a
large fraction of its updates demand technical composition or multi-step
reasoning. A hash-pinned web-text prefix is raw material, not a qualified
training curriculum.

## Admission sequence

No Sai optimizer run may consume a corpus merely because it is readable. The
required order is:

1. verify every upstream source byte, revision, license declaration, and row
   identity;
2. reject pathological text and benchmark overlap;
3. reject exact duplicates and high-confidence near-duplicate families;
4. score every remaining document with the frozen, model-independent complexity
   rubric;
5. build and replay the four-phase curriculum receipt;
6. tokenize losslessly with the selected tokenizer and preserve document
   boundaries;
7. compare the curriculum against a same-document order control at small scale;
8. promote the data schedule only if held-out NLL and source-disjoint real
   capability measurements support it.

The current FineWeb-Edu source is therefore only a candidate pool. Its upstream
educational score and Sai's first-pass hygiene filter are useful but insufficient
to authorize training.

## Frozen bands and phases

`sai.data.curriculum` assigns four bands using observable surface complexity:
sentence length, word length, long-word fraction, bounded lexical diversity,
symbol and digit density, code-line density, reasoning markers, technical
markers, and document length. The bands are:

- `foundation`: short, direct language and low symbolic load;
- `composition`: ordinary explanatory prose joining several facts;
- `reasoning`: causal, conditional, quantitative, or evidentiary composition;
- `specialization`: dense mathematical, scientific, code, or technical text.

These names are hypotheses, not semantic ground truth. The receipt explicitly
states that the score is a deterministic surface proxy. This prevents a neat
heuristic from being misreported as proof of a prerequisite graph.

Documents are emitted in four phases: `grounding`, `integration`, `reasoning`,
and `specialization`. Each band's phase shares sum to one:

| Band | Grounding | Integration | Reasoning | Specialization |
|---|---:|---:|---:|---:|
| foundation | 50% | 25% | 15% | 10% |
| composition | 20% | 35% | 30% | 15% |
| reasoning | 5% | 20% | 40% | 35% |
| specialization | 0% | 5% | 30% | 65% |

This is paced exposure rather than a brittle one-way staircase: foundational
material is rehearsed in every later phase, while specialized material is
absent from grounding and concentrated only after composition and reasoning
have been repeatedly presented.

## Quality and duplicate boundaries

The second-pass quality floor rejects very short documents, low-Latin text in
the declared English population, severe lexical repetition, pathological
character runs, and URL-heavy pages. The curriculum also applies a deterministic
five-word-shingle locality-sensitive sketch. Six of eight bucket minima must
match before a later document is rejected. This is deliberately labeled a
high-confidence filter, not exhaustive semantic deduplication.

The create-only receipt binds the source and decontamination receipt, policy,
rejection reasons, exact band populations, every phase's population and ordered
identity digest, order-independent accepted/emitted identity fingerprints, and
the output bytes. Validation reopens the source evidence and replays every
output row, band, difficulty, duplicate decision, phase mean, and identity.

## Empirical veto

A valid curriculum receipt does not prove that easy-to-hard order improves a
model and does not authorize 4B training. The first experiment must use the same
admitted documents, tokenizer, initialization, optimizer, update count, and
modeled compute in two small-model arms differing only in order:

- the frozen prerequisite-to-specialization curriculum;
- a prospectively frozen deterministic order control.

The curriculum is retained only if it improves or preserves held-out token NLL
and UTF-8-byte-normalized NLL, shows nonnegative source-disjoint aggregate
capability, and introduces no declared domain regression. Tokenizer choice,
content filtering, and ordering must be tested as separate factors instead of
being bundled into an uninterpretable win or loss.

The implemented primary control permutes the exact packed 2,048-token sequence
records after the curriculum stream is frozen. Each record includes both token
IDs and its document-boundary bitset, so this construction preserves the exact
token/target/mask multiset and changes only sequence presentation order. The
control receipt binds the parent stream, frozen seed `2026082201`, exact
permutation, fixed-point count, and a sorted sequence-record multiset digest;
validation replays every output record against its declared parent index. This
is stronger than independently shuffling documents, which could change which
tail tokens survive a fixed-token cutoff.
