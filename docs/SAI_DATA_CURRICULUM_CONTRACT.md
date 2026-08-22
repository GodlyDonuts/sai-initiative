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
6. split train and development by frozen document identity only after the
   global duplicate-family filter, preserving phase order in training;
7. tokenize losslessly with the selected tokenizer and preserve document
   boundaries;
8. compare the curriculum against a same-document order control at small scale;
9. promote the data schedule only if held-out NLL and source-disjoint real
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

Token-stream receipts bind the actual token and UTF-8-byte contribution of each
phase at every declared training prefix. Active phases at any prefix must form
a contiguous prerequisite prefix: `grounding`, then `integration`, then
`reasoning`, then `specialization`. The early 125M/250M/375M-token checkpoints
are not required to contain advanced material prematurely. The complete 500M
stream must contain all four phases.

For the first 499,998,720-token comparison, the freezer enforces exact phase
budgets on sequence boundaries rather than taking a blind prefix of the much
larger admitted corpus:

| Phase | Sequences | Tokens | Share |
|---|---:|---:|---:|
| grounding | 48,896 | 100,139,008 | 20.03% |
| integration | 60,928 | 124,780,544 | 24.96% |
| reasoning | 73,216 | 149,946,368 | 29.99% |
| specialization | 61,100 | 125,132,800 | 25.03% |

The first three cumulative phase boundaries are 48,896, 109,824, and 183,040
sequences: exactly optimizer updates 191, 429, and 715 at 256 sequences per
update. No gradient accumulation window therefore mixes examples from adjacent
difficulty phases. The final 244,140-sequence boundary ends with the declared
172-sequence partial update and preserves the exact 499,998,720-token budget.

The receipt records documents skipped after each phase budget, at most one
truncated tail document per phase, exact emitted phase sequences/tokens/bytes,
and every declared evaluation prefix. Without this budget, a 500M-token cutoff
of the 2.1M-document curriculum could end before the late phases and would not
constitute the intended experiment.

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

The train/development split is performed after this global filter. Every
accepted identity is assigned exactly once by a frozen hash modulus; validation
replays the complete curriculum against both outputs. This prevents the model
from being evaluated on an earlier corpus slice that may overlap the enlarged
training pool.

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

## Do not conflate three curriculum hypotheses

The current experiment tests only a **surface-complexity curriculum**. Its
signals are reproducible and model independent, which makes the contrast clean,
but a document with short sentences can still presuppose concepts that have not
been introduced. Conversely, technical vocabulary does not prove that a
document is pedagogically advanced. A positive or negative result for this
schedule therefore cannot settle the broader data-order question.

Two later hypotheses must remain separate factors:

1. A **semantic-prerequisite curriculum** binds a frozen concept taxonomy and
   directed acyclic prerequisite graph. Every admitted document must record
   concept identities, evidence spans, annotation method and confidence. The
   receipt must measure first exposure, prerequisite coverage before dependent
   exposure, later rehearsal, domain balance, and unresolved prerequisite
   violations. A concept such as color composition cannot be credited as
   ordered merely because its prose is readable; the primitive colors and their
   representations must have measurable prior coverage. No model-generated
   label may be accepted without a prospectively frozen annotator identity and
   an independently audited human or deterministic validation sample.
2. A **model-centric learnability curriculum** uses a separately frozen small
   checkpoint to estimate example learnability, influence, loss, or gradient
   noise. It may be more faithful to the learner than human readability, but it
   risks checkpoint-specific selection and must never use terminal benchmark
   answers or the treatment model's later state. Its scorer, checkpoint, sample
   identities, and score-to-order policy must be frozen before the comparison.

The prospective semantic boundary is executable. `sai-validate-prerequisites`
validates `sai-semantic-prerequisite-taxonomy-v1`: all five Sai domains, a
cycle-free concept graph, non-placeholder annotator/policy/audit identities,
and explicit minimum prior-document counts. The progression analyzer consumes
annotations in the exact curriculum document order, counts only evidence above
the frozen confidence floor, and checks prerequisites before updating the
current document's exposures. Consequently, teaching a primitive and its
dependent operation in the same document does not masquerade as prior coverage.
Its self-hashed report binds the taxonomy, ordered document identities, entire
annotation population, first exposure, per-phase coverage, missing concepts,
and every violation. It remains prospective and cannot authorize training.

Each lane must be compared against an exact same-sequence-multiset order control
and against the surviving surface schedule. Token identities, masks,
initialization, optimizer, training budget, evaluation rows, and modeled compute
remain fixed. The lanes cannot be bundled with a tokenizer or architecture
change. Promotion requires improvement on held-out likelihood and broad,
source-disjoint capability with no declared domain regression. A result at
100M parameters authorizes only the next data experiment; it never authorizes
4B training directly.

This separation is consistent with recent primary evidence: conventional
difficulty curricula can improve early and mid-training but their lasting gains
depend on the signal and pacing ([Zhang et al., 2025](https://arxiv.org/abs/2506.11300));
model-centric influence ordering can outperform human-centered difficulty on
limited-data pretraining ([Schoenegger et al., 2025](https://arxiv.org/abs/2508.15475));
and 2026 learning-dynamics experiments suggest that curricula may primarily
reduce within-phase gradient noise rather than create entirely new acquisition
phases ([Elgaar and Amiri, 2026](https://arxiv.org/abs/2601.21698)). These are
hypotheses to reproduce under Sai's exact data and compute controls, not borrowed
performance claims.
