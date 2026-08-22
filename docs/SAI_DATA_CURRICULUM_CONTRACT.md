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

Future semantic-curriculum and matched-control runs may additionally freeze
predeclared model-only milestone snapshots at exact optimizer steps. Each
create-once snapshot binds the complete run identity, cumulative sequence and
target counters, stream cursor, canonical model-state hash, and artifact hash;
it contains neither optimizer nor RNG state and is evaluation-only. Treatment
and control must use the identical milestone-step set so observation I/O does
not become an unacknowledged factor. Milestones are bound into the run identity,
validated across resume, and cannot be selected after observing a learning
curve. They do not alter or retroactively apply to the live frozen 500M-token
surface-order experiment.

`sai-evaluate-curriculum-milestones` makes those snapshots operational. For
each arm it reconstructs the exact initialization, replays every model-only
snapshot, reopens the terminal checkpoint, and scores the identical
phase-stratified development stream at initialization, every phase boundary,
and termination. The evaluator performs zero optimizer steps and zero backward
calls. `sai-compare-curriculum-milestones` then requires identical initial
state/evidence, development population, milestone schedule, model identity,
and observation geometry across the curriculum and order-control arms. It
reports, separately for every phase, acquisition by the phase's completion
boundary, curriculum-versus-control likelihood at that boundary, subsequent
forgetting relative to control, and terminal retention. A mechanics pass still
cannot promote data or authorize 4B training; real source-disjoint benchmark
confirmation remains mandatory. The live frozen 500M-token surface-order jobs
predate these snapshots and therefore provide final phase retention but not a
retrospective phase-boundary learning curve.

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

The held-out likelihood stream is itself phase-stratified. It contains exactly
256 packed 2,048-token sequences from each of grounding, integration,
reasoning, and specialization, for 1,024 sequences total. Taking the first
1,024 sequences from the phase-ordered development corpus is forbidden because
that would measure almost exclusively grounding material and could make a
curriculum appear successful while concealing regressions on later reasoning or
specialization. The stream receipt binds each phase's exact sequence, token,
and UTF-8-byte contribution. Evaluation reports aggregate and per-phase target
NLL, perplexity, and NLL per admitted UTF-8 byte from the same forward passes.
An aggregate improvement is vetoed if any phase's target-normalized or
byte-normalized NLL regresses against the exact-order control; easier grounding
examples cannot conceal damage to reasoning or specialization.

Real development benchmark evaluation must preserve the same source-disjoint
boundary after curriculum selection. The evaluator accepts either the original
decontaminated source directly or an exact, completed lineage:
decontaminated source -> qualified curriculum -> qualified train/development
split -> frozen training stream. The split job, receipt file hash, receipt self
hash, parent curriculum receipt, train output identity, and stream source
identity must all agree. A derived training subset is never treated as an
unrelated source merely because its file hash differs from its parent.

If held-out NLL and all four phase vetoes pass, retention still requires the
full source-disjoint MMLU-Pro (12,032 rows) and MuSR (756 rows) development
populations for both matched checkpoints. `sai-compare-curriculum-benchmarks`
pairs every row, requires identical scoring, decoding, source, runtime, and
configuration bindings except checkpoint identity, and freezes the decision
before results:

- neither benchmark accuracy may regress;
- at least one benchmark and the unweighted macro must improve;
- the 95% lower bound from 10,000 deterministic paired, domain-stratified
  bootstrap replicates must be strictly positive;
- no reported domain may regress by more than one percentage point.

A favorable aggregate cannot override a benchmark, confidence, or domain
veto. Passing retains the data order for the next scale; it does not authorize
an architecture claim or 4B training.

The benchmark continuation is dependency-staged only after the held-out NLL
receipt passes. It evaluates each matched checkpoint with eight independent
single-H100 MMLU-Pro shards and one independent single-H100 MuSR job, merges
MMLU-Pro on CPU, and requires exact clean accounting for all eighteen GPU jobs
before the terminal CPU comparison. No benchmark job is submitted when the
held-out phase gate fails, and no benchmark result authorizes 4B training.

Before submitting the canary or either matched arm, the launcher requires at
least 24 GiB of hard-limit storage headroom and 10,000 free inodes. The exact
used/hard/headroom snapshot and thresholds are bound into dispatch and replayed
by the continuation. The same minimums are propagated to the canary and both
full H100 jobs, which recheck live quota immediately before model/data loading.
This is an evidence-preservation admission gate, not a scientific criterion.

The implemented primary control permutes the exact packed 2,048-token sequence
records after the curriculum stream is frozen. Each record includes both token
IDs and its document-boundary bitset, so this construction preserves the exact
token/target/mask multiset and changes only sequence presentation order. The
control receipt binds the parent stream, frozen seed `2026082201`, exact
permutation, fixed-point count, and a sorted sequence-record multiset digest;
validation replays every output record against its declared parent index. This
is stronger than independently shuffling documents, which could change which
tail tokens survive a fixed-token cutoff.

The launch-time bundle validator replays the qualified train/development split
once, binds the parent, control, and development stream source receipts to those
exact output hashes, then verifies the control permutation and record multiset.
It removes redundant full-source hashing without removing any evidence check;
the split replay remains the authoritative proof of source bytes.

## Do not conflate three curriculum hypotheses

The current experiment tests only a **surface-complexity curriculum**. Its
signals are reproducible and model independent, which makes the contrast clean,
but a document with short sentences can still presuppose concepts that have not
been introduced. Conversely, technical vocabulary does not prove that a
document is pedagogically advanced. A positive or negative result for this
schedule therefore cannot settle the broader data-order question.

The governing data doctrine is therefore **quality, coverage, then pedagogy**.
Quality removes incorrect, incoherent, contaminated, duplicated, commercial,
and otherwise harmful examples. Coverage verifies the intended capability and
domain mixture. Pedagogy orders the surviving material by actual conceptual
dependency and preserves rehearsal. None of these stages may borrow a pass from
another: a correct specialized proof can be pedagogically premature, and a
simple-looking passage can still depend on unintroduced concepts.

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

The executable scorer is `sai-score-learnability`. It uses a predeclared model-
only milestone and the terminal state from the same independent probe training
trajectory as the weak and strong states. It scores a separately frozen target
stream only after proving that no exact packed token-and-boundary record occurs
in both the probe-training and target streams. Every target sequence receives
weak and strong normalized NLL in integer microunits. The create-only two-file
score population binds the model states, completed probe result, target and
probe stream identities, tokenizer, runtime, evaluator, every record hash, and
the ordered score population. Scoring runs in inference mode and proves zero
optimizer steps, zero backward calls, unchanged model states, and unchanged RNG
state. Exact-record disjointness does not establish near-duplicate or semantic
disjointness; those remain separate mandatory source gates.

The scheduler `sai-build-learnability-curriculum` consumes one already-
qualified packed stream, the immutable score-population root, and one
prospectively frozen policy. Detached score rows are not admissible. The policy
binds the exact weak milestone, strong terminal checkpoint, tokenizer,
evaluator, runtime, phase and band populations, optimizer-aligned phase
boundaries, and a score-independent within-phase order. It forbids treatment
checkpoint state and terminal benchmark feedback.

The builder ranks identical packed records into `ready`, `developing`,
`challenging`, and `stretch` bands using strong-checkpoint normalized loss as
the primary current-difficulty signal and weak-to-strong improvement as the
secondary learning-progress signal,
allocates those bands across grounding, integration, reasoning, and
specialization under an exactly frozen matrix, and preserves ready-record
rehearsal in every later phase. Within each phase records are hash-ranked rather
than loss-ranked, so the changed factor is the phase mixture rather than a
continuous easiest-to-hardest micro-order. Validation reopens the parent,
policy, score receipt, and every score; recomputes every packed-record hash,
rank, band, allocation, permutation, and multiset; and proves that token IDs and
document boundary masks are unchanged. This schedule is explicitly model-
centric and does **not** prove semantic prerequisite ordering. It remains a
separate matched factor and authorizes neither training nor 4B.

Model-relative learnability and semantic prerequisite depth are deliberately
independent axes. A low-loss advanced example is not moved ahead of missing
primitives merely because the probe already knows it, and a prerequisite-
complete example is not presumed learnable merely because a taxonomy approves
its position. A production curriculum must satisfy the semantic graph first,
then pace records within admitted semantic regions using prospectively frozen
model evidence, while retaining foundational rehearsal in later phases.

The executable composition boundary is
`sai-compose-semantic-learnability`. It first replays the taxonomy, qualified
document curriculum, full annotation population, and semantic progression
report. The packed parent must bind that exact curriculum output and its
qualification receipt, use contiguous token-budgeted semantic phases, and have
zero skipped or truncated source documents in every phase. This last condition
is mandatory: a later concept cannot rely on prerequisite exposures that were
present in the audited document file but silently omitted by packing.

Only after semantic replay passes does the composer admit the independently
receipt-bound weak/strong score population. Within each semantic phase it ranks
current mastery by the strong checkpoint's normalized NLL first and uses
weak-to-strong improvement as the secondary learning-progress signal. This
prevents a persistently high-loss, non-improving example from being mislabeled
`ready` merely because its weak/strong difference is small. It divides records into
`ready`/`developing`/`challenging`/`stretch` bands separately inside each
semantic phase and hash-orders records inside a band. Every output phase is
proven to contain exactly the same packed-record identities as the corresponding
parent phase; only within-phase order changes. Validation reopens all semantic
and model evidence, replays every score and source record, and recomputes the
complete permutation and multiset. The composite remains a prospective matched
data factor and authorizes neither training nor 4B.

The prospective semantic boundary is executable. `sai-validate-prerequisites`
validates `sai-semantic-prerequisite-taxonomy-v3`: all five Sai domains, a
cycle-free concept graph, non-placeholder annotator/policy/audit identities,
explicit minimum prior-document counts, and frozen per-concept minimum exposure
counts for every curriculum phase. The first phase with a nonzero exposure
minimum is also the concept's earliest permitted phase: a confident advanced
concept exposure before that phase is a terminal progression violation, even
when its prerequisites have already appeared. The taxonomy also freezes a
maximum number of first-time concept exposures per document; exceeding it is a
separate concept-density violation, preventing a single advanced example from
compressing an entire conceptual layer. The progression analyzer consumes
annotations in the exact curriculum document order, counts only evidence above
the frozen confidence floor, rehashes every claimed non-overlapping character
span against the exact document text, and checks prerequisites before updating
the current document's exposures. Consequently, an unsupported concept label
cannot pass with a detached evidence hash or a token mention: every positive
label must bind at least 16 source Unicode codepoints of substantive evidence.
Teaching a primitive and its dependent operation in the same document does not
masquerade as prior coverage.
Its self-hashed report binds the taxonomy, ordered document identities, entire
annotation population, first exposure, per-phase coverage, missing concepts,
every prerequisite violation, and every missing rehearsal obligation. It
remains prospective and cannot authorize training.
`sai-validate-annotation-policy` prevents the policy hash from naming an empty
or permissive artifact. The exact prospective policy binds the candidate
concept list; requires explicit instruction or demonstrated use with a
source-verifiable Unicode-codepoint span; omits ambiguous or unsupported
labels; sets the confidence floor to 0.8; forbids a concept introduced in the
same document from satisfying its own prerequisite; derives phase only from
the curriculum receipt; and requires blind independent review with the frozen
five-percent disagreement ceiling. It authorizes neither training nor 4B.
`sai-validate-prerequisites build-taxonomy` creates that taxonomy only from a
candidate concept list plus real, immutable annotator-identity, annotation-
policy, and audited-sample artifacts. Their file hashes become the annotation
identities; callers cannot satisfy the production path with invented digest
strings. The audited-sample receipt must bind the exact annotator and policy,
cover at least 100 reviewed documents, reproduce its disagreement arithmetic,
and pass a prospectively capped disagreement rate no greater than five percent.
The create-once output is validated before atomic publication and retains both
training authorizations as false.
`sai-select-prerequisite-audit` freezes the review population before annotation.
It deterministically selects the eight lowest salted identity hashes from each
pedagogically valid combination of four curriculum phases and four surface
bands: 120 documents in 15 equal strata. `grounding:specialization` is excluded
because the qualified curriculum contract requires that cell to be empty; a
review geometry that demanded rows from it would contradict the progression it
audits. The selector revalidates the parent curriculum, preserves full text and
source identity, publishes create-once evidence, and can replay the selection
exactly. Surface bands are used only to prevent an easy or late-phase sample
from dominating review; they are not treated as semantic difficulty or domain
labels.
`sai-select-prerequisite-development-audit` applies the identical salt,
phase/band stratification, and eight-document geometry to the already-qualified
source-disjoint development split. It reopens the split self-hash, exact
development file hash, phase counts, and progression status, then recomputes
the 120-row selection from only the development artifact. This avoids replaying
the 9.5 GB training curriculum several times merely to form a review packet and
prevents the reviewed sample from being part of optimizer training. The output
remains unreviewed and authorizes no training.
`sai-review-prerequisite-audit` then compares the prospective annotator against
an independently identified reviewer on those exact 120 documents. Both sides
must provide canonical concept sets with nonempty evidence spans whose hashes
reopen against the frozen source text. The receipt binds and reopens the sample,
candidate concept list, identities, policy, and both annotation files; it
computes document-level concept-set disagreement rather than accepting caller
arithmetic. Taxonomy construction requires at least 100 reviewed documents and
a prospectively capped disagreement rate no greater than five percent. Evidence
span differences remain inspectable but do not count as semantic label
disagreement when both reviewers selected the same concept. A failed audit is
preserved as evidence and cannot construct the taxonomy.
`sai-validate-prerequisites audit-curriculum` performs this audit as a streaming
replay: it first revalidates the curriculum receipt, then rereads the exact
curriculum and annotation files in lockstep, derives phase membership from the
receipt rather than annotation claims, verifies both byte streams and canonical
population hashes, and publishes the report atomically without overwriting an
existing result.

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
