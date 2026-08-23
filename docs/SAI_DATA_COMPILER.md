# Sai Data Compiler

Status: active compiler design with a live metadata pass. Raw source rows and
compiler judgments remain non-training candidates.

## Thesis

Sai does not build a training corpus by concatenating named datasets at fixed
percentages. A dataset is a reservoir of source evidence. The compiler converts
qualified source evidence into multiple traceable learning representations, and
the curriculum scheduler chooses among those representations according to
concept coverage, prerequisite state, model learning, and measured diversity.

```text
raw knowledge sources
    -> source identity and extraction
    -> epistemic-function analysis
    -> quality, novelty, difficulty, diversity, and grounding metadata
    -> preserve, translate, transform, recombine, or reject
    -> verify every derived representation
    -> global duplicate and contamination accounting
    -> concept/prerequisite graph
    -> curriculum scheduler
    -> packed training stream
```

No arrow erases lineage. Every derivative retains its source identities,
generator and prompt identity, transformations, verification evidence, and
duplicate family.

## English output, global knowledge

English-only means that the model learns and expresses the selected knowledge
in English. It does not mean Western-only acquisition.

High-value Chinese history, Persian poetry, Russian mathematics, French
philosophy, German engineering, Japanese literature, Indian philosophy and
history, Arabic science, Latin and Greek classics, and modern scholarship from
anywhere may enter the candidate pool. For non-English sources, the compiler
must decide whether to:

- use an existing authoritative English translation;
- translate while preserving argument and cultural context;
- translate while preserving literary form as far as possible;
- retain the original only as a source anchor and create English derivatives;
- reject translation because the loss or provenance risk is too high.

Translation never grants a source license, establishes factual truth, or makes a
source unique. Those remain separate evidence.

Language itself is never a rejection reason. Every non-English retained source
receives an explicit `translation_priority` from 1 to 4. High-value sources are
routed to English translation rather than excluded; zero is reserved for
sources already in English or sources rejected for an independent reason.

## Epistemic functions

### 1. Reality anchors

Primary and authoritative human material introduces information about reality:
books, papers, archives, records, reference works, journalism, code,
documentation, court opinions, standards, museums, cultural archives, and
structured databases.

The compiler may clean extraction defects, but it must not silently replace a
reality anchor with generated prose.

### 2. Knowledge distillations

A qualified anchor may produce several grounded representations:

- clean source text;
- concise reference entry;
- beginner, undergraduate, graduate, or expert explanation;
- conceptual summary and prerequisite map;
- FAQ, worked examples, and misconception/correction pairs;
- source-grounded textbook sections.

Derived text must cite exact source spans or structured records in its lineage.
The original and its derivatives share one exposure family so repeated facts
are accounted rather than disguised as unique knowledge.

### 3. Knowledge recombination

The compiler may construct examples that join established concepts from
different domains: biology and information theory, history and economics,
music and Fourier analysis, architecture and structural engineering,
philosophy and computation, literature and psychology, law and logic, or
systems and thermodynamics.

Recombination is admitted only after each prerequisite chain is represented and
the resulting claim or solution is independently grounded or verifiable.

### 4. Procedurally generated reasoning

Rule-defined generators can produce unlimited tasks with known truth for logic,
state tracking, graphs, causal systems, symbolic manipulation, planning,
constraints, algorithms, simulations, counterfactuals, space, and time.

This is the most reusable lesson from Shohin's procedural-data work. Sai will
retain the generators and ground-truth checkers while discarding the assumption
that those tasks validate one particular architecture. A procedural task teaches
composition or execution; it does not introduce unobserved real-world facts.

### 5. Human intellectual expression

Some texts carry information in their form. Literature, essays, speeches,
letters, debates, memoirs, diaries, journalism, criticism, humor, rhetoric,
dialogue, folklore, plays, poetry, philosophical argument, and distinctive
scientific exposition may be protected from rewriting.

The compiler must not flatten all human expression into homogeneous tutorial
language. It may attach explanations or translations while preserving a
licensed English original or authoritative translation as a separate
representation.

## Representation policy

Each source receives one preservation policy:

- `preserve_training_form`;
- `preserve_source_anchor_only`;
- `preserve_plus_derivatives`;
- `derivative_only`;
- `reject`.

This policy is distinct from source quality. A noisy rendering of a reliable
table may justify a clean derivative. A great novel may justify exact
preservation. A dense paper may justify both its original and explanations.
Executable code may justify the original, tests, reviews, and documentation.

## Model-directed selection

Fixed source ratios are replaced by measured marginal utility. Before final 4B
stream construction, proxy experiments may estimate what additional exposure
from each concept, style, difficulty, and reasoning region contributes.

Candidate metadata may include semantic clusters and model-induced gradient
clusters. [Prismatic Synthesis](https://arxiv.org/abs/2505.20161) and
[SPOKES](https://arxiv.org/abs/2606.15216) motivate gradient-space diversity as
an experimental selection signal. It does not supersede source reliability,
quality, prerequisite closure, contamination, or representation lineage.

The controller asks:

> What does another unit of unique exposure from this region add to the model?

It does not ask:

> What percentage did another laboratory assign to this upstream dataset?

## Synthetic generation hierarchy

Synthetic rows are ranked by what they add:

1. paraphrases of one source -- generally low novelty;
2. multi-source, retrieval-grounded explanations or coherent books;
3. cross-domain problems with independent solution and deterministic checks;
4. examples targeted at underrepresented model-gradient regions and verified
   against ground truth.

The compiler prefers higher levels when source grounding and verification are
available. The 2026 [book-organization
study](https://arxiv.org/abs/2607.28109) motivates coherent source-grounded
books rather than isolated rewrites. ICLR 2026 [rule-generated multi-hop
work](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e95016eceae77e945b84554b0f1bb49-Abstract-Conference.html)
supports procedural fictional worlds as a way to teach composition without
pretending that they add real facts.

## Current executable boundary

`sai.data.data_compiler_labeling` defines the comprehensive compiler judgment.
It records epistemic functions, fine-grained domains, source language,
translation and preservation decisions, representations, provenance form,
grounding, prerequisite concepts, cross-domain bridges, thirteen independent
scores, twelve risks, exact evidence spans, and a grounded transformation brief.

`sai.data.nous_compiler_worker` runs this contract through the Hermes loopback
proxy with create-only per-source receipts. One row failure is isolated; other
rows continue, and reruns skip already completed identities. The worker persists
no portal credential, enables no tools, and marks every output
`training_ready=false`.

Model responses sometimes preserve every source token while normalizing Unicode
compatibility, case, or whitespace. The compiler now recovers such evidence only
when the normalized token sequence maps to exactly one literal source span. It
then stores the exact source bytes in the normalized judgment while retaining
the raw model-JSON hash and a text-free repair record with both quote hashes and
source byte offsets. Exact quotes are unchanged; missing, punctuation-altered,
or ambiguously repeated spans still fail closed. The frozen rubric hash remains
`3becf24768708f710439c053b1d0db1513ba03680c83f39503f007aa6a2a61c6`.

This is metadata compilation only. Translation, synthesis, procedural
generation, verification, adaptive selection, and packing are separate compiler
stages that must preserve this lineage.

### Source-agnostic mechanical quality gate

Semantic judgments never override deterministic high-confidence junk evidence.
`sai.data.source_quality_gate` replays provenance-bound candidate populations
and emits text-free, per-identity decisions for contextless MCQ keys, scored
answer sheets lacking their questions, embedded control and replacement
character corruption, repeated-character gibberish, contextless link/markup or
structured fragments, and duplicated boilerplate. Nonpass routes are excluded
from direct admission. The gate deliberately distinguishes hard rejection from
recoverable context and cleanup review, and its mechanical pass is not semantic
admission.

The first 12-population publication covers 8,323 distinct identities: 8,313
mechanical passes, nine cleanup holds, and one hard rejection. Its policy hash is
`f85ae862121974b48210964b9a81abd55ae4a6a35cf7e7758840ba854f9faf0f`
and canonical publication receipt is
`c25127e13c579bb066b887d264da1905bd78f2f3d24c183bba547ea019a2bf66`.
All rows remain `training_ready=false`; the decision streams are evidence about
admission work, not a compiled training corpus.

### Grounded cross-domain bridge verification

Cross-domain metadata becomes useful training material only through paired
source evidence. The development compiler binds each generated bridge to two
source-disjoint anchors and requires four representations: conceptual
explanation, worked transfer problem, counterexample, and analogy limits. An
independent Hermès request then checks every claim against its assigned exact
anchor, verifies the shared structure and transfer solution, and routes the
candidate to retain, revise, or reject. The same-family verifier never confers
final verification: its source-text-free outputs remain
`bridge_verified=false` and `training_ready=false` until an independent model
family, benchmark decontamination, global deduplication, and measured transfer
ablations close.

`sai.data.reservoir_audit_decision` converts a completed, hash-valid aggregate
into a create-only source-work ledger. It never extrapolates a coverage screen
into an acceptance rate and never grants bulk admission. Instead it routes each
source toward rights resolution, quarantine confirmation, cleanup, grounding,
translation, transformation, or representation verification. Its first live
ledger paused FinePDF bulk expansion and rights-blocked OpenWebMath while
prioritizing targeted FineWeb-Edu verification.

The same ledger machinery has now replayed a completed 124-row Common Pile
breadth screen spanning 31 sources. Aggregate receipt
`d79749882b8e306e87997a2e0f13bd558e0bef268356b696e6d140eab656bd22`
and work-ledger receipt
`e036d9d96bfc260fc3d64f6851db00213231545ac1aca2fc1bbe00ed4427ae58`
remain explicitly descriptive. The screen promotes ArXiv Abstracts, Public
Domain Review, Python Enhancement Proposals, and StackExchange to larger
source-specific confirmation, while pausing the observed Wikiteam, arXiv Paper,
peS2o, and Wikimedia representations. USGPO is not promoted despite a 4/4
compiler route because the independent corrected benchmark-word gate found
3/4 overlaps. This is the intended multi-gate behavior: model preference cannot
override contamination evidence.

`sai.data.reservoir_audit_confirmation_plan` makes the next promotion
deterministic. It binds a completed aggregate and corrected contamination
screen, then requires zero observed benchmark overlap, zero quarantine/rights
routes, and at least 500,000 ppm representation-verification signal. The
executable live plan selects seven Common Pile sources for 32 source-disjoint
rows each:
ArXiv Abstracts, GitHub Archive, LibreTexts, Pressbooks, Public Domain Review,
Python Enhancement Proposals, and StackExchange. Plan receipt
`350e96f2c1bbffa473eb7801fcd43548b03141754622c1ba0cd55a1e7bb9e625`
requires exact row and content disjointness from discovery, prefers a different
pinned parent when available, and otherwise reuses the only pinned parent with
exact discovery-line and content-hash exclusions. It still sets
`bulk_training_admission=false` and `training_ready=false`. The earlier receipt
`a48d9860193460e037c095f5483eb18b4b5199ec6b7be05eba8c6ebcfe562676`
is superseded because requiring a different parent for every selected source
was infeasible for four single-parent source collections.

The resulting confirmation population sealed 224/224 rows under receipt
`40e72050e1c5a44d0e7618413d6e731de23232be9982f8f4be5d13eada44b6a5`.
It contains 32 rows per selected lane, verifies 2,637,343,362 compressed parent
bytes, and holds only one parent file at a time. Exact and normalized-token
duplicate replay across discovery plus confirmation found zero flagged pairs
among all 60,378 possible pairs. Corrected benchmark screening found one
contaminated GitHub Archive row and 223 clean rows; the other six lanes were
32/32 clean. The row is evidence against blanket GitHub Archive promotion, not
permission to weaken the boundary. All source rows remain
`training_ready=false` pending compiler, rights, full-source deduplication, and
transformation gates.

The next conversion boundary is executable in
`sai.data.confirmation_promotion` and
`sai.data.common_pile_streaming_pilot`. Promotion requires a 32-row minimum,
at least 500,000 ppm representation-verification routing, zero quarantine,
zero rights holds, zero benchmark-contaminated rows, zero exact or
normalized-token duplicate pairs, and exact identity/content disjointness from
discovery. It authorizes only a bounded streaming pilot.

The pilot then selects one exact hash-pinned parent, prefers a parent unused by
both audits, excludes every audit line and content hash, performs deterministic
bottom-k selection in a text-free first pass, and replays only those exact rows
in source order. The downloaded parent is held one at a time and removed after
full compressed-byte verification. The pilot applies the active binary
benchmark boundary and normalized exact deduplication before an exhaustive
bounded-pilot near-duplicate join. That join covers every surviving unordered
pair with exact SHA-256 five-word-shingle identities, frozen
Jaccard/containment thresholds, and deterministic canonical survivors. Its
receipt explicitly leaves global cross-source near-duplicate filtering open.
The pilot counts short and oversized documents instead of truncating them, and
it remains `training_ready=false` until rights, global deduplication, and
representation verification are independently complete.

`sai.data.attribution_manifest` replays every retained pilot identity back to
its exact raw repository revision, source file, and row index. It emits no
source text; it records the original and canonical license classification plus
attribution and share-alike obligations. Exact retained-document coverage is a
pipeline-lineage result, not external provenance verification or legal
clearance, and the receipt keeps those distinctions explicit.

After two or more bounded source pilots complete,
`sai.data.cross_source_pilot_duplicates` constructs a deterministic
source-stratified sample with a global bottom-k fill. It replays every selected
unordered pair under the identical five-word-shingle policy and separately
counts duplicate components spanning source IDs. This measures whether
source-local cleanup is hiding cross-source redundancy while explicitly
leaving full-reservoir deduplication and training admission false.

`sai.data.license_policy` is a separate exact-declaration boundary. The pinned
cards for the seven Common Pile confirmation sources do not provide one
top-level license; their source READMEs and rows carry source-specific rights
claims. The policy maps exact CC0, Public Domain, CC BY, CC BY-SA, Apache-2.0,
MIT, BSD-2-Clause, and WTFPL aliases to canonical identifiers and explicit
attribution/share-alike obligations. It deliberately does not guess a version
for the two observed “GNU Free Documentation License” rows, and it sends every
unknown or ambiguous value to `rights_hold`. Even a recognized declaration
does not establish provenance or legal clearance. Streaming selection excludes
rights-held rows while preserving counts in its receipt.

The live audit binds seven exact README hashes and all 224 confirmation
declarations under receipt
`357414811d687921225830732feae6f45508707f126c01cf7b01624eaed0df40`.
It recognizes 222 declarations and sends the two unversioned GFDL LibreTexts
rows to rights hold. Hugging Face commit
`e6b1210f26a7fb7e06e45c193131aa71d2c574df` replayed the source-safe artifact
byte-for-byte. Promotion schema v2 now requires this receipt and zero
source-level declaration holds alongside the compiler, contamination,
duplicate, and identity/content-disjointness gates.

Future conversion uses exact-declaration policy v2, which binds the entire
recognized-alias table by hash and adds `ODC-By-1.0` plus `CC-BY-2.0` for the
observed reservoirs. Both retain attribution obligations. The new schema does
not mutate the earlier Common Pile rights receipt or turn a dataset-level
license into a content-level legal conclusion.

`sai.data.reservoir_rights_inventory` has now replayed both candidate
reservoirs. Corrected schema v2 binds 46 source lanes and 45 exact repository
revisions covering all 42,600 files and 23,680,076,298,761 physical candidate
bytes. Five lanes have exact manifest-declaration obligations, 31 require
per-row license evidence, and ten require source-terms resolution. A wrapper
card no longer erases manifest qualifiers such as upstream or generator terms.
The StackV2 HTML pinned tree contains no README; absence is a fail-closed
result, not permission to borrow another revision's terms. Corrected receipt:
`8e72391081af17323aa1e1b8d0480ddbe70dcb232006e6cf37ed7228d34d3d80`.
The earlier 11/31/4 routing is retained only as superseded audit history.

`sai.data.benchmark_boundary_index` builds a non-reversible official benchmark
boundary for 18,235 rows across nine public benchmark views. The r1 13-token
word index contains 27,979,728 unique SHA-256 keys and its r1 8-token code index
contains 1,907,051. Receipt
`073bb9f8a9ab9954ed3913b2414ff718e8f86a5020b2eb1feb18069cd75510f1`
binds the official source revisions, every source byte hash, the model-visible
LiveCodeBench projection, index hashes, and the fact that raw benchmark text was
not persisted. Byte replay subsequently exposed a semantic policy flaw: the r1
code index admitted punctuation-only windows. The receipt is retained as audit
history but is not an active training-data gate. RULER remains a tokenizer-bound
generator gap rather than being silently substituted or mislabeled.

The v2 policy keeps 13-token exact word matching and restricts exact 8-token
code matching to windows with at least four alphanumeric-bearing tokens, three
distinct alphanumeric-bearing tokens, and 16 total characters. The create-only
v2 code index contains 475,804 unique keys and has SHA-256
`d438ea1176ed8357b7139475d469ce42dbe4c147f62cbab301b48e26e68dea39`.
Its 27,979,728-key word index exactly reproduces the r1 SHA-256. Receipt
`9fee65cb9f99813407ea4d5e4c35b4bc0bb7659c1720342f0f50bd1a8c237667`
binds the corrected policy and all source revisions.

The five replacement screens reduced the exact population result from
286/1,879 to 69/1,879 and the Nemotron specialized-reasoning result from 77/96
to 25/96. Their population counts are 6/128, 26/1,024, 28/512, 7/124, and
2/91. The old source-quarantine conclusion is retracted. The v2 boundary and
screens passed local and post-upload replay in Hugging Face commit
`43ae57ee4981c78ae23c111436b1fc9b6aa27023`.
`sai.data.benchmark_contamination_screen` persists aggregate counts and an
ordered decision digest while persisting neither individual decisions nor
source text.

The 91-row modern-source expansion has also closed its compiler pass under
aggregate receipt
`afd82b43ac66f3a485d167b97f79fccc75bc67026c94e182d846a6923f9dea23`
and source-work receipt
`487c50a9fd4ba39ae22e73e4478673762e60fbedef254be387766fa41d740978`.
Although 60 rows received model `retain` verdicts, only 14 routed directly to
representation verification. Nemotron Legal produced the strongest observed
signal at 8/21 representation-verification rows; its remaining sample still
contains cleanup, grounding, quarantine, and transformation work. Nemotron
Specialized v1.2 routed 22/30 rows to factual-grounding review and zero directly
to representation verification. PleIAs Common Corpus routed 16/40 to
quarantine, 11/40 to cleanup, 6/40 to translation, and 6/40 to representation
verification; the independent benchmark screen also located its only two
contaminated rows. No source receives bulk admission from this coverage screen.

`sai.data.text_payload_probe` closes a different accounting gap before bulk
conversion. Its prospective plan selects exact members by SHA-256 rank before
opening their content or consulting file size. Execution holds one member at a
time, verifies the full pinned size and SHA-256, measures only the declared text
column, separates all UTF-8 bytes from the mechanical 200 B–128 KiB useful-size
window, and removes temporary source bytes. A selected member above the frozen
storage cap is blocked and cannot be replaced with a smaller shard. Results are
bounded member measurements, never source-wide yield estimates or admissions.
The separately planned FinePDF probe showed why the useful-size window is an
accounting boundary rather than a rejection rule: 12,013 of 414,000 exact rows
exceeded 128 KiB. Long papers and books must enter a structure-aware
segmentation queue that preserves sections, provenance, and work identity;
blind truncation or blanket rejection would destroy valuable long-form signal.

`sai.data.structural_segmentation` implements that queue's lossless mechanical
boundary. It operates on raw pretraining rows, leaves in-budget rows unchanged,
and prefers paragraph, line, sentence, clause, and word boundaries before an
exact Unicode-character fallback. Every child binds the upstream dataset,
revision, file, row, parent-text digest, segment index/count, UTF-8 byte range,
and child-text digest. The compiler verifies ordered byte reconstruction and
writes a text-free lineage manifest. `sai.data.decontamination` derives a unique
row ID from this child lineage, while `sai.data.attribution_manifest` replays the
same lineage and license obligations. Segmentation grants no quality decision,
rights clearance, benchmark clearance, deduplication status, or training
admission.

`sai.data.external_exact_deduplication` provides the global normalized-exact
layer after benchmark-disjoint representations exist. It streams immutable
JSONL populations into fixed-width, text-free external-sort runs, reduces those
runs with bounded fan-in, and selects the minimum document identity within each
NFKC/casefold/whitespace-normalized SHA-256 group. Each apparent hash collision
is replayed against the full normalized source text before a row is dropped.
The survivor stream is deterministic across input ordering; a text-free drop
manifest preserves custody; temporary indexes are removed. This gate does not
claim semantic or near-duplicate completion and does not itself admit data to
training.

`sai.data.frequency_length_subdocument_deduplication` implements the distinct
subdocument layer. It losslessly divides natural-language documents at
paragraph, line, and sentence boundaries, forward-merges units below the
configured character floor, and preserves Markdown fenced code as indivisible
exact text. Full code-domain documents are conservatively indivisible until a
language-aware structural parser is qualified. Natural-language matching uses
NFKC, casefolding, whitespace collapse, and numeric placeholders; code matching
uses identity normalization.

Fixed-width external-sort records make global frequency independent of physical
sharding. For each group the compiler evaluates
`g_N(C)=C(1-1/N)^(C-1)`, applies
`T(C,L)=ceil(1+(g_N(C)-1) max(0,1-L/L0))`, and clamps the result to `[1,C]`.
Occurrence ordering is by immutable document identity. If the budget boundary
falls within multiple occurrences from one document, the whole document group
is retained. Candidate deletions are reconstructed in original document order
and removed only when a maximal contiguous run reaches `tau_del`; shorter runs
are restored to prevent fragmentation. Changed documents receive new
content-bound identities, while a text-free manifest maps each parent to its
output and records every group frequency, budget, span, and final outcome.

The defaults mirror the paper's reported geometry: `tau_seg=32` characters,
`tau_del=100` characters, `L0=512`, and effective `N=100/3`. They are
experimental priors, not frozen Sai winners. The output cannot advance until
unchanged, keep-one, and adaptive controls are compared with equal source bytes,
tokens, compute, and source-disjoint evaluation. This layer also does not claim
semantic near-duplicate completion or training admission.
The CLI exposes `--retention-policy keep_one_control` and
`--retention-policy adaptive_frequency_length`; each arm writes a distinct
policy hash and explicit completion flag, while the immutable input supplies
the unchanged arm.

## Institutional Books lane

The Harvard Library Institutional Books release is now a pinned, separate
reality-anchor lane. Its exact source revisions, Early-Access terms, metadata
join, book record, translation policy, and executable boundary are documented in
[`SAI_INSTITUTIONAL_BOOKS_COMPILER.md`](SAI_INSTITUTIONAL_BOOKS_COMPILER.md).

Book judgments do not use the generic scalar `difficulty`. They record
linguistic, conceptual, and reasoning complexity independently and propose
evidence-backed prerequisite edges. Archive-supplied identifiers, OCR scores,
duplicate barcodes, and rights evidence remain outside model control.

Non-English technical works route to faithful English representations. For
literature, poetry, and drama, the compiler first seeks a reputable human English
translation; otherwise it requires separately labeled literal and literary
translations and preserves the original-language anchor.

The initial spiral policy retains basic material in every phase at
65%, 40%, 20%, 10%, and 10%, while expert exposure rises from 2% to 35%.
Band admission is prerequisite-graph-based and never replaces the three
complexity axes with a single score.
