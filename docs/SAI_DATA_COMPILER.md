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
routes, and at least 500,000 ppm representation-verification signal. The first
live plan selects seven Common Pile sources for 32 source-disjoint rows each:
ArXiv Abstracts, GitHub Archive, LibreTexts, Pressbooks, Public Domain Review,
Python Enhancement Proposals, and StackExchange. Plan receipt
`a48d9860193460e037c095f5483eb18b4b5199ec6b7be05eba8c6ebcfe562676`
still sets `bulk_training_admission=false` and `training_ready=false`.

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
