# Sai 4B Data Mixture Plan

Status: prospective and data-first. This plan authorizes no 4B training. It
defines the evidence required before a final Sai corpus can exist.

`sai-validate-data-mixture <plan.json>` enforces the structural prospective
`sai-4b-data-mixture-plan-v2` schema. It checks immutable-looking revisions and
hash identities, all five Sai domains, exact per-source and per-phase token
budgets, minimum-phase admission, mandatory rehearsal, optimizer-aligned phase
boundaries, factor-isolated controls, and both training authorization fields set
to false. V2 alone is not final evidence: it cannot prove that a syntactically
valid hash resolves to the claimed artifact.

`sai-validate-data-mixture-evidence <plan-v3.json> --evidence-root <root>` is
the required final prospective boundary. Its relocation-safe
`sai-4b-data-mixture-plan-v3` descriptors reopen every source manifest,
selection policy, title/source license decision, quality audit,
decontamination receipt, and pedagogical-progression receipt. Every file must
be a single-link regular file reached through non-symlink directories; its
bytes must match the declared SHA-256. Receipt roles additionally require the
declared schema/status and a replayed canonical self-hash. Missing evidence,
bare invented hashes, unsafe paths, re-signed content drift, and status drift
all fail. A generic successful-looking status is insufficient: license,
quality, decontamination, and pedagogical receipts must respectively assert the
exact boolean decisions `license_approved`, `quality_qualified`,
`decontamination_qualified`, and `progression_qualified`. Shared global evidence
is permitted only when every source declares
the identical descriptor. No v3 mixture currently passes because the source
selection work is intentionally incomplete.

## Principle

Sai is not allowed to compensate for bad data with a novel architecture. The
final corpus must establish language and symbolic primitives before it demands
their composition, preserve broad rehearsal while difficulty rises, and prove
that every selected source and ordering decision helps under matched controls.
The current 500M-token FineWeb-Edu curriculum is the first controlled data
experiment, not the final corpus.

## Candidate source classes

Every source below remains a candidate until an immutable revision, exact file
manifest, applicable license terms, provenance fields, removal policy, and
benchmark-disjoint receipt pass Sai validation.

| Class | Candidate | Intended contribution | Primary risk |
|---|---|---|---|
| Educational web | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Broad English explanations and examples | Classifier score is not pedagogical truth; web duplication and synthetic contamination |
| Diverse open corpus components | [Dolma](https://docs.allenai.org/training_data/dolma) | Books, encyclopedic material, academic text, code, and web diversity | Component overlap with other sources; source-specific licensing and quality differ |
| Mathematics | [OpenWebMath](https://arxiv.org/abs/2310.06786) | Mathematical prose, notation, derivations, and worked explanations | Benchmark leakage, answer-only pages, rendering loss, and narrow-domain overexposure |
| Code | [The Stack v2](https://github.com/bigcode-project/the-stack-v2) | Permissively licensed code with persistent provenance | License obligations, opt-outs, generated/vendor files, secrets, tests copied into benchmarks |
| Foundational references | exact sources to be frozen | Public-domain books, encyclopedic articles, and introductory educational sequences | Edition drift, OCR damage, duplicated editions, weak document identity |
| Science and technical literature | exact sources to be frozen | Definitions, mechanisms, experiments, engineering, and evidence | Copyright boundaries, PDF extraction damage, citation boilerplate, specialization too early |
| Authored open textbooks | [Open Textbook Library](https://open.umn.edu/opentextbooks) title-level candidates | Human-authored chapter progression, definitions, worked examples, exercises, and prerequisite order | The catalog is a referatory with mixed licenses; every title, edition, file, and original publisher license must be verified separately |
| Programming pedagogy | exact tagged [Python documentation](https://docs.python.org/3/license.html) and exact-commit [Rust Book](https://github.com/rust-lang/book) candidates | Ordered language concepts, executable examples, API semantics, and progressively composed projects | Documentation assumes different prior knowledge; version drift and code/example duplication require explicit handling |
| Public-domain books | selected [Project Gutenberg](https://www.gutenberg.org/policy/license) works acquired through approved bulk channels | Long-form English, historical technical exposition, and selected foundational texts | U.S.-specific copyright determinations, trademark/license wrappers, old errors and pedagogy, OCR/edition variance, and unsuitable material |

## Candidate revision inventory — 2026-08-22

The following repository heads were resolved through the Hugging Face dataset
API on 2026-08-22. These hashes make the investigation reproducible; they are
**not** source manifests, license approvals, selection receipts, or permission
to train. A mutable repository head is never substituted after this inventory
without a new factor declaration.

| Candidate | Resolved revision | Repository license/gate metadata | Sai disposition before admission |
|---|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | ODC-By, public | Current raw educational-web control only; retain exact quality, duplicate, and curriculum audit requirements. |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By, public | Raw mathematics candidate only. A verified 104,680-row `4plus` shard contains incoherent score-5 prose, answer-farm/SEO/commercial-homework material, and only 34.76% `found_math=true`; apply a new Sai filter and full audit before any comparison. |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | `fde8ef8de2300f5e778f56261843dab89f230815` | no repository-level license recorded by the API | Research control only until exact source and redistribution terms are resolved; do not infer permission from public downloadability. |
| [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) | `eeec5caac5cc3758a18f1d3ba4416837a9ba814c` | no single repository license; inherits The Stack v2 provenance and opt-out obligations | Strong code-quality candidate, but admit only file identities with acceptable licenses, preserved attribution/removal metadata, secret scanning, and current opt-out replay. |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | use Stack-Edu's exact parent lineage rather than a moving head | per-file licenses or no detected license; Software Heritage content indirection | Never admit unlicensed files. Prefer the educational subset and a Sai allowlist of accepted SPDX terms over the full pool. |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | `7f48140530a023e9ea4c5cfb141160922727d4d3` | ODC-By, public | Candidate component inventory, not an opaque blend. Select books, encyclopedic, academic, and other components independently and globally deduplicate them against every other Sai source. |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` | Apache-2.0, public; generated by Mixtral-8x7B-Instruct-v0.1 | Synthetic pedagogical-data ablation only. Preserve seed, prompt, audience, and generator provenance; measure factuality and imitation artifacts; never use it as the sole grounding source. |
| [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | `397a2502f2028c659ba411a6c4935b464a7f03aa` | gated, NVIDIA Data Agreement for Model Training | Modern math candidate, especially the declared `4plus` subset. Admission requires explicit acceptance and an exact legal/redistribution review before bytes are downloaded. |
| [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) | `01393d3bd890ddc5c15da8a2c9afb57391277659` | manually gated, NVIDIA agreement | Modern curated-code candidate. Require accepted file licenses, provenance, secret scans, and separation of organic code from generated question-answer material. |
| [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | `2669787c66d18601c0e91167fd520be6c1245865` | manually gated; repository updated 2026-07-07 | Evaluation candidate only after separating organic web from synthetic rewriting. The dataset card warns that synthetic Qwen/DeepSeek-derived subsets may impose model redistribution requirements; those subsets are excluded until resolved. |

Initial source-factor priority is therefore: FineWeb-Edu as the existing web
control; a newly filtered and audited FineMath-derived pool as the first public
math candidate; a license-allowlisted, provenance-preserving Stack-Edu slice as
the first code candidate; and selected Dolma components for non-web diversity.
The bounded FineMath evidence is recorded in
`docs/SAI_FINEMATH_SHARD_AUDIT_20260822.md`; upstream `4plus` scores cannot admit
rows. Cosmopedia and NVIDIA corpora remain separate declared ablations.
OpenWebMath remains blocked on license evidence. No source ratio is frozen
before small matched source-addition screens establish that the source helps
rather than merely looking sophisticated.

Authored sequence is now a first-class source signal. The Open Textbook Library
currently exposes a public JSON API and identifies hundreds of English CC BY or
CC0 records, but it does not host or license every underlying textbook itself.
Sai may use the catalog only to discover candidates. Admission requires the
exact book file, chapter order, original publisher page, title-specific license,
attribution terms, extraction audit, and content review. The initial lane is
limited to CC BY or CC0 works; NC, ND, ambiguous, and title-level license
conflicts remain out of training until separately resolved.

Programming documentation is also kept title-specific. Python's documentation
is under the PSF License v2, with examples and recipes from Python 3.8.6 onward
also available under 0BSD. The Rust Book repository is dual MIT/Apache-2.0 and
explicitly describes later chapters as building on earlier ones while project
chapters rehearse prior concepts. These properties make exact tagged versions
strong candidates for a code prerequisite lane, but neither is treated as a
complete beginner curriculum or blended with scraped code before a matched
source-addition test.

The first exact authored candidate is now frozen. It binds Rust Book revision
`917544888a55e4da7109bdba8c88c893c0da70f4` and CPython revision
`01104ce1beb3135c2e0c01ec835b994c1f55a1c0`, preserves 127 publisher-ordered
chapters and 1,475,885 source bytes without cross-chapter chunking, and records
receipt `80de7bef…08e`. The Python tutorial's own stated audience is used as a
hard curriculum constraint: every Python row requires prior programming
foundations. This candidate remains outside training pending semantic review,
license review, global deduplication, benchmark decontamination, and matched
source-addition and order controls. Full evidence is recorded in
`SAI_AUTHORED_CURRICULUM_CANDIDATE_20260822.md`.

OpenStax is not admitted as one source class merely because it is called open.
Current official licensing information describes its textbooks as
CC BY-NC-SA, while some individual catalog records describe earlier titles as
CC BY. Sai therefore resolves the license of every exact edition at its original
publisher. Current OpenStax book pages also explicitly prohibit using the text
to train or ingest into large language models without OpenStax permission.
OpenStax content is therefore excluded from Sai training unless exact written
permission is obtained; its published scope and sequence may be studied only as
external taxonomy-design evidence. Project Gutenberg is also title-specific:
acquire only through its approved bulk mechanisms, verify each work's U.S.
status and embedded terms, preserve provenance, and separate the underlying
unrestricted text from Project Gutenberg trademark/license material when the
exact terms permit it.

Sai will not ingest an opaque pre-blended corpus and then claim source-level
control. When a candidate distribution contains multiple source components,
the admitted rows must preserve their original component and document identity.

## Global admission boundary

The final mixture is built in this order:

1. Freeze every upstream revision and byte manifest before inspection-driven
   threshold changes.
2. Normalize text without destroying code indentation, mathematical notation,
   tables, units, identifiers, or document boundaries.
3. Enforce source-specific license and removal policies. Preserve attribution
   metadata wherever the source license requires it.
4. Reject secrets, private information, extraction failures, templated spam,
   generated-content loops, severe repetition, answer-key farms, and documents
   without stable provenance.
5. Decontaminate against all development and terminal benchmarks after every
   transformation, including generated annotations or reasoning traces.
6. Perform exact and near-duplicate clustering globally across all sources
   before train/development assignment. Split complete duplicate families, not
   individual rows.
7. Assign quality, domain, difficulty, and eventual concept-prerequisite
   evidence. Preserve raw signals so selection thresholds can be replayed.
8. Freeze the exact source mixture and curriculum. Tokenize only after document
   selection, retaining document-boundary target masks and admitted UTF-8 bytes.
9. Reopen every source, transformation, membership decision, packed token, and
   receipt before any optimizer consumes the stream.

## Pedagogical progression

The final schedule has four functions rather than four labels:

1. **Grounding:** ordinary English, syntax, definitions, quantities, basic
   code forms, elementary mathematical objects, observation, and direct factual
   relations. No dependent concept is credited unless its prerequisites have
   prior measured coverage.
2. **Integration:** explanations that combine established primitives, compare
   alternatives, use examples, and connect representations across prose,
   equations, code, diagrams converted to text, and tables.
3. **Reasoning:** causal, counterfactual, proof-like, algorithmic, quantitative,
   and evidence-based composition with verified intermediate structure where
   available.
4. **Specialization:** advanced code, mathematics, science, and technical
   material whose prerequisite concepts already have coverage.

Grounding material remains rehearsed in every later phase. Increasing
difficulty cannot be achieved by permanently removing basic language or broad
knowledge. Phase boundaries align with optimizer updates so no accumulation
window mixes two scheduling policies.

The current surface score is only one ordering hypothesis. A future semantic
schedule must bind a frozen concept taxonomy and prerequisite DAG, evidence
spans for each annotation, first-exposure statistics, prior prerequisite
coverage, rehearsal counts, and unresolved violations. A separate model-centric
schedule may use a frozen small checkpoint's learnability, influence, loss, or
gradient-noise signal, but never terminal benchmark answers or treatment-state
feedback.

## Factor-isolated data ladder

All comparisons use the same model geometry, initialization, optimizer,
tokenizer, admitted UTF-8/token budget, masks, evaluation rows, and modeled
compute unless that item is the declared changed factor.

1. **Quality selection:** qualified rows versus a prospectively frozen less
   selective population from the same source pool.
2. **Source diversity:** selected multi-source mixture versus selected web-only
   data at equal admitted tokens and equal compute.
3. **Surface order:** prerequisite-to-specialization ordering versus an exact
   permutation of the same packed sequence records.
4. **Semantic prerequisites:** concept-DAG ordering versus the surviving surface
   schedule, using the same sequence multiset.
5. **Model-centric pacing:** frozen learnability/influence ordering versus the
   surviving semantic or surface schedule.
6. **Mixture ratios:** only after source classes independently pass, compare
   prospectively frozen ratios; never tune them on terminal benchmark answers.

The ladder begins with one matched approximately 100M-parameter model because
data must be selected before architecture. A positive one-seed result is
provisional. Promotion requires repeated seeds, held-out likelihood normalized
per token and admitted UTF-8 byte, broad source-disjoint capability, and declared
per-domain retention. Architecture comparisons then inherit the selected data
bytes unchanged through 100M, 300M, and 1B.

## Required evidence

The final data authorization must bind:

- ordered source revisions, manifests, licenses, removal-policy timestamps, and
  exact admitted rows;
- source, domain, quality, difficulty, concept, phase, and duplicate-family
  populations in both documents and admitted UTF-8/tokens;
- every rejection reason and a deterministic audit sample containing accepted
  and rejected rows;
- global exact/near-duplicate and benchmark-decontamination receipts;
- train/development family-disjointness and ordered identity hashes;
- tokenizer tree, lossless round trip, fertility by source/domain, packed token
  shards, target masks, and phase/update boundaries;
- exact comparison checkpoints, environments, seeds, compute, row-level
  evaluations, uncertainty intervals, and regression vetoes.

No aggregate quality score, upstream dataset reputation, or training-loss curve
can substitute for this evidence. A source or schedule that fails remains a
documented result and is removed from the final mixture rather than explained
away by architecture changes.

## Promotion boundary

A data factor is retained only if it improves or preserves held-out token NLL
and UTF-8-byte-normalized NLL, produces a credible nonnegative aggregate on
broad source-disjoint development tasks, and stays within every predeclared
domain regression floor. The final public boards are not used to iteratively
tune data. Passing the data ladder authorizes architecture selection; it does
not authorize the 4B run.
