# Sai 4B Data Mixture Plan

Status: prospective and data-first. This plan authorizes no 4B training. It
defines the evidence required before a final Sai corpus can exist.

`sai-validate-data-mixture <plan.json>` enforces the prospective
`sai-4b-data-mixture-plan-v1` schema. It requires immutable revisions and
source receipts, all five Sai domains, exact per-source and per-phase token
budgets, minimum-phase admission, mandatory rehearsal, optimizer-aligned phase
boundaries, factor-isolated controls, and both training authorization fields set
to false.

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
