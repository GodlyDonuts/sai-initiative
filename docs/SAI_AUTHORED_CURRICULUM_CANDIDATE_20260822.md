# Sai Authored Curriculum Candidate — 2026-08-22

Status: candidate-only evidence. This authorizes no training, architecture
promotion, or 4B run.

## Why this exists

Sai treats data order as part of the learning algorithm. A document is not
admitted merely because it is technically accurate or advanced. Dependent
ideas must follow measured exposure to their primitives, and later phases must
continue rehearsing foundations. The intended rule is dependency order, not a
naive readability sort: “yellow + blue = green” belongs only after the learner
has encountered yellow, blue, color identity, and combination.

Authored instructional sequences are useful because their publisher order
carries stronger pedagogical information than shuffled web pages. That signal
still requires verification. A book's table of contents is a candidate
curriculum, not proof that a model learns best from it.

## Frozen sources

The create-only importer `sai.data.authored_curriculum` read two exact upstream
archives without extracting archive links or special members:

| Source | Exact revision | Archive SHA-256 | Ordered chapters | License evidence |
|---|---|---|---:|---|
| The Rust Programming Language | `917544888a55e4da7109bdba8c88c893c0da70f4` | `7eab4100b632c2d963ddc158b36e2d7e629f25ae4d3a781421537be3537ce6d8` | 111 | Apache-2.0 text SHA `0f276308…e1c`; MIT text SHA `0621878e…720` |
| CPython 3.13 documentation tutorial | `01104ce1beb3135c2e0c01ec835b994c1f55a1c0` | `88bb164ad7955549564a552e8eaedef66a1eb7d07c8f8a86969d3e0491238911` | 16 | PSF-2.0 text SHA `78b12c3a…3bb` |

Rust order is read from `src/SUMMARY.md`, SHA-256
`cf36f3d2c46320747f62e050649f2a5b9d32fcaa009605742a1908ff8d02ce61`.
Python order is read from `Doc/tutorial/index.rst`, SHA-256
`cd83ebcaa40518589e623eb2d8ef774cad0fa6cc7a86666dfb77e7a900540da7`.
Every selected chapter remains byte-exact, including headings, code fences,
directives, indentation, and examples. Cross-chapter chunking is forbidden.

The generated candidate contains 127 rows and 1,475,885 exact source bytes.
Its JSONL is 1,589,711 bytes with SHA-256
`c0c3158cfa8a133cb473459487ea4ed7514c09b8eeef2980cdf42769549724d1`.
The receipt self-hash is
`80de7bef204793694cff606972823cd37e77740eb70d0cc0176e7c33f6de908e`.

All 127 rows have also been frozen into an independently reviewable packet.
Rows are sorted by a salted review identity and hide source path, publisher
order, provisional stage, and declared prerequisites. The blind packet is
1,573,811 bytes with SHA-256
`2d662e9e394cd14ad0d7ce1c8058923c49837310663a7e0b57c2501ff8e34106`.
The separate key is 67,930 bytes with SHA-256
`b0e2c3d982b7fdb93fcd8bcbf9d4f224cf264a70f133fd4a9ed3ed97e766f9f0`.
Review-packet receipt `f052ff87…b906` requires independent annotator and
reviewer identities, exact evidence spans, at most five-percent concept-set
disagreement, and preservation of the measured disagreement before
adjudication. It contains no completed labels and authorizes no training.

`sai-adjudicate-authored-review` now makes that boundary executable. It reopens
the candidate, blind packet, hidden key, concept list, and semantic policy;
requires all 127 completed rows from two distinct identity artifacts; verifies
every taught-concept and defect span against exact chapter text; and measures
taught-concept, assumed-prerequisite, instructional-quality, admission, and
defect-category disagreement separately. A failed dimension cannot be averaged
away by agreement elsewhere. Even a pass authorizes only taxonomy/progression
analysis, not training. No completed human labels or pass receipt currently
exist.

`sai-compile-authored-review` removes a fragile annotation ambiguity before
adjudication. Reviewers submit exact verbatim evidence quotes instead of
hand-computed offsets. The compiler accepts only quotes that occur exactly once
in the immutable blind chapter, resolves them to Unicode-codepoint half-open
spans, and rejects missing, repeated, short, overlapping, unknown-concept, or
below-confidence evidence. Its receipt is explicitly candidate-only: compiling
model-assisted or human draft labels does not establish reviewer independence,
does not count as completed human review, and cannot authorize data admission
or training. The downstream adjudicator also rejects blank `admit` labels,
taught/assumed concept overlap, and sub-policy evidence spans; identical empty
reviews therefore cannot manufacture a high agreement score.

`sai-generate-authored-model-review` supplies a bounded candidate-annotation
lane when independent human labels are not yet complete. Each invocation uses
one of two exact, sealed, cross-family reviewers (Qwen3.5-9B or SmolLM3-3B) on
one H100. The runner sees the blind packet, concept list, and policy only; the
hidden review key is neither an input nor accessed. Greedy offline generation
has a fixed context, output, and repair-attempt budget. Every accepted row must
already pass the exact-quote compiler, and resumable raw rows are identity- and
source-hash-bound. These model drafts can accelerate triage and quantify
cross-family disagreement, but their receipts keep human review, audit
qualification, training, and 4B authorization false.

## Progression semantics

The Rust Book begins with installation, hello-world programs, a guessing-game
project, variables, types, functions, and control flow. Its later sequence
introduces ownership, data structures, error handling, generics, testing,
projects, concurrency, asynchronous programming, unsafe Rust, and a final web
server. The importer preserves this exact order and assigns four provisional
stage bands only for review.

The official Python tutorial states that it is for programmers new to Python,
not people new to programming. Therefore all 16 Python rows explicitly require
the prior concept `programming_foundations`; they are not placed in Sai's
grounding phase merely because Python is commonly described as easy. The large
Python library and language-reference collections are excluded from this
candidate rather than being mistaken for an introductory sequence.

Stage totals are 14 grounding candidates, 34 integration candidates, 34
reasoning candidates, and 45 specialization candidates. These labels are
prospective. Only evidence-span semantic annotation and independent review may
turn them into a qualified prerequisite schedule.

## Admission work still required

Before a row can become a Sai pretraining document:

1. complete title/source-level license and attribution review;
2. map exact concept exposures under the frozen semantic annotation policy,
   with cited source spans and independent blinded review;
3. verify that every dependent exposure has sufficient prior prerequisite
   coverage and that fundamentals recur in later phases;
4. globally exact- and near-deduplicate against every other Sai source;
5. decontaminate against development and terminal benchmark identities after
   every transformation;
6. test the authored-source addition against an equal-token control; and
7. test publisher/prerequisite order against an identical-document order
   control with matched tokenizer, model, initialization, optimizer, and
   compute.

The authored sequence is a small pedagogical spine, not a complete corpus. It
cannot replace broad English, mathematics, science, code, technical knowledge,
or carefully selected web coverage. Its purpose is to supply verified
definitions, examples, and composition order around which broader sources can
be paced.

## Promotion rule

Retain the source and its order only if matched small-scale evidence improves
or preserves phase-stratified held-out NLL/UTF-8 byte and broad source-disjoint
capability without a domain regression. A favorable training-loss curve alone
is insufficient. Failed source or ordering hypotheses remain recorded and are
removed rather than explained away by changing the architecture.
