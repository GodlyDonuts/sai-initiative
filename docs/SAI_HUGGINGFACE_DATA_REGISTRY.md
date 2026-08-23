# Sai Hugging Face Data Registry

Status: active public registry at
[`Godlydonuts/Sai`](https://huggingface.co/datasets/Godlydonuts/Sai).

The registry is the durable transfer surface for Sai data. It is not a flat
dataset and it does not imply that every published artifact is training-ready.

## State machine

Every record advances through explicit, non-interchangeable states:

1. `source_reference`: upstream repository, revision, license/terms, and exact
   hashes;
2. `candidate`: bounded source material awaiting compiler judgment;
3. `compiler_judgment`: source-bound quality, representation, translation, and
   prerequisite decisions;
4. `verified`: derived material whose grounding, duplication, contamination,
   and transformation checks passed;
5. `curriculum`: records assigned to epistemic function and prerequisite-aware
   spiral phases;
6. `training`: final immutable shards with tokenizer, ordering, exposure, and
   replay receipts.

Only state 6 may be consumed by a training job. A model judgment is not a
verification result, and an upstream quality label is not a Sai admission.

## Repository layout

- `registry/` stores the complete local artifact hash index, upstream source
  registry, and sanitized build manifests.
- `compiler_judgments/` stores strict Hermes outputs that remain
  `training_ready=false` until verification.
- `curriculum/` stores Sai-authored concept graphs and scheduling inputs.
- `verified/` is reserved for verified representations.
- `training/` is reserved for final packed populations.

The first registry commit is
`89152fff3e47d85e35e75cad6b419b4f304a4e85`. It contains 22 paths: the dataset
card, a 462-file local artifact index, five source-registry entries, a sanitized
Institutional Books pilot manifest, 14 FineWeb-Edu compiler judgments, and four
foundational-syllabus artifacts. The local artifact-index SHA-256 is
`a86a6e294ed8dd42f9a0743ce508febce65fa5f8e545c3deafa08bb40481d9ec`.

## Publication boundary

Each indexed artifact has an explicit publication disposition. Default is
staged review, not upload. Source-specific terms always override the dataset
card.

- Gated Institutional Books text and excerpt-bearing candidates are
  `reference_only_gated_no_redistribution`; only sanitized hashes and counts are
  public.
- Sai-generated curriculum graphs are publishable.
- FineWeb compiler judgments are published with upstream lineage but remain
  non-training artifacts.
- FineMath, Dolma, and authored-source artifacts stay staged until their exact
  source-specific license and redistribution bundles are complete.

## Frontier audit mirror

Dataset commit
[`de17529bd3ba9ea67355c26985b70350e6b8377f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/de17529bd3ba9ea67355c26985b70350e6b8377f)
adds seven source-safe files and passed a byte-for-byte post-upload replay. The
512-row frontier screen has population receipt
`b772920d9c86d5eddeae69f338e53a7d1b520f161f034b0616c3ee2140088631`
and zero-pair duplicate receipt
`26ebacefe45c320a9f319ecf42f63fe5c510236879581e1cbc48293bc65fc8b1`.
The 91-row PleIAs/Nemotron expansion has population receipt
`2c91d84c8ed64a008f46c72062a5f387ce6bebac1a08ad08fa41d9079a05b5eb`
and zero-pair duplicate receipt
`35df65eee58e3a4b67cb2f409666a201350cdba8c27438ff992b53a7b3397f8b`.

The combined exact-content report covers all 1,879 candidates in the five
screen populations and reports zero cross-population pairs under receipt
`e31954f5bd2b220004c6b19c0dd35949052f74a464f0ad009af476e2f6dff0be`.
This does not estimate full-reservoir or semantic duplication. Only lineage,
population receipts, duplicate receipts, and the combined statistics were
published; raw candidate text and evidence-bearing judgments remain local.

Credentials never enter the registry. The local `.env` is ignored by Git,
mode-restricted, and used only by authenticated clients.
