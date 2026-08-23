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

Dataset commit
[`bd7d7cd92bfe61d8b9b0dfda8790d11d0fa3cdef`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/bd7d7cd92bfe61d8b9b0dfda8790d11d0fa3cdef)
adds the completed 128-row original-reservoir aggregate under receipt
`c0706f92535aded29c679fff5c35798a6380c01b58dc9bdf95ffd155f9a76359`.
Commit
[`e388950b30231779215b688b1defbfcbf785f3df`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e388950b30231779215b688b1defbfcbf785f3df)
adds its deterministic source-work ledger under receipt
`7cd1a6b040eaa00a40eb37f2578045780815931d6f712a43d5bd33848a4e250e`.
The ledger is descriptive triage, not a source-yield estimate or training
admission.

## Benchmark boundary mirror

Dataset commit
[`ad178281de02625f043359a89070e905944452b9`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/ad178281de02625f043359a89070e905944452b9)
stores the non-reversible official public-benchmark boundary under
`registry/benchmark-boundaries/official-public-20260824-r1/`. The 895,351,296
byte word index has SHA-256
`f470834f84ff24dbd1bee66c115a460e913b5384470cc8a91e09c56459111acd`;
the 61,025,632 byte code index has SHA-256
`8e1238e2fe382b532638dd82692647ebf431eb99905f2c64c9b487654f3ecb86`.
Remote LFS object sizes and SHA-256 identifiers, plus the downloaded receipt,
replayed against the local files. The index contains hashes only, never raw
benchmark prompts or answers.

Dataset commit
[`5dc89bfeceadf56663a8f00c479f5d41d5229671`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/5dc89bfeceadf56663a8f00c479f5d41d5229671)
adds the five source-safe population screens. Each screen binds its population
receipt, index receipt, policy, aggregate source/stratum counts, and ordered
decision digest. Individual candidate decisions and source text are not
published. The exact receipts are:

- original 128: `a851a4210bdd48b525b46cada58eabca77c932afee00d86109f55b048dbb1613`;
- weighted 1,024: `e28c7afb9799671d5fc7d6275fd82299da23271380b964a85eb09b71ee903db4`;
- frontier 512: `f2e3c82b1386f6c668757a53fc6dd5e554ea79b6ecf992667ac155d3d74c933c`;
- Common Pile 124: `71fcf44f96340feb73c399c82b4be00fd8c555bf5287c78b1464d9ad06bbb54c`;
- PleIAs/Nemotron expansion 91: `cca8dfbd408d6f4bf8914e8e64b78c75db05cf927734131160f92099d741524b`.

Credentials never enter the registry. The local `.env` is ignored by Git,
mode-restricted, and used only by authenticated clients.
