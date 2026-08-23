---
license: other
pretty_name: Sai Data Compiler Registry
language:
- en
tags:
- pretraining
- curriculum-learning
- data-quality
- llm
---

# Sai Data Compiler Registry

This repository is the durable, organized evidence and data registry for the
Sai Initiative. It implements a data compiler, not a single undifferentiated
mixture. A file appearing here does not make it training-ready.

> **Important split warning:** Hugging Face may display a `train` or `test`
> split inherited from an upstream Parquet dataset. These are source-layout
> names, not Sai verification labels. The current upstream `train` views are
> not approved Sai training data, and upstream `test` files are not the Sai
> evaluation boundary.

The target is an English-first polymath model trained through a spiral
curriculum. Raw datasets are source material. They must pass provenance,
rights, benchmark-contamination, duplication, semantic quality, grounded
transformation, prerequisite, curriculum, and final stream-replay gates before
training admission.

## Admission states

`source reference -> candidate -> model judgment -> independent verification -> curriculum -> training`

Only the final state may be consumed by a training job. Current exact
training-ready volume is **zero bytes**. No 4B training run is authorized by
this registry.

The physical registry currently contains **8,802,247,613,960 bytes**
(8.802247614 TB decimal; 8.0055975686 TiB), exceeding both the 8.5 TB decimal
capacity target and the earlier 8 TiB binary target. This is source-lake
capacity, not accepted corpus volume.

## Layout

- `sources/`: byte-identical, revision-pinned source snapshots. These remain
  candidate material and preserve their source-specific licenses and terms.
- `registry/`: immutable upstream revisions, source manifests, rights routing,
  benchmark boundaries, pilots, and audit receipts.
- `evidence/`: source-safe publications and measured conversion-yield receipts.
- `candidates/`: bounded candidate populations whose source and redistribution
  boundaries permit publication.
- `compiler_judgments/`: source-bound Hermès classifications and transformation
  plans; these are not automatically training examples.
- `curriculum/`: Sai-authored concept graphs and prospective spiral policies.
- `verified/`: reserved for independently verified derived representations.
- `training/`: reserved for final decontaminated, tokenized,
  curriculum-scheduled shards with complete lineage.

## Current evidence

- The physical source lake contains 13,974 LFS objects and
  8,802,247,613,960 bytes (8.0055975686 TiB) at immutable data head
  `cc8576fbb3f949bdaf59049a150c1fa1d35f47c3`. Every included object was
  replayed against its pinned upstream size and SHA-256 with zero mismatches.
- The lake includes complete selected snapshots of FineMath-4plus, three
  Nemotron specialist datasets, UltraData-Math L1, 31 Common Pile families,
  and 10,000 PleIAs Common Corpus data shards, plus 1,250 path-ordered
  FinePDFs shards. Hugging Face stopped the next FinePDFs batch at the public
  storage boundary; no partial sixth batch was committed.
- The exact file-level manifest and receipt are published under
  `evidence/materialized-source-lake/20260825-r1/`. The canonical receipt is
  `0715eefc3c3bda8ee800fc4c80155df461055da3bbf2a473ad6c93cf93bea9d8`.
- A 38-source admission matrix joins every materialized byte to the exact
  rights work route and keeps ten independent compiler gates fail-closed. Its
  canonical receipt is
  `e70b65ebec4d451be5d4a7094fe798e1154019a0db79cf64d99ec1ff6ee26ab6`.
- FineMath-4plus has a complete 64-shard mechanical census covering all
  6,699,493 rows, 34,126,971,204 UTF-8 text bytes, and 9,573,187,002 upstream
  tokens. It found zero byte-exact duplicate rows and seven normalized
  duplicate rows. The source-safe publication receipt is
  `bb578f5e969e8d15d96ae40ae3511d4dd6d2d9c42e834e5c641204719d53e4c2`.
- A conservative FineMath pass retained 56,654 of 6,699,493 rows before the
  official benchmark boundary and 52,277 afterward. It rejected 4,377 rows:
  1,422 with word-shingle overlap, 3,290 with eligible-code overlap, and 335
  with both. An independent parallel replay reproduced receipt
  `b0ba86aaa60dddfdfae6653882d489fde1ecf3ab0f043d9a3954bdd38e191277`.
- A strict bare-MCQ-answer-key filter found zero matches in those 52,277
  conservative FineMath survivors, while the broader completed Hermès audits
  already flag 36 answer-farm, 149 SEO/content-farm, and 97 corrupted rows.
  These remain exclusion evidence, not acceptable training examples.
- A source-agnostic mechanical gate now replays 8,323 distinct candidates from
  12 current populations. It routes 8,313 to mechanical pass, holds nine for
  duplicated-boilerplate cleanup, and hard-rejects one contextless scored
  physics answer sheet that also contains 136 embedded backspace controls.
  There are zero cross-population candidate-identity overlaps. The source-safe
  publication is under
  `evidence/source-mechanical-quality-gate/20260826-r1/` with canonical receipt
  `c25127e13c579bb066b887d264da1905bd78f2f3d24c183bba547ea019a2bf66`.
  A mechanical pass is not semantic or training admission.
- Completed Hermès receipts contain 6,751 proposed cross-domain assignments
  across 730 directed labels. A 512-pair, source-disjoint development proposal
  population has been frozen across 290 labels. No proposal is called
  connection training data until paired-source synthesis and independent
  verification complete.
- Two exact source inventories reference 23,680,076,298,761 physical candidate
  bytes (21.5369039313 TiB) across 42,600 files. The inventory is broader than
  the physical lake; references are not claims of local custody or
  training-ready text.
- Six immutable audit populations contain 2,103 rows.
- Two bounded Common Pile pilots contain 3,290 benchmark-disjoint,
  within-source near-deduplicated candidates with attribution manifests.
- A complete Common Pile PEP census yields 567 benchmark-disjoint,
  near-deduplicated candidates awaiting Hermès compilation.
- A 1,024-row CC0 arXiv temporal screen leaves 1,023 benchmark-disjoint rows and
  zero near-duplicate pairs for Hermès compilation.
- A complete text-free census of the same exact arXiv snapshot covers all
  2,504,679 rows. It excludes 1,060 prior-audit identities and 45,463 short rows,
  measuring 2,458,156 mechanically eligible unique rows containing
  2,380,856,330 UTF-8 text bytes. These are mechanical candidates, not approved
  training bytes.
- The portable r7 conversion-yield ledger separates referenced, sampled,
  mechanically eligible, and training-ready quantities and contains no
  machine-local absolute paths.
- Fourteen strict FineWeb-Edu Hermès compiler judgments and Sai-authored
  foundational-syllabus graph artifacts remain preserved as earlier evidence.

## Rights and source boundary

Source-specific rights govern every item. The repository-level `other` label
does not replace source licenses, terms, attribution, or share-alike
obligations. Gated Institutional Books text is not redistributed here. A
source snapshot appearing under `sources/` is still non-training-ready until
its required source-wide or per-row rights route, decontamination, deduplication,
quality compilation, and final admission evidence are complete.

## Verification

Published evidence is hash-manifested. Important releases include canonical
receipt hashes, file SHA-256 values, immutable upstream revisions, and remote
replay commits. Code, validation contracts, exact status, and the research
ledger are maintained at
https://github.com/GodlyDonuts/sai-initiative.
