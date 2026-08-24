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
- A source-agnostic mechanical gate now replays 10,371 distinct candidates from
  13 current populations. It routes 10,360 to mechanical pass, holds nine for
  duplicated-boilerplate cleanup, routes one short contextless bibliographic
  form to context review, and hard-rejects one contextless scored physics answer
  sheet that also contains 136 embedded backspace controls.
  There are zero cross-population candidate-identity overlaps and zero exact
  source-content overlaps. The current local source-safe publication has
  canonical receipt
  `df34d6507032269351df3d841032e068de5ff986dcbcb7d5f92f212e98e82385`.
  Its 2,048-row OpenCoder addition and explicit content-duplicate census are
  published as 28 exact files under
  `evidence/source-mechanical-quality-gate/20260826-r3/`. All 28 files plus this
  card were downloaded back byte-identically from immutable dataset commit
  `444b1c482ff7e510d68f7e7115f1bf1d2087c936`. The authorized Stokes evidence
  mirror contains those files plus the card as 29 exact files under manifest
  SHA-256
  `929c6b46f7e4de7ca17c5fc360337465d33ec62ba284fbd5ca052c7d61a73c89`.
  A mechanical pass is not semantic or training admission.
- A host-diverse OpenCoder code-web audit replayed one exact 286,437,437-byte
  Parquet shard and all 197,882 rows. From 162,487 length-eligible unique
  contents, the 8,192-row screen rejected two mechanical failures and 39
  official-boundary overlaps. The frozen 2,048-row Hermès population spans
  1,922 web hosts with at most two rows per host. Receipt
  `53abfd09fb2bc71b17dba5b922c1eaa2c7752cb216654e1557b442701937e7c9`
  binds the exact source, card, population, lineage, and boundary indexes. Its
  frozen 276-row Hermès promotion screen passed only computer-science coverage
  and failed representation verification, quarantine, educational value, and
  technical depth. Full audit is stopped. The 286,437,437-byte local
  acquisition cache was reclaimed after all OpenCoder workers stopped; the raw
  shard was never uploaded here and remains recoverable at its pinned upstream
  revision. Screen and reclamation receipts
  `29a7ceed9841f99213d4087a40e0107277a07793b490760ee800242bcad7be70`
  and
  `a45fa68a77018c1b58900b58aa703ad295c249183528ec8495816fe95c6ac172`
  are published under
  `evidence/opencoder-promotion-screen/20260826-r1/`. The bounded result is not
  a whole-source quality estimate, and the card's MIT declaration does not
  establish rights provenance for every underlying web page.
- The exact 512-row frontier-source compiler is complete: 348 `retain`, 125
  `review`, and 39 `reject` verdicts, but only 25 rows route to representation
  verification while 244 route to quarantine. FineWeb2-HQ and both
  Ultra-FineWeb snapshots are bulk-paused; Nemotron specialized reasoning and
  UltraData-Math L1 remain targeted-recovery sources. Source-safe aggregate and
  decision receipts are
  `9c2d0e49d062c1886f9809b8852c4c48c38f41b40bea210631d1cc0f7236c6de`
  and
  `7656214825b6f66984c007f00a6f089c5d2c43791af6b775c466db56d952a48d`,
  published under `evidence/frontier-source-audit/20260825-r1/`.
- Source files are reclaimed only when the whole exact object is proven
  unusable, or after all retained rows have a byte-hash-verified filtered
  replacement. Rejected rows are excluded immediately; mixed raw shards stay
  until replacement custody exists. Provenance and benchmark-version evidence
  is retained when deleting its pointer would save no backing-object bytes.
  Pinned, re-downloadable acquisition caches that were never authoritative Sai
  custody may also be reclaimed after active dependencies end and source-safe
  audit plus recovery coordinates are durable; doing so does not assert that
  every upstream row is unusable.
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
