# Sai Public Domain Review scoped candidates — 20260825-r1

This directory contains a bounded, replayable candidate representation of
Public Domain Review text. It is **candidate data**, not a finished training
shard and not an assertion of legal clearance or content-quality verification.

## Exact population

- 1,342 frozen pilot identities were replayed against their live source pages.
- 1,253 reproduced both the frozen Common Pile extraction and the active
  quotation-excluded scope hash: 995 Collections, 243 Essays, and 15
  Conjectures.
- 85 source pages drifted and four pages were unavailable; all 89 are absent
  from the candidate file.
- The retained text contains 5,919,449 UTF-8 bytes across 1,253 rows.
- 883 `blockquote` or `q` elements containing 412,039 codepoints were removed
  from the retained pages. Page HTML was never persisted.
- A post-transformation replay against Sai's pinned official benchmark boundary
  found zero exact 13-token word-shingle hits and zero eligible 8-token code
  hits. All 1,253 rows therefore survive this bounded screen.

The benchmark screen covers the exact public boundary recorded by its receipt;
it does not prove absence from every possible benchmark, every future release,
or the full upstream source.

## Source and reuse obligations

The source pages are published by
[Public Domain Review](https://publicdomainreview.org/). Collection-page scope
is bound to the site's
[reuse policy](https://publicdomainreview.org/reusing-material/); Essay and
Conjecture scope requires an exact page-specific link to
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The dataset
excludes page images and explicitly tagged quotations because those can carry
different rights.

Each candidate retains:

- the exact source-page URL;
- the upstream dataset row and source license;
- the source-provenance and rights-record hashes;
- the active scope-audit result hash;
- attribution-required and share-alike-required flags; and
- the current source-response hash used to reproduce the scoped text.

Anyone reusing the candidate text must preserve attribution, link back to the
corresponding Public Domain Review page, identify CC BY-SA 4.0, indicate this
quotation-removal transformation, and distribute adaptations under compatible
share-alike terms. This documentation records source evidence and obligations;
it is not legal advice.

## Immutable hashes

- Materialization receipt SHA-256:
  `52484c5f8b22d79b231e71d2d03962fd10ea18b29c6740c02b86afd25ebd7741`
- Scoped candidate file SHA-256:
  `04626195b471db0a543a1de7a7b3feb672696ac0069da9a1025c64e11b2ddbf6`
- Ordered scoped-record digest:
  `e1c40e40e9c8848efe4a919d5795d4ea37c9537b6eb7c12f201f92ebdbee2f0c`
- Text-free materialization results SHA-256:
  `2a516e1a91b873581e21c9c32c56daa92926e63bf35c572e1e2604455d4b7e0b`
- Post-scope decontamination receipt SHA-256:
  `9a051d33874a8515938d072914dbe3888e7cde52ed7eff7754c12d7efd528097`
- Text-free contamination decisions SHA-256:
  `f04ab68b952e187671bb3725ad80cc5665b3f9df43f9b81570a0c0e862518069`

The candidate file remains `content_quality_verified=false`,
`legal_clearance_established=false`, and `training_ready=false`. The live
Hermès compiler and independent work lanes decide which rows deserve later
representation verification; no compiler verdict alone can promote a row.

All seven published files were force-downloaded and replayed from exact
Hugging Face dataset commit
[`6885a18a0a98eb10c3d5d0e73ad276dd49a99a0d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6885a18a0a98eb10c3d5d0e73ad276dd49a99a0d).
The remote candidate file, all 1,253 candidate signatures, all 1,253 text-free
contamination decisions, both receipts, and the zero-contamination conclusion
reproduced exactly.

## Grounded representation stage

The clean materialized population can now be converted into an exact
compiler-bound representation workload after the complete bounded compiler
population closes. The population builder joins each clean PDR identity to its
content route, rights route, compiler receipt, and requested derivative types.
It never treats a model route as an admission decision and skips rows for which
the compiler requested preservation only.

The Hermès generation contract emits one entry for every requested derivative
type, literal source citations, optional prerequisite-edge claims, and
cross-domain bridge candidates whose external side is explicitly unverified.
The aggregate separates generated text from source text, replaces published
source citations with hashes, and carries CC BY-SA attribution and share-alike
requirements into every derived record. Generated candidates remain
`training_ready=false` until post-generation contamination screening, global
deduplication, factual/source-claim verification, bridge anchoring, and
independent representation review complete. This is code-only preparation; no
generation or model training has been launched by this stage.
