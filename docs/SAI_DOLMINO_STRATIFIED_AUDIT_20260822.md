# Dolmino 100B stratified source audit — 2026-08-22

This is a deterministic content diagnostic for the exact public dataset revision below. It is not a training-source admission. Its purpose is to separate unique content, source metadata, license disposition, and future exposure weights before Sai builds the 4B curriculum.

## Frozen identities

- Dataset: `allenai/dolma3_dolmino_mix-100B-1025`
- Revision: `f23942ae8a8114af6e992efe8188ce8c531acd16`
- Inventory receipt: `7e62c6006697110eba5857831d11903a14c41f76cb3547b866b69a4aa076ebf2`
- Audit specification SHA-256: `b57346afd617f1fa6a0e0ad66b28d9fcc1282bc1d2ee78697807c1076946e0bf`
- Deterministic plan receipt: `319625ca6483390ef12a68fc1058215ae087b2006c8a82c894ecfb745ec0e140`
- Plan file SHA-256: `3a02a4c992ef5c0fbc227124d431c2ae27d65b5f5be6abaa16cbe104e677294b`
- Selected-member identity SHA-256: `32cfb3e120eb5672cc335efa56c0fb1c6f3c12b80628aca6c288cac43fe03f8e`
- Aggregate receipt: `03cb36f0dbc85483b6fa732ce8fdf7bacd5bb12c573e6a7df06a902b24a5abdd`
- Aggregate file SHA-256: `619b300bccb472dab066b9a70360f8f62087bc4d0e71dbd17a43ad35cc179fd6`

The plan selected 167 shards totaling 2,051,280,977 compressed bytes by salted hash rank, never by file size. Each member was downloaded alone, matched against the frozen upstream byte count and SHA-256, audited completely, and removed before the next member was acquired. All 167 member audits passed.

## Measured population

| Stratum | Shards | Physical rows | Within-shard repeated texts | License metadata |
|---|---:|---:|---:|---|
| High-quality web | 48 | 26,800 | 5 (0.0187%) | 26,800 missing |
| High-quality science PDFs | 36 | 1,570 | 99 (6.3057%) | 1,570 missing |
| Stack-Edu language/quality bands | 60 | 1,576,289 | 0 | 346,961 permissive; 1,229,328 `no_license` |
| Core math/code/STEM | 7 | 167,354 | 225 (0.1344%) | 5,203 permissive; 15,777 `no_license`; 146,374 missing |
| Synthetic instruction/reasoning | 16 | 238,087 | 0 | 238,087 missing |
| **Total** | **167** | **2,010,100** | **329 (0.0164%)** | **352,164 permissive; 1,245,105 `no_license`; 412,831 missing** |

Unlike the previously audited Dolma 3 foundation-mix sample, this Dolmino sample has no repeated document IDs within any shard and very little repeated text within shards. That does **not** prove global uniqueness: duplicate identities across the 167 sampled shards or across the complete 71,090-shard dataset were not measured. Canonical global deduplication remains mandatory.

The license result is the immediate admission constraint. Only 17.52% of sampled rows carry permissive metadata; 61.94% explicitly carry `no_license`, and 20.54% have no license field. In the dominant Stack-Edu stratum, 77.99% are `no_license`. Sai will not treat the dataset wrapper's ODC-By label as a per-document or per-code-file license decision. Stack content enters the canonical pool only through the accepted-license allowlist and preserved provenance; missing and `no_license` rows remain excluded unless independently resolved.

## 4B data decision

Dolmino is retained as a **late curriculum candidate**, not as the foundational stream and not as an opaque 100B blend.

1. High-quality web and science rows remain candidates after benchmark decontamination, source-policy review, quality filtering, and global deduplication.
2. Stack-Edu rows require per-file permissive-license admission and current provenance/opt-out replay.
3. Core math/code/STEM rows are selected component by component; upstream repetition or mixture weight is not treated as unique information.
4. Synthetic instruction and reasoning rows are delayed until the reasoning/specialization phases, preserve generator and prompt provenance, and receive separate factuality and contamination checks.
5. The final stream stores each admitted canonical document once. Rehearsal and upsampling are represented in a separate, explicit exposure ledger aligned to curriculum phases and optimizer-update boundaries.

No row was selected for training by this audit. The next critical operation is a scalable complete-source canonicalization pass that performs cross-shard/global exact and near-duplicate clustering, license filtering, benchmark decontamination, and phase assignment before tokenizer training or packing.
