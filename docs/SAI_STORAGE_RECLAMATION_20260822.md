# Sai data-build storage reclamation — 2026-08-22

This receipt records an explicitly authorized, irreversible deletion of four obsolete, re-downloadable Shohin model-weight trees to make room for the Sai 4B data build. It does not authorize deletion of Sai data, code, runtimes, checkpoints, or live-job outputs.

## Admission checks

- Newton scheduler showed the only active or dependent jobs as `770154`, `770155`, `770952`, `770503`, and `770505`; their commands, work directories, logs, and exported paths were all under `/lustre/fs1/home/sa305415/sai-initiative*`.
- No active process command referenced any deletion target.
- Every target was a literal, nonsymlink directory on device `2121101648`, owned entirely by UID `1227834669`, with zero symlinks and zero non-owner entries.
- Each target was renamed to a unique same-parent quarantine before deletion. Only those four quarantines were made owner-writable and traversed with one-filesystem, depth-first deletion.
- All four original paths and all four quarantine paths were absent after deletion.

## Permanently deleted targets

| Target | Allocated KiB | Entries | Source revision SHA-256 | Config SHA-256 | `SHA256SUMS` SHA-256 |
|---|---:|---:|---|---|---|
| `/lustre/fs1/home/sa305415/shohin/artifacts/external/qwen3.6-35b-a3b-995ad96e` | 70,241,496 | 130 | `3a1dd18603518dd5b9cce88ed68b03b15690cfd2ec08fd9882cfaaf4ffd2de60` | `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99` | `06c9d8d8419244f2d001cb351e164f356718d9d77138e898b13afee35856f56e` |
| `/lustre/fs1/home/sa305415/shohin/artifacts/external/nemotron-3-super-120b-a12b-fp8-7d7e5797` | 125,371,256 | 49 | `ad9469e785107fdaee8fbca749a039e5dd9fc7dde9b6ab209a17848b0047ec54` | `ff5d6d643b288d4149b0bf820ecb5fe87dd9bbc08b6b811241c57840e11e30e3` | `8bb8bb898794651791de9d79c1041fe0ec6ad0f54a97b03f52620bd6e245ce92` |
| `/lustre/fs1/home/sa305415/shohin/artifacts/external/nemotron-ultra-transfer-basis-183968f_r1` | 2,552,188 | 8 | `72b9c31cfe862b4bb903d0ba7da4452df43875fdf32cd2c388ca836164338c8c` | `0c939f324c8910f5ebdafbe2a56d7e4e074c50042a3b4f26326bf71a3fe33929` | `047a261ce246c5c71f84454a491e7ab7448d320cdf2ab7c6836875183058688b` |
| `/lustre/fs1/home/sa305415/shohin/artifacts/external/mixtral-8x22b-instruct-cc88a6c` | 274,672,000 | 72 | `8642b7c629dc46cc5ab978b352bcf020ad64b826cf933154396aa0cf041b0340` | `9c4a6138d84029ab666943613e3d5844d2ea8fd6149f44f77188c62e2915e0f5` | `46b8475d98e2a49f9a81329287beb9d450dfd4d7a74886e8780708764a8f3fe7` |

The exact pre-deletion allocation sum was **472,836,940 KiB** across **259 entries**. These bytes are permanently nonrecoverable from local storage; upstream source identities and the complete model-manifest hashes remain recorded above. Small acquisition records stored outside the deleted roots were preserved.

## Settled result

- Pre-deletion quota observation: `1,030,530,576 KiB / 661,537 files`.
- Stable post-deletion observations at `2026-08-22T18:44:19-04:00` and `2026-08-22T18:44:24-04:00`: `557,697,012 KiB / 661,281 files`.
- Observed net recovery: **472,833,564 KiB** (approximately **450.93 GiB**) and **256 files**.
- The 3,376 KiB / 3-entry difference from the target allocation sum is attributable to concurrent live Sai job activity during the observation interval, not a target mismatch.
- The live Sai graph remained unchanged after deletion: `770154`, `770155`, and `770952` running; `770503` and `770505` dependency-pending.

This reclamation removes storage capacity as the immediate constraint on constructing the provenance-bound, deduplicated, curriculum-ordered 4B training stream.
