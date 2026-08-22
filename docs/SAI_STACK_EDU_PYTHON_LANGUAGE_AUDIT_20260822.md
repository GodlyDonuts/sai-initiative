# Sai Stack-Edu complete Python metadata audit — 2026-08-22

This is the complete five-shard metadata audit of Stack-Edu's pinned Python
population. It did not retrieve or inspect Software Heritage code content and
authorizes neither training nor the 4B Sai run.

## Frozen population

- Dataset: `HuggingFaceTB/stack-edu`
- Revision: `eeec5caac5cc3758a18f1d3ba4416837a9ba814c`
- Language: `Python`
- Members: exactly `train-00000-of-00005.parquet` through
  `train-00004-of-00005.parquet`

| Shard | Bytes | Source SHA-256 | Audit job | Elapsed | Receipt self-hash |
| ---: | ---: | --- | ---: | ---: | --- |
| 0 | 500,150,393 | `f0e25975bff184163a7ff1aca53678617c6eace1328b1b2c51a39ff79a6262bd` | 770555 | 242 s | `a74f9e1ffb7badc40b9e912672fe4c6bb044de3ae7099950d28a1afd462c4f88` |
| 1 | 500,201,312 | `59e042dc6d5448b935f908c0d41afaad73ba8f2dc33ecde4b5ef395e5689d9e1` | 770564 | 234 s | `324dc9eb438194dd744df6d97bbbe5676b95370cc217fbe2c505b233d443eb3d` |
| 2 | 500,172,909 | `976e623d0a8d2b8fb1c3fec0fbd1d7f237a80a19b32e91f399259cb33e9208f7` | 770565 | 257 s | `194c39f069e17d99d159f1bacf9d313f7b6d8f66cab15f2d1c929b5c10c0ae05` |
| 3 | 500,169,900 | `81d0de22657b77282312e38db7b1ea4edaf0fcb5e04d4fa0b276f2e4e68872b8` | 770566 | 243 s | `76ab81c1d3ed60fa3b7888617da2f071cd0dd34dabd02e89b5bf775b9bcdb285` |
| 4 | 500,100,572 | `6190c01cdb603cf602ac01f770a476f7b924630f2a46b1bed89622903d869a73` | 770567 | 242 s | `f1ec9c55c49ae491075cbe8d7d87ce0a919d9e8b2274d2e492fb86f54f58fd60` |

All five jobs completed `0:0` with zero restarts on `evc2`; every stderr was
empty. Each job scanned its complete shard, wrote a balanced metadata-only
sample and receipt create-once, then reopened the exact source and replayed all
decisions.

## Per-shard result

| Shard | Rows | Candidates | Candidate ppm | No-license | Permissive without detected license | Duplicate repository paths |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,057,204 | 102,579 | 20,283 | 4,143,566 | 3,698 | 87 |
| 1 | 5,057,204 | 102,801 | 20,327 | 4,145,411 | 3,707 | 90 |
| 2 | 5,057,204 | 102,576 | 20,283 | 4,146,026 | 3,751 | 99 |
| 3 | 5,057,204 | 103,224 | 20,411 | 4,143,781 | 3,739 | 106 |
| 4 | 5,057,203 | 103,386 | 20,443 | 4,143,851 | 3,641 | 101 |

No shard contains an internally duplicated blob ID. This does not prove that a
blob or repository path does not recur in a different shard.

## Complete-language replay

Aggregate job `770574` ran the immutable implementation at commit
`876cd507e94a3ad2e83e1c38e375058fac43d145`. It validated the complete shard
index set, reopened all five receipts and sources, recomputed every decision,
published one receipt, then independently repeated the five-shard replay.

- State: `COMPLETED`, exit `0:0`
- Elapsed: `1,189` seconds
- Restarts: `0`
- Node: `evc1`
- Resources: four CPUs, 16 GiB, zero GPUs, no requeue
- Aggregate canonical self-hash:
  `9a00f007c4d24d4aaefdb42b885fad1ab948ce4f75c81b82d7bfa4e85912a6f1`
- Aggregate file SHA-256:
  `0eb6cf0e36df2a6dd5ab14b198e5d251f0930692871a11121e2498bc9d8c5c00`
- stdout SHA-256:
  `4e9820da1ba895b343f339ab5283b20c92fb0351108a3554ffa4cf7dd7c42b9b`
- stderr SHA-256: empty-file digest
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The read-only aggregate root is:

`/lustre/fs1/home/sa305415/sai-initiative/artifacts/external/stack_edu_eeec5caa_python_aggregate_r1`

## Aggregate result

| Measurement | Exact value |
| --- | ---: |
| Shards | 5 |
| Rows | 25,286,019 |
| Declared code-content bytes | 68,977,578,447 |
| Rows labelled permissive | 4,563,384 |
| Permissive rows with no detected license | 18,536 |
| Rows labelled no-license | 20,722,635 |
| Conservative metadata candidates | 514,566 |
| Candidate fraction | 2.0349% |
| Candidate declared bytes | 1,024,974,382 |
| Within-shard duplicate blob IDs | 0 |
| Within-shard duplicate repository paths | 483 |

The policy requires `license_type=permissive`, a nonempty detected-license set
drawn entirely from Sai's narrow permissive SPDX allowlist, integer Stack-Edu
score 4 or 5, UTF-8 encoding, and 128 through 1,000,000 declared bytes. Passing
that rule nominates metadata only; it does not admit the corresponding code.

## Decision

`training_authorized=false` and `four_b_training_authorized=false`.

Before any nominated blob can become a Sai lesson, current opt-outs must be
replayed and exact content must pass source/license review, secret and personal
data scanning, cross-shard and global duplicate clustering, benchmark
decontamination, semantic quality review, prerequisite annotation, curriculum
placement, and a matched source-addition experiment. The aggregate explicitly
records `cross_shard_duplicate_identity_check_complete=false`; the summed 483
within-shard duplicate repository paths cannot be presented as a global
duplicate count.
