# Sai Stack-Edu Python metadata audit — 2026-08-22

This receipt records a metadata-only audit. It did not retrieve or inspect code
content from Software Heritage, authorize training, or authorize the 4B run.

## Frozen source

- Dataset: `HuggingFaceTB/stack-edu`
- Revision: `eeec5caac5cc3758a18f1d3ba4416837a9ba814c`
- Parquet member: `Python/train-00000-of-00005.parquet`
- Bytes: `500150393`
- SHA-256: `f0e25975bff184163a7ff1aca53678617c6eace1328b1b2c51a39ff79a6262bd`
- Sai commit: `6dae4d60016aed8e8bb29d22158785b68a316c33`

The upstream metadata schema and dataset revision are pinned to the public
[Stack-Edu dataset](https://huggingface.co/datasets/HuggingFaceTB/stack-edu).
Current opt-out replay must use the current
[Stack v2 record](https://huggingface.co/datasets/bigcode/the-stack-v2) before
any nominated blob can advance.

## Execution

- Newton job: `770555`
- State: `COMPLETED`, exit `0:0`
- Elapsed: `242` seconds
- Restarts: `0`
- Node: `evc2`
- Resources: four CPUs, 16 GiB, zero GPUs, no requeue
- stdout SHA-256: `65afa0d7ffcc02b39abf5785453124020fc51c2dacd3cf851ee21d88849f2726`
- stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The job performed the full scan, wrote the outputs atomically, and then reopened
the source and replayed every decision. The durable evidence root is:

`/lustre/fs1/home/sa305415/sai-initiative/artifacts/external/stack_edu_eeec5caa_python_shard0_r1`

Every member is a single-link regular file in a nonwritable evidence tree.

## Result

| Measurement | Value |
| --- | ---: |
| Rows | 5,057,204 |
| Declared code-content bytes | 13,818,724,813 |
| Rows labelled permissive | 913,638 |
| Permissive rows with no detected license | 3,698 |
| Rows labelled no-license | 4,143,566 |
| Conservative metadata candidates | 102,579 |
| Candidate fraction | 2.0283% |
| Candidate declared bytes | 202,585,023 |
| Duplicate blob IDs | 0 |
| Duplicate repository paths | 87 |

The candidate policy required all of the following: `license_type=permissive`,
a nonempty detected-license set drawn entirely from Sai's narrow permissive SPDX
allowlist, integer Stack-Edu score 4 or 5, UTF-8 encoding, and declared length
from 128 through 1,000,000 bytes. A metadata candidate is only a nomination for
later source review.

## Evidence hashes

- Canonical receipt self-hash:
  `a74f9e1ffb7badc40b9e912672fe4c6bb044de3ae7099950d28a1afd462c4f88`
- Receipt file SHA-256:
  `e7d14f6a3d01a8a29aca615317b9346bc97a493e5d88691e7ebefa40fa81b3b1`
- Balanced 64-row metadata sample SHA-256:
  `b230c2cce3589a0fcec64d80784b93a9d0b38f314fbc65eac1802bed307d5338`
- Balanced sample ordered SHA-256:
  `4a66e33d7e1ebb1807c10fda4d58572c6c287eed9ea443c0d074f6b2efd2fb21`

## Admission decision

`training_authorized=false` and `four_b_training_authorized=false`.

No Stack-Edu code is admitted until current opt-outs are replayed and the exact
content passes provenance/license review, secret and personal-data scanning,
global and benchmark deduplication, benchmark decontamination, semantic quality
review, prerequisite annotation, and curriculum-placement review. Even after
those gates, the source must win a matched-data ablation before receiving
material mixture weight.
