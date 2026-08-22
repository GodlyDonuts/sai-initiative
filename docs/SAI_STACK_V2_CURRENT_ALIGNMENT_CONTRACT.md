# Sai Stack-Edu current-release alignment contract

Stack-Edu quality metadata is useful evidence, but its frozen revision predates
the current opt-out-enacted release of The Stack v2. An older row therefore
cannot become Sai training data merely because it has a high educational score
and a permissive-license label.

The current authority is
[`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2)
at revision `e565caa3a78c2423bd374333a472b049eb090e47`, release `v2.2.0`.
The official dataset card says this release removes repositories that opted out
before `2026-07-29`, removes repositories belonging to users or organizations
that are no longer on GitHub, and requires users to update to the most recent
usable release. It also states that bulk source-content access requires an
agreement with Software Heritage and INRIA and that original license and
attribution requirements remain binding.

## Two immutable metadata steps

`sai-align-stack-edu-current freeze-snapshot` accepts only:

- the complete Python Parquet shard set from the exact current revision;
- the exact downloaded dataset card containing the release, cutoff, update,
  and Software Heritage terms;
- a self-hashed access-evidence receipt binding the same card and revision;
- regular, single-link files whose bytes, SHA-256 hashes, row counts, shard
  indices, and required columns replay exactly.

The resulting `sai-stack-v2-current-python-snapshot-v1` receipt remains
metadata-only and authorizes neither content retention nor training. A newer
usable Stack release invalidates its operational currency and requires a new
prospective snapshot rather than mutation of the old receipt.

`sai-align-stack-edu-current align` then reopens the complete deduplicated
Stack-Edu candidate population and intersects it with current Stack metadata on
the exact tuple:

`(repo_name, path, blob_id)`.

An old candidate is retained only when the exact tuple still exists and the
current row is Python, non-vendor, non-generated, permissively licensed, and
uses only the conservative Stack-Edu license allowlist. Absence is treated as
removal, not as a transient lookup failure, because the current snapshot must
be a complete shard set. Duplicate current occurrences are counted and a
deterministic first eligible occurrence is retained.

The alignment receipt records every removal reason, current row count, matched
occurrence count, exact parent identities, and the ordered output hash. Its
validator rescans all current metadata and reconstructs the full intersection.

## Still not training data

Current-release membership closes only the opt-out drift between the older
Stack-Edu snapshot and the latest usable Stack v2 release. Every retained row
still requires all of the following before source admission:

1. authorized Software Heritage content acquisition;
2. exact compressed-object and decoded-byte verification;
3. Git/SWH SHA-1 identity plus independent SHA-256 and byte-length receipts;
4. license and attribution preservation;
5. secret, PII, malware, generated-code, and quality scanning;
6. exact and near-content deduplication across the complete Sai mixture;
7. benchmark and development-set decontamination;
8. semantic concept/prerequisite annotation and curriculum placement;
9. a matched same-record ordering experiment with retention vetoes.

No missing gate may be inferred from another. In particular, current presence
does not prove quality, a high quality score does not prove pedagogical order,
and a curriculum label does not authorize 4B training.
