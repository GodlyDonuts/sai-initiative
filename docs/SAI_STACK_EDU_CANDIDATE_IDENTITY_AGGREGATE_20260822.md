# Sai Stack-Edu Python candidate identity aggregate — 2026-08-22

This is the completed metadata-only identity population derived from all five
pinned Stack-Edu Python shards at revision
`eeec5caac5cc3758a18f1d3ba4416837a9ba814c`. It contains no acquired source
content and authorizes neither source retention nor training.

## Execution evidence

- aggregate job: `770639`
- state: `COMPLETED`, exit `0:0`, elapsed `3,100` seconds, zero restarts
- node/resources: `evc1`, 8 CPU, 48 GiB RAM, no GPU
- source commit/runtime: `df7969710708ab704f5b30dae21bd27242a53b73`
- exact five-shard dependency: `770634:770635:770636:770637:770638`
- stdout SHA-256:
  `df14705adb11028dde4050ce6aa75c2c1d30043a3a2c1a6c5ed515d8be98c875`
- stderr: empty, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The wrapper printed the same qualified status and canonical receipt identity
after creation and after its full replay:

- status: `candidate_identity_population_deduplicated_content_not_acquired`
- canonical receipt identity:
  `ec6b0aa945f5df800caa03176ece2cad2f1e9f6c6bdfa5417cfc08caddc80762`

## Frozen outputs

- `candidates.jsonl`: 215,285,040 bytes, mode `0444`, one link, SHA-256
  `7429c9d4189fb33d1eb92ae255b77b02d6e533dc33598b4025931397d35007c5`
- `receipt.json`: 4,875 bytes, mode `0444`, one link, file SHA-256
  `39d2dec5d98d8d5c18b4d8743a8b3ec67b00446249762c7926168707e936c9d6`
- ordered candidate identity SHA-256:
  `506dcf3b5a8b50a3f2b06fe37d107e0b283ac052e4365b5d91cf7425ff59598f`

An independent lightweight pass reopened every JSONL row and recomputed:

- input candidates: 514,566
- unique blob identities: 514,566
- unique repository/path pairs: 514,559
- repositories represented: 127,672
- candidate file SHA-256: exact match
- all blob identities unique: true
- canonical receipt hash recomputed from the unsigned payload: exact match

Seven repository/path pairs contain distinct blob identities. They are reported
rather than silently removed because the upstream blob identity, not a mutable
path, is the exact-content key. The population declares 1,024,974,382 bytes of
potential source content. No cross-shard duplicate blob was found.

## Interpretation

This closes only the old Stack-Edu metadata filtering and exact blob-identity
deduplication stage. Every row still requires intersection with the current
Stack v2 opt-out-enacted snapshot, exact source-byte acquisition under the
applicable access terms, SHA-1/SHA-256/length verification, attribution, bounded
secret and PII findings, global exact and near deduplication, benchmark
decontamination, usefulness and correctness review, semantic prerequisite
placement, curriculum rehearsal, and matched source-addition evidence.

Accordingly, both `training_authorized` and `four_b_training_authorized` remain
false. The aggregate is a candidate universe, not Sai training data.
