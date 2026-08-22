# Sai FineMath Filter V1 Result — 2026-08-22

## Immutable input and policy

The first conservative filter was run against the exact previously audited
FineMath `4plus` shard:

- dataset revision: `e92b25a616738fe95dc186b64dfb19f9c8525594`;
- source file: `finemath-4plus/train-00000-of-00064.parquet`;
- source bytes: `286,267,316`;
- source SHA-256:
  `5d0b26114b4cf309c82a4ba8f6f45857480b27812e23a46b61b45e8bbed61fe5`;
- rows: `104,680`;
- filter-policy SHA-256:
  `8ecb2df64f14429f5596393104c7456ef3dabab1cf4818158fe47d1dcd12f57c`.

V1 required upstream score 5, `found_math=true`, language confidence at least
0.98, a valid non-denied host, at least 160 words, two distinct mathematical
signal classes, two distinct explanatory-structure classes, no declared risk
pattern, no excessive embedded links, and no repeated exact text after its
first occurrence. The policy was committed before this full-shard result.

## Terminal result

V1 accepted **0 / 104,680** rows. This is an immutable empty-candidate result,
not a training corpus:

- receipt status: `filter_empty_no_candidate`;
- receipt self-hash:
  `9cdca70b1b19710f802fe4143d3565769eeffef5217c2e490705528e724c06aa`;
- receipt-file SHA-256:
  `e5f941401d169dbaa825f4f01d56d5ff47f1e5ce30b3d87b408f45685c77685b`;
- accepted output: 0 bytes, SHA-256 `e3b0c442...b855`;
- rejected review packet: 64 deterministically selected rows, 378,682 bytes,
  SHA-256
  `042bbe42a9b9a40f9c4c4fe747d7d27b6f4af710249c4e40a5f83c0f39fb4fc8`.

Rejection counts are overlapping, not a partition. The largest were language
confidence below 0.98 (`104,313`), upstream score below 5 (`94,692`), absent
`found_math` (`68,295`), insufficient distinct math signals (`57,054`), and
insufficient explanatory structure (`34,075`). The selector also rejected
`6,882` denied answer-farm hosts and the previously declared essay-service,
gambling, answer-key/homework, and SEO pattern matches.

## Calibration diagnosis

The empty result does **not** establish that FineMath contains no useful rows.
The language-confidence floor was badly calibrated to this source's metadata:
its p50/p90/p95/p99 values are approximately
`0.8791 / 0.9471 / 0.9576 / 0.9730`. Only `367` rows in the entire shard reach
0.98.

A read-only funnel using every unchanged non-language V1 rule found `3,114`
rows that pass all other criteria. Of those, language floors retain:

| Language floor | Rows passing every other V1 rule |
|---:|---:|
| 0.98 | 0 |
| 0.97 | 2 |
| 0.95 | 65 |
| 0.90 | 690 |
| 0.80 | 1,760 |

The V1 policy and zero result remain unchanged. Sai will not quietly relax the
threshold and call the same experiment a pass. The next step is a separately
versioned, prospectively frozen human-review ladder comparing no language floor,
0.90, and 0.95 while holding every other selection rule fixed. That review will
measure accepted-row precision and rejected-row false negatives before any
FineMath source-addition screen.

No row from V1 is admitted to training. No result here authorizes an optimizer
job, architecture promotion, or 4B training.
