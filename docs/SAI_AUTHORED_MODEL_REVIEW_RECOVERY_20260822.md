# Sai authored-curriculum model-review recovery — 2026-08-22

Two independently configured model reviewers were intended only to prioritize
human inspection of the 127-row authored programming curriculum. They were
never human reviewers and could not qualify labels, data, training, or an
architecture.

## Terminal first attempt

Both initial jobs loaded their exact immutable model snapshots on an H100 and
failed deterministically on blind review row 0 after exhausting three frozen
generation attempts:

| Reviewer | Job | State | Elapsed | Node | Restarts | Failure |
|---|---:|---|---:|---|---:|---|
| Qwen3.5-9B | `770450` | `FAILED 1:0` | 562 s | `evc46` | 0 | `review row 0 exhausted frozen attempts` |
| SmolLM3-3B | `770451` | `FAILED 1:0` | 55 s | `evc46` | 0 | `review row 0 exhausted frozen attempts` |

The Qwen stdout SHA-256 is
`6409fd7714623b93a7c8342ee94891a03533f60f8392fb3e2e8b5eba795b344f`
and stderr SHA-256 is
`f044089abe762a81092c110caa3a1b6d37c924bd646edf917810807056297fb1`.
The SmolLM3 stdout SHA-256 is
`ae6553de7a5df10fc299a5a976cc99fa3f5d8bddbcc0f37e82571c6f7031be48`
and stderr SHA-256 is
`6c76414ab1c109686b3ba4df06b15df81ecf84be81f278294de70cce2dda2b0b`.
Both stdout files contain only the allocated H100 identity. The old runtime
published no draft, receipt, or completed model-review row. Dependency-held
comparison job `770471` was cancelled without execution.

## Bounded software repair

The first-attempt runtime discarded invalid response details at terminal
failure. Later pushed changes do not relax semantic evidence validation. They:

1. preserve and replay every rejected response, prompt hash, token count, and
   exact rejection in an immutable failure artifact;
2. constrain the maximum number of assumed/taught concepts, evidence quotes,
   and defects and explicitly request a smaller exact JSON object;
3. canonicalize model list ordering before applying the unchanged set/order
   contract; and
4. map whitespace-equivalent model quotes back to the unique literal chapter
   substring while retaining the minimum 16-codepoint, exact-span, and
   uniqueness checks.

Repository validation at recovery commit
`78db07e242e78e892889dcda04281e1c5228303b` passed 615 tests plus Black, Ruff,
and every Slurm syntax check. The exact runner source SHA-256 is
`0fc3f07cd78c9daa189d634b5681453f5f7b8ab888985b139ede870dd09ece1b`.
The blind packet and packet-receipt hashes remain respectively
`2d662e9e394cd14ad0d7ce1c8058923c49837310663a7e0b57c2501ff8e34106`
and `41f305a80dd7a2ed26d9fb4b6305f903ee34cfef24de1f6a719bbd6624416697`.
No chapter, concept vocabulary, annotation threshold, model snapshot, or human
qualification rule changed.

## Fresh collision-safe execution

- Qwen3.5-9B review: job `770735`, output root
  `authored-model-review-qwen35-9b-78db07e-r1`
- SmolLM3-3B review: job `770736`, output root
  `authored-model-review-smollm3-3b-78db07e-r1`
- cross-family diagnostic comparison: CPU job `770738`, dependency
  `afterok:770735:770736`

Each reviewer is an independent single-H100, no-requeue request with the known
bad-node exclusions frozen in its job script. The fresh output roots were
absent before submission. The comparison can execute only after both complete
and it remains a disagreement-ranking diagnostic for human review. Model-model
agreement can never satisfy the independent human identity attestations or
authorize a semantic curriculum.

## Preserved second-attempt evidence and narrower normalization

The `78db07e` execution improved observability and demonstrated that the
repair acted on the intended boundary, but it did not complete:

- Qwen job `770735` failed `1:0` after 304 seconds, zero restarts, on `evc42`.
  Its three row-zero responses consistently supplied evidence-backed taught
  concepts while duplicating those same identities in the ungrounded assumed
  list. Its preserved failure artifact SHA-256 is
  `d3d7fea2e47064e4cba499ce6757c7297dd2b72c6f519237b0e81390d44389fa`.
- SmolLM3 job `770736` failed `1:0` after 260 seconds, zero restarts, on
  `evc46`. It published two individually replayable candidate rows before row
  2 repeatedly returned an empty taught set with recommendation `admit`. Its
  preserved failure artifact SHA-256 is
  `df240d7ec91f35e838fb085c120acdc12fcf9fd4dfc4bcae409771b6efd0b1b2`.
- comparison `770738` was cancelled without execution. Neither reviewer
  published a complete draft or result receipt.

These preserved responses support two conservative deterministic rules. An
explicitly quoted, confidence-qualified taught label takes precedence over the
same ungrounded assumed label, so the duplicate is removed only from the
assumed list. A row with no taught concepts is normalized from `admit` to
`revise`; it can never enter an admitted population. Neither rule creates a
concept, quote, confidence score, defect, or positive recommendation.

The narrower normalization was added and pushed at
`a14e6eae69ed592d8744f86e3f1b9648f4f5d387`. Full validation passed 616
tests plus Black, Ruff, and all Slurm syntax checks. Fresh collision-safe
review jobs are Qwen `770761` and SmolLM3 `770762`; CPU diagnostic comparison
`770763` depends on both. All original and second-attempt artifacts remain
immutable and are not reused as successful rows under the new runner identity.

## Preserved third-attempt evidence and unsupported-evidence discard

The `a14e6ea` execution advanced both reviewers through three independently
replayable rows, then stopped on row 3 without publishing a complete review:

- Qwen job `770761` failed `1:0` after 110 seconds, zero restarts, on `evc42`.
  Its preserved row-3 failure artifact SHA-256 is
  `1483701116ef55291f70a010689926db9ecb5edb5faf272deae85c8acf3e4751`.
  One response supplied an ambiguous second quote; two responses repeated the
  top-level instructional-quality value inside each taught-concept object.
- SmolLM3 job `770762` failed `1:0` after 42 seconds, zero restarts, on `evc46`.
  Its preserved row-3 failure artifact SHA-256 is
  `8ae9d8accfc272070e826072417ffa569b18d80cf2ac2ed6107738b38496d84c`.
  Its three responses respectively included a short unsupported token,
  a placeholder sentence absent from the source, or both alongside a literal
  evidence span.
- comparison `770763` was cancelled without execution. Neither reviewer
  published a complete draft or result receipt.

Commit `2aaf0f3384d8e53661ff95c24687875748af5bd9` adds only conservative
normalization supported by these exact failures. A nested instructional-quality
field is removed only when it exactly repeats the row-level value. Evidence
quotes are retained only when they canonicalize to a unique literal source span
meeting the unchanged 16-codepoint minimum; unsupported, ambiguous, duplicate,
or shorter strings are discarded. A taught concept with no surviving evidence
is discarded and an empty-taught `admit` remains downgraded to `revise`.
Unknown fields, invented concepts, confidence thresholds, defect validation,
and human qualification requirements remain unchanged. Full validation passed
617 tests plus Black, Ruff, every Slurm syntax check, and the whitespace check.

Fresh collision-safe diagnostic jobs use sealed runtime
`sai-initiative-runtime-8346674-r1`: Qwen `770785`, SmolLM3 `770786`, and CPU
comparison `770787` with dependency `afterok:770785:770786`. Both GPU jobs are
independent single-H100, no-requeue requests. Their output roots and the
comparison output were absent before submission. They cannot qualify labels or
authorize data; a successful comparison only prioritizes later human review.

## Terminal fourth attempt and end of model-parser iteration

The fourth attempt crossed the prior row-3 boundary but did not complete:

- Qwen job `770785` failed `1:0` after 104 seconds, zero restarts, on `evc42`.
  Four rows are individually replayable. Row 4 exhausted all three attempts on
  unsupported or ambiguous defect quotations. Its failure artifact SHA-256 is
  `ed4e7be97d325df1943c93217fe40630ee3c36f5346de74c23004ace43704dbb`.
  Stdout/stderr SHA-256 values are respectively
  `12da5e6bd425d7147764ca038a82cee7acefb7ab578bc7d82554055ff7bb2d92`
  and `e5827b7bec294a2f89ca727870df72d52ca9db8007b80f0e3496c97b88dc4897`.
- SmolLM3 job `770786` failed `1:0` after 74 seconds, zero restarts, on `evc46`.
  Five rows are individually replayable. Row 5 first exceeded the frozen
  evidence-list limits, then produced two non-JSON responses. Its failure
  artifact SHA-256 is
  `674a4c84b72b5c4fa81fead6f2d088e10b99c823077907e97b42e0ab7cd2ccde`.
  Stdout/stderr SHA-256 values are respectively
  `6409fd7714623b93a7c8342ee94891a03533f60f8392fb3e2e8b5eba795b344f`
  and `1bc85f51d5c4e08a95e758c75d330e733127b11e2c835f74c04f9108c29d1f18`.
- comparison `770787` was cancelled without execution.

This closes the bounded model-parser recovery lane. No fifth model attempt will
be created by relaxing quotation, evidence-count, JSON, confidence, or source
span requirements. The preserved partial rows remain diagnostic-only and
cannot be spliced into a completed review. The authored packet now returns to
the already-frozen independent human-review path; only two complete human
identity attestations can qualify its semantic labels.
