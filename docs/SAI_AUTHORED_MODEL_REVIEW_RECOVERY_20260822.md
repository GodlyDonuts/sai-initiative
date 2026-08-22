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
