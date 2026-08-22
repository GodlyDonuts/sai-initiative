# Sai authored-review model diagnostic — 2026-08-22

## Disposition

The offline model reviewers are **not qualified annotators**. Their outputs are
candidate triage material only and cannot admit, order, exclude, or otherwise
change training data. The mandatory boundary remains two distinct blind human
review identities followed by exact-quote compilation and disagreement review.

No accepted review row, compiled population, training authorization, or 4B
authorization was produced by the runs below. All six allocations ended before
row 0 was accepted. The live matched curriculum-versus-order-control experiment
was not changed, interrupted, duplicated, or made dependent on these runs.

## Exact executions

All jobs used one H100 on `evc46`, exited `1:0`, and had zero restarts.

| Job | Reviewer | Runtime commit | Seconds | Preserved failure SHA-256 |
|---|---|---:|---:|---|
| 770603 | Qwen3.5-9B | `e5fa3b3b2ac8b5b7b62c9a5c57739649f593b787` | 277 | `61605e1fb655332667eb492ca7a5ec9e27037d16727dc6005578ba82e5f9d983` |
| 770604 | SmolLM3-3B | `e5fa3b3b2ac8b5b7b62c9a5c57739649f593b787` | 43 | `d578f96b8d0ac5bbc587e8c511824c1c91a0f5e6f33bf774056a11ae0abb13ef` |
| 770606 | Qwen3.5-9B | `ec0005e65787a26017d006e7fb565d5e37f66196` | 86 | `efc00a2674d9052d3fd84c6a42acfe457588ea44a13b3bdef36e6995f27f5041` |
| 770607 | SmolLM3-3B | `ec0005e65787a26017d006e7fb565d5e37f66196` | 33 | `f849204a5ca932cd1c6650d0fa54dd73e49219875f8e878c7b87016f0a2b84a5` |
| 770611 | Qwen3.5-9B | `0cf7a165cbbba9f8384a011e2ec0e243668b6112` | 85 | `94b0f616f89f317f23fbd07a2d76e978effc9e01d8b8c391aa77c6873b17a4a6` |
| 770629 | Qwen3.5-9B | `94903fe7d2e79c8df3050faa87b4b999a73bb7bd` | 104 | `d3d7fea2e47064e4cba499ce6757c7297dd2b72c6f519237b0e81390d44389fa` |

The files are create-only, mode `0400`, self-hashed failure receipts containing
the exact prompt hash, input/output token counts, generated response, response
hash, compiler rejection, source-text hash, and blind review identity for all
three frozen attempts. No failed response is silently discarded.

## Findings

1. The original Qwen prompt produced a repeated concept list until the exact
   2,048-token output ceiling truncated the JSON. The repaired runner now
   reports output-budget exhaustion explicitly and bounds review breadth.
2. SmolLM3 initially paraphrased chapter text and later appended commentary or
   recommended `admit` with no taught concept. Exact quotes and admission logic
   correctly rejected every attempt.
3. Qwen then produced compact JSON, but returned semantically unordered lists.
   List ordering is now canonicalized mechanically; duplicates and invalid IDs
   remain rejected.
4. Qwen collapsed a source line break inside an otherwise verbatim quote. A
   whitespace-only candidate may now map back to one unique exact source span.
   Paraphrases, missing spans, and ambiguous spans remain rejected.
5. After those repairs, Qwen continued assigning the same concepts as both
   taught and assumed despite receiving an exact overlap diagnostic. That is a
   semantic contradiction, not a formatting defect, and remains a hard fail.

## Data-first conclusion

These executions demonstrate why upstream model confidence cannot substitute
for source truth or independent human judgment. The tooling repairs improve
evidence preservation and remove only semantics-free formatting variance; they
do not convert either reviewer into an admissible label source. Automated
review is stopped at this boundary. The next semantic-curriculum action is
human blind review of the frozen 127-row packet, followed by exact compilation,
agreement measurement, disagreement adjudication, and a matched same-record
order experiment. No architecture or scale decision may bypass that sequence.
