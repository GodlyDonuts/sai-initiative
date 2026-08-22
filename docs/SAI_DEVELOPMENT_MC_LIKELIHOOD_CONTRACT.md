# Sai source-disjoint development MC likelihood contract

This evaluator is a cheap, paired development screen. It is **not** an official
MMLU-Pro or MuSR result, a public-terminal benchmark, or sufficient evidence to
promote an architecture. Its result JSON sets all three claims to `false`.

The input must be immutable UTF-8 JSONL with no blank lines. Every row must use
exactly one of these schemas; aliases and additional upstream metadata fail
closed and must be transformed by a separately audited converter:

```json
{"benchmark":"mmlu_pro","row_id":"...","domain":"...","question":"...","choices":["...","..."],"answer_index":0}
{"benchmark":"musr","row_id":"...","domain":"...","context":"...","question":"...","choices":["...","..."],"answer_index":0}
```

The caller freezes the expected row count and SHA-256 of the canonical ordered
row-ID list. Row IDs must be unique. It must also supply an immutable
`sai-development-mc-source-disjoint-v1` receipt whose benchmark-source hash and
training-source hash match the run and whose method is exactly
`identity-and-contamination-audit`. A boolean assertion without this bound
receipt is not accepted.

## Frozen Shohin population conversion

`sai-build-development-mc` converts the existing paired Shohin
`shohin-dense-public-benchmark-question-v1` and
`shohin-dense-public-benchmark-assessor-v1` JSONL artifacts. It accepts only
MMLU-Pro and MuSR, requires caller-pinned hashes, row count, and ordered-ID
hash, and recomputes the original Shohin row identity and normalized prompt
hash before parsing. MMLU-Pro uses the final question and sequential lettered
options in its frozen five-shot prompt; MuSR uses its exact domain-specific
hint placement, final question, and sequential numbered choices. Assessor
letters and one-based MuSR answers become the canonical zero-based index.

The supplied `sai-decontamination-receipt-v1` must itself be hash-pinned,
internally valid, and name both exact question and assessor inputs as ordered
decontamination boundaries. Its admitted training output must still match the
recorded path, byte count, and SHA-256. The converter emits both the strict
seven-field `sai-development-mc-source-disjoint-v1` receipt consumed by this
evaluator and a full `sai-development-mc-population-conversion-v1` audit that
records input hashes, matched boundary entries, parser contract, exact row
coverage/order, training evidence, and deterministic output hashes. Outputs
are create-only, staged and fsynced before per-file atomic publication, with
rollback on an ordinary publication failure.

Example (all hashes and counts are mandatory frozen inputs):

```bash
sai-build-development-mc \
  --benchmark mmlu_pro \
  --questions /absolute/full.questions.jsonl \
  --assessors /absolute/full.assessors.jsonl \
  --expected-questions-sha256 "$QUESTIONS_SHA256" \
  --expected-assessors-sha256 "$ASSESSORS_SHA256" \
  --expected-rows 12032 \
  --expected-identity-order-sha256 "$IDENTITY_ORDER_SHA256" \
  --training-decontamination-receipt /absolute/decontamination.receipt.json \
  --expected-training-decontamination-receipt-sha256 "$DECONTAMINATION_FILE_SHA256" \
  --output-source /absolute/mmlu_pro.development.jsonl \
  --output-disjoint-receipt /absolute/mmlu_pro.disjoint.json \
  --output-conversion-receipt /absolute/mmlu_pro.conversion.json
```

For every choice, the evaluator tokenizes the common prompt and the prompt plus
`" " + choice` with special tokens disabled. The prompt token IDs must be an
exact prefix of the combined token IDs. It sums the continuation-token natural
log probabilities and divides by the continuation token count. The highest
normalized value wins, with the lowest index winning an exact tie. This prompt
and score differ from the official benchmark contracts, which is why the
result must remain development-only. The separate decoding contract records
that this is teacher-forced likelihood with no generation, sampling,
temperature, or generated-token budget.

Every result binds hashes for the checkpoint bundle, config bundle, tokenizer
bundle, evaluator code, additional runtime files and environment, source bytes,
ordered row identities, training source, disjointness receipt, and scoring contract. It preserves
per-choice raw and normalized likelihoods, per-row decisions, aggregate
accuracy, and domain accuracy. Evaluation runs under inference mode, restores
the model's prior train/eval state, and rejects parameter or buffer mutation.
