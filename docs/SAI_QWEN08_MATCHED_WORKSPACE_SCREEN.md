# Sai Qwen3.5-0.8B matched-workspace screen

This is the first capable-host Sai architecture test. It does not authorize a
4B run and it cannot become an improvement claim without real benchmark rows.

## Frozen parent and data

- Parent: `Qwen/Qwen3.5-0.8B` revision
  `2fc06364715b967f1860aea9cf38778875588b17`, causal text path only,
  `752,393,024` parameters.
- Parent parameters and buffers remain immutable. Forced-fast inference is the
  unchanged parent, not a bypass through a modified backbone.
- Training source: the separately decontaminated FineWeb-Edu 500M source,
  tokenized with the exact parent tokenizer. The first screen consumes the
  frozen 61,035-sequence prefix at length 2,048.
- Seed: `2026082108` for both arms.

## One isolated factor

Both arms attach the identical `19,938,304`-parameter workspace:

- hidden width 1,024;
- workspace width 512;
- 16 learned slots and 8 heads;
- four shared reactor blocks with intermediate width 2,048;
- two reactor iterations; and
- a zero-initialized reader output projection.

The treatment carries slot state from the first iteration into the second.
The control resets to the same compiled slots for both iterations and averages
the two branches. It therefore executes the same compiler, eight reactor-block
applications, and reader with the same parameters. Both reset branches remain
connected to the loss, matching backward work as well as forward work. The
only intended contrast is recurrent state propagation.

At sequence length 2,047 and two iterations, each probed decision adds exactly
`5,473,566,720` modeled matmul/attention FLOPs under the repository ledger to
both arms, excluding the identical frozen-parent forward.

Each 2,048-token sequence supplies deterministic next-token probes at positions
`255, 511, 767, 1023, 1279, 1535, 1791, 2046`; cross-document targets are masked
identically. The objective is selected-token cross-entropy plus frozen-parent
KL with coefficient `1.0`. AdamW updates only workspace tensors. Every admitted
target, skipped boundary, update, byte prefix, modeled FLOP, peak-memory, and
checkpoint identity must be reported.

## Decision boundary

The unchanged parent, recurrent treatment, and reset-state control use the
same normalized-choice-likelihood evaluator on the complete 12,032-row
MMLU-Pro and 756-row MuSR development populations. Upstream parent-pretraining
contamination is unknown and must remain labeled; the Sai factor-training
source is exactly disjoint from both boards.

The recurrent factor survives only if all conditions hold:

1. its unweighted two-board macro is at least `+0.5` percentage points versus
   both the unchanged parent and reset-state control;
2. neither board is worse than either comparator by more than `0.5` points;
3. the paired 95% lower confidence bound versus reset-state control is positive
   on at least one complete board; and
4. forced-fast logits remain byte-identical to the unchanged parent.

A miss rejects this workspace recurrence. A pass authorizes a larger sub-4B
confirmation only; it does not authorize Sai 4B training.
