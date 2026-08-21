# Sai 100M Experiment-Plan Contract

Status: implemented prospective planner. Training remains unauthorized.

## Purpose

The 100M mixer screen must distinguish architecture quality from differences in
data exposure or compute. `sai-experiment-plan` therefore binds six immutable
inputs before it emits any run identity:

1. the architecture-tournament contract;
2. the exact 48K model-geometry plan;
3. a qualified 48K tokenizer receipt;
4. one benchmark-disjoint ordered token-stream receipt;
5. a pinned training-environment receipt; and
6. [`SAI_100M_TOURNAMENT_TEMPLATE.json`](SAI_100M_TOURNAMENT_TEMPLATE.json).

The plan is prospective. It always records zero submitted GPU jobs, zero
training updates, and `training_authorized=false`. Building or validating it is
not the user's official training order.

## Fixed primary screen

The primary screen contains gated GQA, the 3:1 Gated DeltaNet hybrid, and the
3:1 KDA/gated-MLA hybrid. Every family receives seeds `20260821`, `20260822`,
and `20260823`. All use causal next-token prediction, 2,048-token packed
sequences, a 524,288-token full update, AdamW, BF16 parameters/activations,
FP32 optimizer and recurrent state, and an identical warmup/cosine schedule.

The iso-data arm consumes exactly 1,048,576 packed sequences, or a capacity of
2,147,483,648 valid tokens, from one ordered stream. The stream receipt supplies
the cumulative admitted UTF-8 bytes at every selected prefix, so equal bytes are
verified rather than inferred from token counts.

## Exact iso-FLOP arithmetic

Whole packed sequences are the smallest executable compute unit. Rounding each
family to a convenient number of optimizer updates would make an “equal FLOP”
claim false. The planner instead computes the least common multiple of the
three analytical forward-plus-backward FLOP quanta and chooses the largest
common multiple no larger than the cheapest iso-data arm.

For the frozen 48K/100M geometries, the exact result is:

| Family | FLOPs per sequence | Iso-data FLOPs | Exact iso-FLOP sequences |
| --- | ---: | ---: | ---: |
| gated GQA | 1,543,772,307,456 | 1,618,762,591,062,982,656 | 678,678 |
| GDN hybrid | 1,319,091,830,784 | 1,383,168,035,556,163,584 | 794,277 |
| KDA/MLA hybrid | 1,313,807,007,744 | 1,377,626,496,952,172,544 | 797,472 |

Every iso-FLOP run executes exactly
`1,047,724,302,079,623,168` modeled training FLOPs. A final partial global batch
is retained when necessary; discarding it would break equality. Hardware
counters, wall time, energy, and useful bytes/GPU-second remain measured outputs
and are never substituted for the analytical matching denominator.

## Required upstream receipts

The tokenizer receipt must prove a 48K vocabulary, byte fallback, exact
round-trip behavior, and preserved special tokens. The ordered-stream receipt
must bind tokenizer identity, source-manifest identity, benchmark disjointness,
cross-document target masking, total geometry, and cumulative UTF-8 bytes for
all requested prefixes. The implemented binary format and replay rules are in
[`SAI_ORDERED_TOKEN_STREAM_CONTRACT.md`](SAI_ORDERED_TOKEN_STREAM_CONTRACT.md).
The environment receipt binds exact Python, Torch, CUDA, and Triton versions.

The planner reopens and revalidates every input whenever its output is checked.
Missing, mutated, reordered, shortened, mismatched-tokenizer, or prematurely
authorized evidence fails closed.

## Still required before training

- Build and qualify the actual 48K tokenizer.
- Freeze the real benchmark-disjoint ordered stream and its prefix-byte ledger.
- Capture the exact qualified CUDA environment.
- Prove the production kernels match the CPU oracle.
- Receive the user's explicit official training order.

No architecture is promoted from this plan. It only makes the future comparison
honest and exactly replayable.
