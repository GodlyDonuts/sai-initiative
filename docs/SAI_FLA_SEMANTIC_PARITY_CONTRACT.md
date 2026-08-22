# Sai FLA Semantic Parity v2

This prospective gate is independent of `sai-full-delta-mixer-fla-parity-v1`.
Existing v1 failures remain failures and are never reclassified by a v2 result.
The v2 scope is the exact Sai-to-FLA 0.4.2 packed mapping and its direct
forward/backward numerical semantics, not model quality or training.

## Frozen qualification population

The prior seeds `20260821`, `20260822`, and `20260823` are calibration evidence
and are prohibited from production v2 receipts. Production receipts accept only
the prospectively frozen seeds `20260824`, `20260825`, and `20260826`. Each seed
runs both GDN and KDA at sequence lengths 1, 63, 64, and 65, including an
unconditional batch-row reset where the numeric segment ID repeats.

Each seed creates a separate receipt. All three receipts are required for a
three-seed claim. There is no averaging across seeds, families, lengths, or
tensors, and a failed seed cannot be replaced.

## Exact per-tensor gates

For reference tensor `r` and FLA tensor `f`, the error ratio is
`RMS(f-r) / (RMS(r) + 1e-8)`. Every value must be finite and every comparison uses
strict less-than:

| Direct operation | Tensor | Maximum ratio |
|---|---|---:|
| packed causal convolution | output `y` | 0.001 |
| packed causal convolution | input gradient `dx` | 0.001 |
| packed causal convolution | weight gradient `dw` | 0.001 |
| packed recurrence | output `o` | 0.005 |
| packed recurrence | `dq` | 0.007 |
| packed recurrence | `dk` | 0.008 |
| packed recurrence | `dv` | 0.007 |
| packed recurrence | `dg` | 0.015 |
| packed recurrence | `dbeta` | 0.015 |

The convolution gate runs independently for the q, k, and v paths. A primitive
failure vetoes that family and cannot be rescued by a favorable aggregate.

## Structural invariants

Every case also verifies exact packed offsets, an unconditional row boundary,
depthwise convolution weight casting and shape, SiLU activation, normalized q/k
materialization, log-decay and beta materialization, `scale=1`, external q/k
normalization, final-state suppression, and the exact family-specific FLA flags.

GDN and KDA receive separate statuses. One family may pass its test mechanics
without qualifying or concealing failure in the other. A production status
requires real CUDA BF16, exact FLA 0.4.2, and every case for that family.

## Claim boundary

- No optimizer or parameter update is executed.
- No training GPU job is submitted by the module or job file.
- A pass is not public-benchmark evidence, architecture promotion, or training
  authorization.
- These bounded lengths do not replace the exact B8 x 2,048 mechanics canary.
- The v2 job is a single-H100 execution template with no launcher, dependency,
  retry, cancellation, or self-submission behavior.
