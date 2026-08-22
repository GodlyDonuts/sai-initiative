# Sai FLA Semantic Parity v3

V3 is a prospective, dual-lane mechanics gate. It does not change, replace, or
reinterpret any v1 or v2 file or receipt. It executes no optimizer step, makes
no model-quality claim, promotes no architecture, and authorizes no long
training. Its only possible downstream admission is the separately frozen exact
B8 x 2,048 one-update hybrid mechanics canary.

## Frozen population and veto structure

The three nonreplaceable production seeds are `20260827`, `20260828`, and
`20260829`. Each seed produces one create-only, self-hashed receipt. Every seed
runs GDN and KDA separately at lengths 1, 63, 64, and 65. The packed geometry
includes unconditional row resets even when the numeric identity on opposite
sides of a row boundary is the same.

Every family, length, q/k/v path, lane, and tensor is an independent veto. A
failure is never averaged with another result and one family cannot qualify the
other. All three seed receipts are required for a three-seed mechanics claim.

## Lane A: upstream-supported FP32 and FP16 protocol

Lane A is grounded in the convolution forward/backward acceptance protocol at
the pinned FLA upstream commit `ca910f8` for FLA 0.4.2:

- FP32 uses the upstream `swish` spelling for SiLU; FP16 uses no activation,
  matching the supported dtype and activation combinations rather than
  extrapolating the FP32 SiLU claim.
- Bias is absent, packed resets are explicit, and q, k, and v are tested
  independently at every bounded geometry.
- Output, input-gradient, and weight-gradient relative RMSE must each be
  strictly less than `0.001`.
- Every tensor must be finite.

The bounded Sai geometry is not a byte-for-byte replay of the large upstream
test shapes. It uses the exact upstream operation, dtype/activation semantics,
and acceptance rule on the geometry relevant to this mapping. Lane A is
explicitly **not** BF16 qualification.

## Lane B: BF16 noninferiority to Torch

Lane B tests Sai's actual BF16 convolution and SiLU path. Values, weights, and
output gradients are generated on CPU and quantized exactly once to BF16. Exact
clones feed three computations:

1. segmented pure FP64 depthwise causal convolution plus SiLU, including FP64
   autograd, as the mathematical oracle `O`;
2. segmented Torch BF16 convolution plus SiLU; and
3. packed FLA BF16 convolution plus SiLU.

For each output, input gradient, and weight gradient, all values must be finite,
shapes and dtypes must be exact, and all of the following must hold:

`abs(FLA - O) <= abs(Torch - O) + 0.5 * BF16_ULP(O)` elementwise.

`RMS(FLA - O) <= RMS(Torch - O) + RMS(0.5 * BF16_ULP(O))`.

The ordered-BF16 distance from FLA to the ideally rounded oracle may be at most
one representable rounding step worse than Torch for every element. Signed zero
is collapsed and negative/positive subnormals are consecutive around zero. The
half-ULP calculation explicitly covers zero, subnormals, signs, and normal
power-of-two boundaries. Only FP64 machine-arithmetic slack is added to the
envelope; there is no fitted BF16 scalar tolerance.

## Unchanged mapping and recurrence gates

Each case reuses the v2 structural mapping check: exact offsets, depthwise
weight cast and shape, SiLU, external q/k normalization, log-decay/beta
materialization, family flags, final-state suppression, and literal `scale=1`.

The packed recurrence thresholds remain unchanged and strict:

| Tensor | Maximum relative RMSE |
|---|---:|
| output `o` | 0.005 |
| `dq` | 0.007 |
| `dk` | 0.008 |
| `dv` | 0.007 |
| `dg` | 0.015 |
| `dbeta` | 0.015 |

## Claim boundary

A production family status requires real CUDA BF16, exact FLA 0.4.2 from
upstream commit `ca910f8`, a clean FLA source tree, operator source files
resolved inside that pinned tree, and every required check in the receipt. The
receipt records the full Git head, resolved source paths, and source hashes.
Even a clean three-seed result is mechanics evidence only. It is not benchmark
evidence, a training result, architecture promotion, or permission for a long
run. The job file is a single-H100 create-only execution template and contains
no launcher, dependency, cancellation, or self-submission behavior.
