# Sai CPU Model Generator Contract

Status: implemented reference mechanics. Training remains unauthorized.

## Purpose

The architecture tournament now has an executable CPU oracle for its three core
families:

- gated grouped-query attention;
- three Gated DeltaNet layers followed by one gated GQA layer; and
- three Kimi Delta Attention layers followed by one gated MLA layer.

The oracle is intentionally slow and readable. Production Triton/CUDA kernels
must reproduce its declared causal and recurrent behavior before they can enter
an official training graph. Passing this oracle does not establish model quality.

## Frozen mechanics

- bias-free projections and zero dropout;
- RMSNorm and PreNorm residual blocks;
- SwiGLU FFNs;
- tied token embedding and output projection;
- head-specific post-attention sigmoid gating;
- partial RoPE in the gated-GQA branch;
- NoPE latent K/V compression in the gated-MLA branch;
- scalar decay for Gated DeltaNet;
- channel-wise decay for KDA; and
- FP32 recurrent state for the CPU delta-rule oracle.

KDA uses the recurrence:

`D_t = diag(alpha_t) S_(t-1)`

`e_t = beta_t (v_t - k_t^T D_t)`

`S_t = D_t + k_t e_t^T`

`o_t = S_t^T q_t`

The tests prove that channel-wise KDA exactly reduces to scalar-decay GDN when
every channel receives the same decay, and that splitting a sequence across a
recurrent-state boundary reproduces the full reference result exactly.

## Parameter matching

The analytical ledger counts every embedding, mixer, FFN, layer norm, and final
norm parameter. It is regression-tested against instantiated PyTorch modules for
all three families. The scale planner changes only FFN intermediate width, in
multiples of 64, to place each family near 100M, 300M, 1B, or 4B total parameters.

The compute ledger uses a declared convention: one multiply-add is two FLOPs and
counts dense projections, depthwise convolutions, recurrent updates, quadratic
attention, FFN matmuls, and tied-output logits at sequence length 2,048. It omits
elementwise nonlinearities, normalization, embedding lookup, and loss. Production
runs must add measured hardware counters; the analytical ledger is the stable
planning denominator, not an MFU claim.

The primary frozen geometry receipt uses 48K vocabulary. Separate receipts can
be deterministically generated for 32K and 64K. Smaller-vocabulary fixed-total
variants receive a larger FFN; tokenizer-only comparisons must instead retain
the 64K body geometry so segmentation and parameter reallocation remain distinct.

Fairness has two separate contrasts. Iso-data holds the ordered UTF-8 byte stream
fixed and reports the resulting FLOP difference. Iso-FLOP holds the compute budget
fixed and allows each model to consume a different-length prefix of the exact same
ordered stream. A report may not claim that both bytes and FLOPs were identical
when the architectures execute different operations.

## Boundaries

This is not a fast kernel, a throughput claim, or a training authorization. The
following remain required before an official 100M run:

1. compare the oracle against the qualified FLA KDA/GDN kernels on Linux/CUDA;
2. implement variable-length packed boundaries without recurrent-state leakage;
3. bind optimizer, data order, UTF-8 byte budget, FLOPs, seeds, and environment;
4. prove production parameter counts match the selected ledger;
5. qualify forward, backward, chunkwise, recurrent, cache, and reset paths; and
6. receive the user's explicit official training order.

Primary implementation references are the official
[Kimi Linear model](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base),
[Kimi Linear report](https://arxiv.org/abs/2510.26692), and
[Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
