# Sai Adaptive-Compute Falsification Plan

Status: prospective research contract. No architecture is selected and no
training is authorized.

## Breakthrough thesis

Sai's highest-upside hypothesis is not another always-on sequence mixer. It is
a baseline-preserving language path plus a conditional internal computer:

- a private, query-conditioned latent workspace;
- recurrent updates whose dynamics are trained to settle;
- optional workspace-mediated sparse semantic memory; and
- a controller trained on measured slow-path benefit per added FLOP.

This is a hypothesis, not the chosen Sai architecture. The prior Shohin result
showed why an interesting mechanism cannot be promoted from narrow gains: severe
MuSR and CorrectBench regressions outweighed its coding improvements. The new
system must therefore prove positive conditional value on real benchmarks before
the controller, semantic memory, or 4B integration is allowed to matter.

Recent primary evidence makes the ingredients worth testing. Gated DeltaNet-2
reported strong 1.3B/100B-token controlled results, especially on multi-key
retrieval. Memory Layers at Scale reported gains over higher-compute dense and
matched MoE models, particularly on factual tasks. A 2026 fixed-point study
showed that settling dynamics can make added recurrent depth safe on controlled
algorithmic tasks. None of those results proves that a combined latent workspace
improves broad language capability.

## Unchanged primary screen

The existing 100M tournament remains exactly three families and 18 runs:

1. gated GQA;
2. the 3:1 Gated DeltaNet/GQA hybrid; and
3. the 3:1 KDA/gated-MLA hybrid.

The adaptive-compute thesis is not a fourth primary arm. It begins only after a
base mixer wins the frozen iso-data and iso-FLOP comparison. The 100M → 300M →
1B → 4B scale ladder, source-disjoint data rules, three seeds, public benchmark
boundary, and requirement for the user's explicit training order remain intact.

## Three claims that must survive

### H1 — positive oracle slow-path value

For a fixed selected backbone, a small isolated slow path must improve a
measurable subset of source-disjoint examples. Per-row oracle selection between
fast and slow must beat both the fast model and an equal-inference-FLOP fast-path
control. If oracle selection cannot win, no learned router can rescue the idea.

### H2 — useful and depth-safe recurrence

At recurrence horizons 1, 2, 4, and 8, official correctness must improve or
plateau rather than unravel. Held-out computational depths are also evaluated at
16 iterations. Fixed-point residual, state norm, update norm, and decoder margin
are diagnostics; convergence alone never counts as correctness.

### H3 — learnable compute allocation

An out-of-fold controller must predict paired slow-minus-fast capability gain net
of measured FLOPs. It must recover most of the oracle benefit under a fixed
compute budget without routing primarily by prompt length, benchmark name, or
other superficial domain cues.

Sparse memory, typed side channels, and learned exact anchors are independent
secondary hypotheses. They do not enter the full system unless they win their
own matched contrasts.

## Ordered kill ladder

### Gate 0 — no-training feasibility

Before optimizer work, publish exact parameter, forward/backward FLOP,
activation-memory, and inference-cost ledgers for the 16-slot workspace. Measure
candidate tokenizer fertility on the frozen corpora. Microbenchmark workspace
and retrieval kernels with full memory traffic included.

Kill any design that exceeds its parameter/activation budget, loses its claimed
efficiency after memory movement, or materially expands important token domains.

### Gate 1 — isolated workspace and oracle routing

At the 300M factor-screen stage, compare one selected base checkpoint under:

1. forced fast path;
2. a zero-output-initialized 16-slot slow path;
3. an equal-inference-FLOP larger fast-path control; and
4. per-row oracle choice between the frozen fast and slow outcomes.

The bypass must be bitwise identical to the fast path at initialization. Shared
backbone drift is not called a regression firewall: a preserved fast checkpoint
and forced-fast evaluation remain mandatory.

Kill the workspace if paired oracle selection has no positive 95% confidence
bound against both controls on the frozen source-disjoint macro, or if any
retention benchmark crosses the declared serious-regression boundary.

### Gate 2 — recurrence causality

Starting from the same checkpoint, compare terminal-answer training alone with
terminal answer plus fixed-point training. Evaluate horizons 1/2/4/8 and held-out
16 against visible chain-of-thought and extra fast-path compute at matched
inference FLOPs.

Kill the recurrent thesis if added iterations reduce official correctness, if
gains disappear under FLOP matching, or if verifier/fixed-point scores rise while
official answers do not.

### Gate 3 — regret controller

Generate fast and slow outcomes only on the frozen label-building split. Define
regret from paired official capability outcomes and measured compute, not answer
likelihood alone. Train and calibrate the gate out of fold, then freeze it before
the source-disjoint evaluation split is opened.

Report oracle gain, captured oracle gain, AUROC, calibration, precision at each
compute budget, selective capability-versus-FLOP curves, and false-positive
regression rate. Kill the controller if compute-adjusted utility is non-positive
or if it cannot preserve the forced-fast retention envelope.

### Gate 4 — sparse semantic memory

Only after H1–H3 survive, compare workspace-only trainable sparse memory with:

- no memory;
- an equal-total-parameter dense FFN reinvestment; and
- an equal-active-FLOP alternative.

Bind retrieval identities and report utility, collision rate, load balance, HBM
traffic, bandwidth, latency, factual capability, reasoning retention, and exact
data provenance. Kill it if gains are not credible on broad benchmarks, if they
depend on contaminated memory, or if realized throughput erases the active-FLOP
advantage.

### Gate 5 — orthogonal representation and anchor ablations

Typed numeric/identifier/structure side channels keep the same surface token IDs
and body geometry. Test numeric, identifier, unit, and structural channels
separately before bundling. Learned exact anchors are compared against the
selected periodic exact-attention system independently of workspace and memory.

Parser coverage, parse failures, copying, Unicode round trips, sequence length,
and general-English retention are first-class outputs. These mechanisms are
removed if they cannot win alone.

### Gate 6 — integration and replication

Combine only independently surviving components and repeat every constituent
ablation. Promote beyond 300M only when the integrated system beats its strongest
single component. Replicate at two seeds and a second backbone family before 4B.

## Benchmark and accounting boundary

Development uses frozen, source-disjoint real benchmarks with official or
executable row-level scoring. At minimum it must cover the failure domains that
falsified Shohin: code, instruction following, multi-step reasoning, and
self-correction. The complete HumanEval+, MBPP+, IFEval, MuSR, and CorrectBench
boards remain a declared terminal public boundary rather than an iterative
training signal.

Every comparison binds identical decoding where applicable, paired row
identities, confidence intervals, per-domain retention, total and active
parameters, training and inference FLOPs, wall time, energy when available, peak
memory, and useful work per GPU-second. A favorable average never overrides a
serious single-benchmark regression.

## Explicit non-claims

- Zero initialization preserves the fast path only at initialization.
- Lower slow-path likelihood is not proof of higher capability.
- A fixed point may be stably wrong.
- A learned verifier is not an official scorer.
- Sparse parameters are not cheap if memory traffic dominates.
- Tokenizer parameter savings are not gains if sequence expansion consumes them.
- The full tri-memory model is not authorized merely because its ingredients are
  individually plausible.

## Primary references

- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
- [Think Shallow, Solve Deep](https://arxiv.org/abs/2608.18222)
- [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)

These sources motivate experiments. Sai promotes only from its own matched,
source-disjoint benchmark evidence.
