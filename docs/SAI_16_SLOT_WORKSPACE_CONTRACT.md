# Sai 16-Slot Workspace Mechanics Contract

Status: implemented no-training CPU reference and analytical planner. No
backbone family is selected and no optimizer work is authorized.

## Scope

This is Gate 0 for Sai's adaptive-compute thesis. It adds exactly one factor to
each already-frozen 300M mixer geometry: a private 16-slot latent workspace. It
does not change the three-family/18-run 100M screen, shrink or reallocate the
base, add a controller, enable sparse memory, add typed side channels, introduce
learned anchors, or train a fixed-point objective.

The machine-readable plan is
[`SAI_16_SLOT_WORKSPACE_MECHANICS.json`](SAI_16_SLOT_WORKSPACE_MECHANICS.json).
It is reproduced by:

```text
sai-workspace-plan \
  --geometry-plan docs/SAI_48K_SCALE_GEOMETRIES.json
```

The planner always reports a training hold, zero GPU jobs, zero updates, no
selected family, and an unchanged primary screen.

## Frozen reference geometry

All 300M candidates have hidden width 768. The workspace uses:

- 16 learned slots;
- width 384;
- six heads of width 64;
- four distinct reactor blocks;
- SwiGLU width 1,536 inside each block; and
- recurrence horizons 1, 2, 4, 8, and 16 using the same four-block reactor.

The compiler uses the slots to cross-attend to the current prompt. For packed
mechanics, it can see only the final contiguous document segment. Each reactor
block receives the immutable compiled slots again, preventing the recurrent
state from silently forgetting the request. Workspace attention is
bidirectional. The reader changes only the final next-token position.

The reader output matrix is initialized to exact zero. Forced-fast mode calls
the base model directly and executes no workspace operation. At initialization,
forced slow executes the complete workspace but its logits are bitwise identical
to forced fast. This is an initialization invariant—not a claim that later joint
training cannot alter the fast path.

## Exact parameter ledger

The bias-free reference contains:

| Component | Parameters |
| --- | ---: |
| Learned slots | 6,144 |
| Compiler, including norms | 885,888 |
| One reactor block | 2,360,064 |
| Four-block reactor | 9,440,256 |
| Reader, including zero output and norms | 885,888 |
| **Workspace total** | **11,218,176** |

No learned regret controller is hidden in this count. The unmodified 300M bases
therefore become:

| Base family | Base parameters | Base + workspace |
| --- | ---: | ---: |
| gated GQA | 298,786,560 | 310,004,736 |
| Gated DeltaNet hybrid | 299,283,072 | 310,501,248 |
| KDA/MLA hybrid | 298,246,872 | 309,465,048 |

The base is not reduced to keep the combined system near 300M, because doing so
would destroy the exact forced-fast control.

## FLOP convention

For one next-token decision with a 2,048-token prompt, one multiply-add counts as
two FLOPs. Matmul and attention operations are included; normalization, softmax,
nonlinearities, and the selected base model are excluded and must be reported by
production profiling.

| Reactor iterations | Added forced-slow forward FLOPs |
| ---: | ---: |
| 1 | 2,789,892,096 |
| 2 | 3,093,454,848 |
| 4 | 3,700,580,352 |
| 8 | 4,914,831,360 |
| 16 | 7,343,333,376 |

The compiler costs 2,475,687,936 FLOPs, each full reactor iteration costs
303,562,752, and the reader costs 10,641,408. These figures describe one
decision; multiplying them by generated tokens without accounting for caching
or changing prefix length would be invalid.

The activation ledger is explicitly limited to analytical incremental tensor
geometry. At length 2,048 it reports a largest stage of 1,781,760
elements, or 3,563,520 bytes at BF16, and a persistent 6,144-element slot state.
It is not a framework allocator peak. Exact CUDA peak, memory traffic, wall time,
and useful work per GPU-second remain required before training readiness.

## Mechanics invariants

The test suite proves:

- analytical workspace parameters equal the instantiated module;
- forced fast is a bitwise direct-base bypass;
- zero-initialized forced slow is bitwise fast while internal slots update;
- enabling the reader can change only the last position;
- packed documents cannot exchange workspace information;
- parameter count is recurrence-invariant and FLOPs rise by the exact per-step
  amount;
- invalid dimensions, modes, horizons, plan mutations, and no-training-field
  mutations fail closed; and
- the checked-in JSON exactly replays the existing 48K geometry plan.

Passing mechanics does not show capability improvement. Only the separate
oracle contract can establish whether the slow path has positive value.
