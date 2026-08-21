# Sai Initiative

Sai is Project Shohin's return to its original objective: build the strongest
practical model near four billion parameters. This repository is the live
scratchpad and implementation surface for that effort.

Nothing is called an improvement until it beats the unchanged parent and an
equal-compute control on real, source-disjoint benchmarks.

**Training hold:** no model training, GPU submission, large-weight restoration,
or teacher-trace generation begins until the preparation audit is complete and
the user gives an explicit official training order.

## Current target

- **Name:** Sai
- **Size:** approximately 4B parameters
- **Provisional parent:**
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- **Deployment:** one checkpoint, one pass, one H100 or smaller inference tier
- **Focus:** English, code, math, science, technical reasoning
- **Reasoning:** direct and deliberate behavior in one model; no mandatory
  hidden-draft/revision call

## Why this repository exists

Always-revise Shohin failed its first broad public test. Across HumanEval+,
MBPP+, IFEval, MuSR, and CorrectBench, it scored `42.806%` macro versus
`54.022%` for the original and `49.911%` for an equal-compute control. The
`-33.201 pp` MuSR and `-20.839 pp` CorrectBench regressions close mandatory
revision as a route to general intelligence.

Sai starts from those negative results instead of hiding them.

## Candidate stack

### Behavior-preserving skill learning

Sai v0 trains a narrow adapter on verified, benchmark-decontaminated math,
code, logic, science, technical, and instruction data. Every optimizer window
also replays broad parent behavior. The candidate minimizes task loss plus
frozen-parent token KL. The equal-compute control executes identical forwards
with KL weight zero.

### Deep reasoning without compulsory verbosity

Following the strongest lesson from DeepSeek-R1, small-model reasoning starts
with verified long-form distillation from a stronger teacher, not pure RL from
scratch. Multiple traces are retained only when rule-based math, code, or logic
verification succeeds. Direct-response examples remain in the same mixture.
Only an SFT checkpoint that survives the public gate may enter bounded GRPO.

Long reasoning is a capability, not a ritual. Native inference may select a
direct or deliberate trajectory from the request itself, while fixed-direct
and fixed-deliberate ablations measure whether the selection helps.

### Tokenizer capacity reallocation

The parent tokenizer is the control. A Sai tokenizer candidate preserves all
special tokens, ASCII/bytes, English, code, numeric forms, math/LaTeX, science,
and technical notation, while measuring unused multilingual pieces that could
be removed. Retained rows keep their exact parent embeddings. New merged pieces
are initialized from their parent segmentations and repaired through continued
pretraining. Vocabulary surgery advances only if round-trip behavior,
English/code/math fertility, and public benchmarks pass against the untouched
tokenizer at equal tokens and updates.

## First public gate

The complete official HumanEval+, MBPP+, IFEval, MuSR, and CorrectBench boards
are conjunctive. A candidate must:

- beat original and equal-compute macros by at least `1.0` point;
- remain within `1.0` point of both comparators on every benchmark;
- beat each comparator on at least four of five benchmarks; and
- be nonnegative against both comparators on MuSR and CorrectBench.

One serious regression vetoes a favorable average.

## Build status

- [x] retire always-revise compute and free every GPU request;
- [x] encode the five-board benchmark gate and historical falsification;
- [x] prototype frozen-parent replay KL with a matched zero-weight control;
- [x] begin a lossless tokenizer-capacity auditor;
- [x] implement deterministic, benchmark-disjoint freezing for skill, direct,
  deliberate, replay, and RL-prompt banks;
- [ ] run that freezer on the exact admitted source populations;
- [ ] qualify untouched and reduced tokenizer candidates;
- [ ] package the immutable 4B parent and Sai training runtime;
- [ ] run a low-token SFT/control pilot;
- [ ] run all five complete public boards;
- [ ] promote only if every gate conjunct passes.

## Repository layout

- `src/sai/gates/` — real-benchmark promotion decisions
- `src/sai/data/` — verified role populations and contamination filtering
- `src/sai/training/` — behavior preservation and reasoning training
- `src/sai/tokenizer/` — vocabulary capacity measurement and surgery
- `tests/` — fail-closed regression coverage
- `docs/` — frozen contracts and experimental evidence

Historical Shohin evidence remains in
[`GodlyDonuts/shohin-ettr`](https://github.com/GodlyDonuts/shohin-ettr).
Sai-specific implementation and results live here.
