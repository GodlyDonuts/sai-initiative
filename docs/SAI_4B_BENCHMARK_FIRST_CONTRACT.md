# Sai 4B Benchmark-First Contract

Status: prospective. This document authorizes no GPU work by itself.

## Parent and changed factor

The provisional parent is
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Sai v0 changes
only a narrow adapter trained on a frozen skill population and a frozen replay
population. The primary candidate adds frozen-parent token KL on replay. Its
equal-compute control runs the same adapter geometry, skill rows, replay rows,
forward passes, tokens, updates, seeds, and optimizer with KL weight zero.

Both checkpoints remain single-pass causal language models. Neither receives a
draft, verifier result, benchmark label, or second model call at inference.

## Reasoning curriculum

The first SFT stage combines verified cold-start reasoning traces with direct
answers and broad parent-behavior replay. Long traces are generated in groups
by a stronger teacher and retained only after rule-based answer or execution
verification. The model is trained to emit a final answer even after long
deliberation and to use a short path when extended reasoning is unnecessary.

An RL stage is not automatic. If SFT passes the public gate, a bounded GRPO
candidate may optimize rule-verifiable math, code, and logic outcomes. It must
retain the same replay objective and face a matched SFT-only control.

## Tokenizer candidate

The untouched parent tokenizer is the primary control. A reduced tokenizer may
remove only pieces that are unused across every admitted training and public
evaluation prompt and decode entirely to one unsupported script. Special
tokens, byte fallback, ASCII, English/Latin, code, numbers, Greek and math
symbols, LaTeX, and mixed-script pieces are protected.

This initial rule produces a conservative capacity estimate, not a tokenizer.
Any built candidate must prove lossless round-trip, exact retained embedding
rows, initialized replacement rows, corpus fertility, and matched continued
pretraining before a GPU benchmark comparison.

## Public decision

The complete official HumanEval+, MBPP+, IFEval, MuSR, and CorrectBench boards
must use identical model-visible prompts and decoding across the unchanged
parent, equal-compute control, and candidate. Each score binds benchmark source,
ordered identities, prompt/decoding contracts, and checkpoint hashes.

Promotion requires every condition:

1. candidate macro is at least `1.0` point above the original;
2. candidate macro is at least `1.0` point above equal compute;
3. no benchmark is more than `1.0` point below either comparator;
4. candidate beats each comparator on at least four of five benchmarks;
5. MuSR is nonnegative against both comparators; and
6. CorrectBench is nonnegative against both comparators.

A pass authorizes broader confirmation. It does not lock the architecture or
authorize a release claim.

## Historical falsification

The predecessor always-revise system scored `42.806%` macro, versus `54.022%`
for the original and `49.911%` for equal compute. MuSR regressed `33.201` points
and CorrectBench `20.839` points against the original. The executable gate must
reject those exact results, which is covered by the test suite.
