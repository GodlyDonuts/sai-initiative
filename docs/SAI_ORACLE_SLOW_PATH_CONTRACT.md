# Sai Oracle Slow-Path Evaluation Contract

Status: implemented offline analyzer with synthetic and adversarial tests. No
real slow-path result exists and no training is authorized.

## Purpose

Before training a regret controller, Sai must prove that any useful routing
policy exists. `sai-oracle-slow-path` computes the unattainable per-row oracle
ceiling from three independently produced source-disjoint development manifests:

1. forced fast;
2. forced slow at one frozen recurrence horizon; and
3. an independent equal-inference-FLOP fast-path control.

The analyzer never calls a model, scorer, benchmark, or trainer. It only validates
sealed completed rows and writes one deterministic receipt. The five terminal
public boards remain unopened during development.

## Frozen development slots

Exactly one benchmark occupies each failure-domain slot:

- primary code;
- secondary code;
- instruction following;
- multi-step reasoning; and
- self-correction.

The actual benchmark names and versions must be frozen before outputs exist.
Every manifest binds benchmark source, ordered row identities, prompt contract,
decoding, official scorer, environment, system checkpoint, fast-path checkpoint,
system configuration, and completed-run receipt by SHA-256.

Each row binds its identity, visible prompt, output, official score, score weight,
modeled and executed inference FLOPs, output tokens, and infrastructure status.
Slow rows additionally bind the workspace plan, workspace candidate, recurrence
horizon, final update RMS, and output-delta RMS. Missing, duplicated, unscored,
or infrastructure-failed rows abort analysis; they never become incorrect
scientific answers.

## Pairing and compute equality

Fast and slow must be two modes of the same checkpoint, configuration, completed
run, and preserved fast path. The equal-FLOP control must be an independent
checkpoint. Within each benchmark, all three modes must have identical row order,
prompt bytes, weights, scorer, decoding, and environment.

For every row, both modeled and executed slow FLOPs must exactly equal the
control. Fast must cost strictly less. Wall time and hardware counters remain
diagnostics and cannot replace FLOP equality.

The oracle routes slow only when its official row score is strictly greater than
fast; ties remain fast. The primary control follows that identical route mask,
using the equal-FLOP fast control where the oracle used slow and ordinary fast
otherwise. This makes oracle and mask-matched control exactly equal in compute.
The full-budget forced control is also reported as a conservative comparison.

## Decision

Scores are weighted within each development benchmark and macro-averaged across
the five slots. A deterministic benchmark-stratified paired bootstrap, seeded
from all 15 manifest identities, produces 95% intervals for oracle minus fast and
oracle minus mask-matched control.

Support requires every conjunct:

- both paired 95% lower bounds are positive;
- oracle macro is at least one point above fast and mask-matched control;
- no slot regresses by more than one point against either;
- oracle beats both on at least four of five slots;
- multi-step reasoning and self-correction are nonnegative against both; and
- exact modeled and executed compute matching holds.

A pass sets only `next_falsification_gate_authorized=true`. It keeps
`architecture_locked=false` and `training_authorized=false`. It does not prove a
learned controller can recover the oracle value.

## Remaining real-evidence prerequisites

The analyzer is ready, but real manifests cannot exist yet. Before Gate 1 can be
run, Sai still needs:

- a winning mixer from the unchanged primary screen;
- a frozen source-disjoint five-slot development boundary;
- a completed-run/checkpoint lineage receipt;
- a trained isolated workspace and equal-FLOP control, after the user's official
  training order; and
- official row-level outputs for all three modes.

Synthetic success tests demonstrate analyzer behavior only. They are not Sai
capability evidence.
