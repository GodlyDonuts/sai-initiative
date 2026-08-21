# Sai Completed-Run Lineage Contract

Status: implemented replay validator and adversarial fixtures. No real training
receipt exists, and this contract does not authorize training.

## Why this exists

A checkpoint hash alone cannot establish what was trained, whether retries
duplicated updates, whether the promised data and compute budgets were honored,
or whether a reported fast path is still the preserved baseline. Every future
Sai capability result must therefore resolve to a
`sai-completed-run-lineage-receipt-v1` bundle before it is admissible.

The prospective experiment plan remains immutable and says training is on hold.
After the user gives an official training order, a separate
`sai-training-authorization-receipt-v1` must name the exact plan artifact and
authorized run identities. A completed receipt cannot rewrite history by
changing the plan's authorization fields.

## Portable replay bundle

`sai.training.lineage` reopens, rather than merely trusts:

- the frozen 300M adaptive experiment plan and the exact planned run;
- the separate official-order authorization receipt;
- an exact relative-path checkpoint tree with full membership, byte counts,
  member hashes, and tree hash;
- immutable architecture, geometry, tokenizer, stream, environment, source,
  runtime, kernel, configuration, workspace-plan, and candidate identities;
- the parent completed run, parent checkpoint, and parent fast-state projection;
- every successful allocation's host/GPU/log/accounting identities and a
  contiguous half-open committed-update range; and
- exact optimizer-step, sequence, valid-token, admitted-byte, modeled-FLOP,
  measured-GPU-second, and update-ledger evidence.

The validator rejects absolute or parent-traversing artifact paths, symlinks,
hard-linked files, special files, missing or extra checkpoint members, requeued
allocations, update gaps/overlap, budget drift, skipped/nonfinite/overflow
updates, unauthorized runs, and any self-hash or artifact mutation. A failed
infrastructure allocation may be preserved only as an explicit zero-committed-
update attempt; the final attempt must succeed and the committed ranges must
still cover the plan exactly once.

## Frozen fast-path meaning

For the first workspace gate, the parent backbone is frozen. The treatment may
change only the latent workspace. Its canonical fast-state tensor projection
must therefore equal the parent's fast-state projection exactly. Forced-fast
evaluation is a true retained baseline inside the same system, not a bypass
around a backbone that training changed.

The equal-FLOP fast control begins from the same parent and uses the same seed,
ordered stream, and exact training budget, but changes the declared
`fast_path_capacity` factor. Treatment and control have different checkpoints
and roles inside one comparison group. Joint backbone/workspace training is a
different scientific hypothesis and requires a new prospective contract.

## Current boundary

The existing 16-slot JSON is a mechanics plan, not a training plan. Real lineage
cannot be produced until the unchanged 100M tournament selects one backbone and
a path-independent `sai-300m-adaptive-experiment-plan-v1` freezes the treatment
and equal-FLOP control runs. Synthetic tests prove only that falsified lineage is
rejected; they are not model evidence.
