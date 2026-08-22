# Sai 300M/1B Generic Training Preparation

The prepared entry point is `sai.training.scale_training`; the corresponding
single-GPU Slurm template is `jobs/sai-sub4b-scale-training-single-h100.sbatch`.
Neither artifact launches work, selects a mixer family, or contains a default
family. The only admitted scales are `300m` and `1b`.

## Fail-closed admission

Every run must supply all of the following as explicit immutable inputs:

- one exact row from `SAI_48K_SCALE_GEOMETRIES.json`, whose checked parameter
  ledger is within one percent of the named target;
- a `sai-sub-4b-scale-promotion-v1` receipt naming the exact family and scale,
  binding the prior-scale evidence receipt, and attesting a source-disjoint
  real-development-benchmark win against a matched equal-compute control;
- exact training and development stream identities using 2,048-token packing;
- all optimizer, batch, update, seed, checkpoint, and development budgets, with
  no scientific hyperparameter defaults in the generic CLI;
- the exact clean git commit and geometry-file hash;
- the qualified FLA 0.4.2 kernel receipt; and
- the production CUDA/BF16 full-DeltaMixer parity report covering GDN and KDA.

The promotion receipt records an external development-evidence decision. This runner checks
its exact schema, hash, scale, family, and gate assertions, but does not recompute
the underlying development benchmarks. The evidence receipt it names must therefore
remain available for independent audit.

The run receipt binds the promotion receipt, code, environment receipts, model
configuration, streams, optimizer, exact sequence/UTF-8 prefix, seed, and
checkpoint lineage. It retains the existing claim limit: mechanics and held-out
development NLL are not public-benchmark improvement, architecture promotion, a
scaling result, or authorization for a larger model.

## Preserved boundaries

- The existing 100M runner, CLI defaults, run identities, launch graph, and job
  template are unchanged. The generic CLI cannot admit `100m`.
- No 4B scale exists in the generic parser or Slurm scale allowlist.
- The Slurm file is a single-H100 execution template and contains no `sbatch`,
  retry, requeue, array, dependency, cancellation, or self-submission logic.
- No family has been promoted by preparing this runner.
- Exact H100 memory fit and runtime are not yet qualified for every
  scale/family/batch combination. Preparation does not claim they fit within the
  24-hour request; a bounded mechanics run must establish fit before a long run.
- Training-token budgets, learning rates, batch geometry, and development sizes
  are deliberately not invented here. They must be frozen from evidence before
  submission and compared under matched budgets.
- One-H100 preparation is not distributed-training qualification and makes no
  multi-node or InfiniBand performance claim.
- No job was submitted and no checkpoint was trained by this preparation.
