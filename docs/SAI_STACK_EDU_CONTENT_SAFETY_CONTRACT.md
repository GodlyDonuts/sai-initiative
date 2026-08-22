# Sai Stack-Edu bounded content-safety contract

Exact provenance and byte identity do not make source code suitable training
data. `sai-scan-stack-edu-content-safety` consumes only a fully replayed
`sai-stack-edu-content-verification-v1` bundle and emits one deterministic
finding for every aligned source object.

The bounded scanner separates three outcomes:

- `rejected_high_confidence_sensitive_or_invalid` for private-key headers,
  recognized production credential formats, non-placeholder credential
  assignments, NUL bytes, or non-whitespace control characters;
- `manual_review_required` for personal-email candidates, JWT-like or
  high-entropy strings, generated-code markers, extreme/minified lines,
  excessive line repetition, or Python 3 parse failures;
- `candidate_clean_by_bounded_scanner` only when none of those bounded signals
  fire.

The last outcome is deliberately named **candidate**, not clean source. Regex
and entropy scans are incomplete. They cannot establish the absence of novel
secret formats, personal information, malware behavior, dependency supply-chain
risks, or subtle generated and benchmark-derived code.

Every finding binds the exact repository, path, blob ID, content SHA-256,
ordinal, policy hash, decision, reasons, and measured signals. The receipt
reopens the opt-out alignment, content index, and complete sealed content
bundle; rescans every byte; verifies the parent again after scanning; and
publishes immutable findings. Any mutation, missing row, reordered identity,
or policy drift fails replay.

This gate never deletes or silently rewrites source. High-confidence rejects
are excluded by a later create-only selection. Review findings require a
separately frozen human or independently qualified adjudication population.
Even a zero-finding receipt retains `training_authorized=false` and
`four_b_training_authorized=false` until global deduplication, decontamination,
semantic prerequisite placement, and matched source-addition evidence pass.
