# Sai Stack-Edu bounded safety-selection contract

`sai-select-stack-edu-safety-candidates` turns the bounded scanner's complete
finding population into a deterministic **candidate-only** selection. It is not
a source-admission or training gate.

The policy has no silent exceptions:

- high-confidence sensitive or invalid rows are always excluded and cannot be
  overridden;
- rows with no bounded finding remain candidates automatically;
- every `manual_review_required` row must have exactly one self-hashed review;
- a review may retain or exclude only that exact ordinal and content hash; and
- every retained row preserves the original candidate order and byte identity.

Reviews bind a pseudonymous reviewer identity hash, UTC completion time,
prospectively enumerated rationale codes, disposition, ordinal, content hash,
and a canonical row hash. Missing, duplicate, detached, reordered, or mutated
reviews fail the complete population. An empty review file is valid only when
the scanner emitted no manual-review rows.

The selector replays the entire safety scan and its parent byte bundle, freezes
the exact review artifact, recomputes every selection decision, and publishes a
create-only selected-candidate file and receipt. High-confidence rejections are
not emitted. Manual exclusions are not emitted. No source bytes are rewritten.

The word *candidate* is essential. Selection resolves only this bounded safety
pass. Independent source accuracy and usefulness sampling, global duplicate
control, benchmark decontamination, license attribution, semantic prerequisite
placement, curriculum rehearsal, and matched source-addition evidence remain
mandatory. The receipt always retains `training_authorized=false` and
`four_b_training_authorized=false`.
