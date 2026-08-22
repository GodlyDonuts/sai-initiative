# Sai Semantic Prerequisite Candidate

Status: candidate seed only. It authorizes no training, data reordering,
architecture promotion, or 4B execution.

`SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json` is the first concrete input
to the evidence-bound taxonomy builder. It contains 50 concepts, exactly ten in
each Sai domain: English, mathematics, code, science, and technical systems.
The graph is cycle-free and includes cross-domain dependencies such as numbers
before program literals, ratios before scientific units, functions before
feedback, evidence before experiments, and tests before reliability claims.

This seed deliberately starts with two primitives: written symbols and numbers.
Every later concept has declared prerequisites. Dependent concepts require at
least 32 prior confident documents for each prerequisite. Each concept also has
prospectively declared per-phase exposure minima; the primitive concepts must
remain represented through grounding, integration, reasoning, and
specialization rather than disappearing after first exposure.

## What this proves

- The current taxonomy schema can express a nontrivial cross-domain dependency
  graph and replay its rehearsal obligations.
- The graph parses, covers all five domains evenly, and passes the structural
  acyclicity and threshold validator.
- The concept list is an inspectable artifact rather than an implicit prompt or
  an annotator's unrecorded ontology.

## What this does not prove

- The 50 concepts are not a complete ontology for a strong 4B model.
- Names, edges, and thresholds have not yet passed subject-matter review.
- No curriculum document has been mapped to these concepts.
- The prospective annotation policy is now explicit and hash-bound: positive
  labels require verifiable source spans, ambiguous labels are omitted and
  flagged, same-document exposure never satisfies a prior prerequisite, and
  blind concept-set disagreement is capped at five percent. No annotator
  identity, completed calibration population, or human audit has yet been
  accepted for this list.
- Surface curriculum job results cannot validate or waive this semantic gate.

Before the list becomes a prospective taxonomy, it must be expanded and audited
against authored instructional sequences and diverse source samples. At a
minimum the review must identify missing primitives, overly broad concepts,
ambiguous evidence spans, false dependency edges, domain-specific gaps, and
whether 32 prior documents is sufficient for stable exposure. The frozen
annotator must then label exact evidence spans and confidence, and an independent
sample of at least 100 documents must remain at or below the five-percent
disagreement cap.

Only after those artifacts exist may `sai-validate-prerequisites
build-taxonomy` publish a prospective taxonomy. The full curriculum annotation
population must then replay with zero prerequisite violations, no missing
concepts, and no phase-rehearsal shortfall. Even a semantic PASS authorizes only
a matched small-model order comparison against the same-record surface and
permuted controls; it does not authorize 4B training.
