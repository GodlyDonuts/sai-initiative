# Sai Foundational Syllabus Candidate

The original 50-concept semantic seed is intentionally preserved because the
authored-source review artifacts already bind its exact bytes. It is too coarse
to serve as Sai's final data syllabus: for example, it moves from numbers and
arithmetic directly toward functions without separately representing place
value, signed numbers, fractions and decimals, equations, or coordinates.

`SAI_FOUNDATIONAL_SYLLABUS_ADDITIONS_CANDIDATE.json` adds 75 concepts without
mutating that lineage. The deterministic composer produces a balanced
125-concept candidate: 25 concepts in each of English, mathematics, code,
science, and technical systems. New foundations include written word structure,
clauses, negation, coreference, place value, fractions, equations, types,
iteration, scientific classification, atoms, calibration, and procedures.
Later composition includes calculus, matrices, concurrency, evolution,
distributed systems, and machine learning only after their declared
prerequisites.

Every dependent addition requires 32 prior confident documents for each
prerequisite. A concept's earliest phase cannot precede any prerequisite's
earliest phase. Grounding concepts remain rehearsed in all four phases;
integration and reasoning concepts also retain later rehearsal. The composed
artifact is replayed through Sai's production taxonomy validator, proving exact
fields, complete five-domain coverage, prerequisite existence, phase geometry,
and acyclicity.

This remains a candidate syllabus, not an accepted ontology. Its names, edges,
granularity, omissions, and exposure counts require independent subject-matter
review. A new annotation policy must bind the composed bytes, reviewers must
calibrate on real source documents, and the admitted corpus must demonstrate
coverage and zero premature exposure. The candidate authorizes no data
reordering, optimizer work, architecture promotion, or 4B training.

The first graph-risk replay found only two roots, 263 hard edges, 66 cross-domain
hard edges, and a maximum hard-prerequisite depth of 12. These are diagnostics,
not failures by themselves, but treating every helpful relationship as a hard
gate could over-serialize the curriculum. `sai-audit-foundational-syllabus`
therefore freezes direct and transitive depth, centrality, cross-domain edges,
roots, leaves, and an exact review set. Every flagged edge must be classified as
hard, supporting, or removed before document annotation begins.
The replay also found one concrete inherited phase inversion:
`code.testing` begins in integration although its declared hard prerequisite
`english.evidence` begins in reasoning. Final application must rephase one side
or reclassify/remove that edge; preserving the contradiction cannot pass.

`sai-build-foundational-syllabus-review` makes that classification executable
without a server. Its exact offline workspace covers all 125 concepts and all
263 current edges, not just the 67 flagged concepts. For every concept, a
reviewer must assess inventory verdict, name, granularity, earliest phase, each
existing edge, missing prerequisites, and a written rationale. Complete JSONL
can be exported only after every row is reviewed. The workspace is evidence
collection; it cannot itself qualify or mutate the syllabus.

`sai-compare-foundational-syllabus-reviews` then requires two complete files
with distinct reviewer identities and exact file hashes. It compares concept
verdicts, names, phase placement, granularity, every existing edge
classification, and every proposed missing prerequisite. Narrative rationales
are preserved but need not be textually identical. Any structured disagreement
is published with both decisions and leaves subject review unqualified; neither
reviewer is silently preferred. Even complete consensus still requires a final
replay that applies the agreed graph and reruns acyclicity, phase, and coverage
checks.

`sai-apply-foundational-syllabus-reviews` implements that final replay for
complete structured consensus. Hard edges remain exposure gates; supporting
edges move to a separate hash-bound context sidecar; removed edges disappear;
and agreed names, added prerequisites, and phase changes are applied. The
revised hard graph must again pass exact production taxonomy validation.
Concept rejection or non-appropriate granularity cannot be represented as a
rename or edge edit, so either outcome stops for an explicit split/merge/removal
candidate. Even a successfully applied consensus remains unqualified until a
new annotation policy, human calibration, and real source-coverage evidence
bind the reviewed bytes.
