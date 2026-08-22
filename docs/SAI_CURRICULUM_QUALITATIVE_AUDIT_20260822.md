# Sai Curriculum Qualitative Audit — 2026-08-22

## Scope

This is a deterministic human-readable spot audit of the exact source-disjoint
development population used by the first 500M-token curriculum experiment:

- source path:
  `/lustre/fs1/home/sa305415/sai-initiative/artifacts/fineweb_edu_500m_curriculum_development_bee14ad_r1.jsonl`;
- source bytes: `94,204,028`;
- source SHA-256:
  `56ebbf200bae4ce21c454bd80e91328ab9a486798c83a0b729477f57e0122289`;
- source split receipt file SHA-256:
  `3db73115b5d7551033c292515be2dd34e8250b2885ac08e609ada88df5f61023`.

The sample is reproducible. For every document, compute
`SHA256(b"sai-review-v1" || bytes.fromhex(identity_sha256))`, then select the
ten lowest values independently within each declared phase. The phase document
counts are `1,753` grounding, `4,596` integration, `7,989` reasoning, and
`6,771` specialization. This yields forty reviewed documents without inspecting
model scores or changing the frozen experiment.

This is a bounded spot audit, not a statistical estimate of the entire corpus.
It can prove that a failure mode exists, but its small sample cannot estimate
the failure mode's corpus-wide prevalence.

## Observations

The sample supports the receipt's narrow claim that the phases are ordered by a
deterministic surface-complexity proxy. It does **not** support treating those
phase names as a semantic prerequisite graph.

- Grounding includes useful foundational explanations of phonetics, words,
  reading fluency, and unit conversion. It also includes an electrostatics
  exercise, actuator engineering, planetary science, offshore seismic testing,
  and abstract discussion of mental models. Those examples can presuppose
  concepts that a new learner has not acquired.
- Integration includes grammar and graph-reading explanations, but also a
  press release, historical narrative, cell-aging research, electric-vehicle
  advocacy, and astronomy prose. The phase is heterogeneous rather than a
  consistently compositional teaching tier.
- Reasoning includes crop-modification methods and executive-function material,
  but also introductory sedimentary-rock exposition, biographies, dietary
  advice, and general-interest news. Longer prose or causal markers do not prove
  that an example teaches or requires multi-step reasoning.
- Specialization includes polynomial-calculus material and technical science,
  but also introductory traffic-light detection, general plate tectonics,
  mythology, and popular-science reporting. Foundation rehearsal is intended,
  yet a surface score alone cannot distinguish useful rehearsal from
  pedagogically misplaced material.
- Web artifacts remain visible: press-release framing, social-media prompts,
  awkward grammar, malformed punctuation, boilerplate, and prose whose factual
  reliability is not established by the upstream educational score.
- Every row declares the broad `english` source domain. The current pool does
  not independently guarantee the final desired proportions or quality of
  code, mathematics, science, and technical material.

## Decision boundary

The live 100M comparison remains useful for exactly one falsifiable question:
does this reproducible surface-complexity order beat a deterministic permutation
of the identical packed sequence records? It cannot establish that Sai has a
semantic curriculum, that the underlying mixture is final-quality, or that the
data is ready for 4B training.

Consequently:

1. the current experiment may retain or reject the surface ordering only;
2. no favorable result may waive the separate source-mixture and semantic-
   prerequisite experiments;
3. the final Sai corpus must add explicit high-quality code, mathematics,
   science, and technical populations rather than infer those proportions from
   a generic English-web label;
4. the next curriculum lane must bind concepts, evidence spans, a directed
   acyclic prerequisite graph, minimum prior exposures, and later rehearsal;
5. deterministic and human-audited samples must measure factual reliability,
   self-containment, pedagogical clarity, web debris, and prerequisite
   violations before a larger optimizer run;
6. every data factor must still beat an identical-document/order or mixture
   control on held-out likelihood and source-disjoint real benchmarks.

The governing example is literal: a document teaching that yellow plus blue
produces green is not correctly ordered merely because it has short sentences.
The curriculum must first demonstrate prior, evidence-bound exposure to the
primitive colors and their relevant representations.

