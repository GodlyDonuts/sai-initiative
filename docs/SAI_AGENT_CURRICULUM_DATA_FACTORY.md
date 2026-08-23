# Sai Agent Curriculum Data Factory

Status: implementation-ready, authentication and calibration pending. This process
selects and orders data. It does not itself make any document training-ready.

## Objective

Build a large English-first pretraining population whose documents are both high
quality and presented after the concepts they assume. A source's prestige does not
override its learnability. A correct advanced paper can be harmful in the grounding
phase if the model has not yet seen the vocabulary, definitions, and methods that make
the paper intelligible.

The factory uses `stealth/ox-alpha` as a scalable teacher through the Nous Portal, but
does not run 1,000 unconstrained tool-using processes. It creates 1,000 deterministic
logical shards and schedules a smaller, measured number of physical Stokes workers.
Each worker makes bounded OpenAI-compatible classification calls. This retains the
parallelism while respecting the account-wide RPM/TPM limit and avoiding a 429 storm.

Official integration points:

- Nous inference endpoint: `https://inference-api.nousresearch.com/v1`
- model identity: `stealth/ox-alpha`
- Hermes/Nous model catalog: `https://hermes-agent.nousresearch.com/docs/api/model-catalog.json`
- Portal authentication is supplied at execution time and is never written into a
  receipt, request artifact, Slurm submit line, or repository file.
- The exact loopback-only Hermes subscription proxy
  `http://127.0.0.1:8645/v1` is allowed for local calibration. It attaches the OAuth
  credential without exposing it to the labeler; no non-loopback HTTP endpoint is
  accepted.

## Source ladder

1. **Grounding:** carefully licensed introductory textbooks, clear encyclopedic
   explanations, basic educational material, elementary worked examples, basic code,
   and concrete science. Definitions and direct examples dominate.
2. **Integration:** intermediate textbooks, coherent tutorials, technical manuals,
   documentation, multi-concept worked problems, and clean representative code.
3. **Reasoning:** proofs, extended derivations, verified solution traces, systems
   explanations, advanced code, and selected scientific exposition. Synthetic
   reasoning is allowed only here and only with generator and verifier provenance.
4. **Specialization:** advanced documentation, mature permissively licensed
   repositories, and research papers after their prerequisites have been established.
   Foundational rehearsal remains present.

Research papers are a high-value late source, not an automatic quality label. PDF
extraction artifacts, references, boilerplate, tables without context, retractions,
unsupported claims, duplicated versions, licenses, and prerequisite load must be
handled before admission. Survey/tutorial papers and well-written introductions can
enter earlier than dense methods sections; abstracts alone do not substitute for
teaching evidence.

## Decision pipeline

1. Preserve exact source revision, row identity, content SHA-256, license, and source
   type. Reject secret/private material before any external request.
2. Apply deterministic hygiene: decoding, minimum useful length, corruption,
   exact/near deduplication, benchmark decontamination, license, and obvious boilerplate.
3. Present each surviving document to one comprehensive frontier-model curator for
   the production bulk pass. The curator evaluates curriculum placement, quality,
   factual reliability, source risk, and pedagogical utility in one response. The
   document is treated as untrusted prompt text; tools are disabled. Three blinded
   perspectives are retained for calibration samples, ambiguous rows, rare domains,
   and high-value audit samples rather than spent on every routine document.
4. Each judgment records quality, English suitability, domains, concepts taught,
   prerequisites assumed, difficulty, curriculum phase, pedagogical role, risks,
   confidence, and exact evidence quotes. The host converts quotes to source-bound
   offsets and hashes.
5. In the bulk path, retain only a frontier-model `retain` judgment with quality and
   English scores at least 3/4, confidence at least 0.8, a non-reject curriculum
   phase, and zero blocking risks. Borderline rows go to the additional-review queue.
   The three-perspective audit path retains its two-of-three majority rule.
6. Re-run global deduplication and benchmark decontamination after extraction,
   normalization, or synthesis. Only then map accepted documents into the frozen
   120B-token phase ledger and tokenize/pack them.

## Calibration before the large fan-out

The first live operation is not the full corpus. Freeze a diverse human-labeled set
stratified by source type, domain, quality, language, and expected phase. Run the exact
three-perspective prompt on that set and measure retention precision, rejection recall,
phase confusion, calibration by confidence, and error rates for research papers,
code, math, and general web separately. Prompt bytes and thresholds are frozen before
the held-out calibration partition is opened.

Initial execution ladder:

- 100 documents x three judgments: API/schema and rate-limit probe.
- 10,000 documents x one comprehensive judgment, with a stratified three-judgment
  audit subset: calibration and throughput estimate.
- 1,000 logical shards across a measured physical concurrency: full source inventory.

The free teacher endpoint is not the final bulk classifier. After the stratified
10,000-document calibration, train a small local multi-head student on the frozen
teacher/human labels (quality, English suitability, domain, risks, prerequisites, and
phase). Use it to score the full corpus cheaply, then send its uncertainty tail,
distribution-edge samples, and every prospective high-value inclusion back through
the three-perspective teacher. This active-learning loop turns tens of thousands of
expensive judgments into coverage over millions of documents without pretending that
one weak local score is final evidence.

The student is admitted only if a held-out, source-stratified human/teacher partition
shows the predeclared precision and calibration targets. Teacher and human labels stay
authoritative; the student may triage or prioritize, never override a blocking risk or
license/decontamination failure.

Physical worker count is increased only while completed-row throughput rises and 429,
timeout, malformed-JSON, and provider-error rates remain within the frozen operational
budget. Retries preserve the same request identity and never become extra votes.
The live `stealth/ox-alpha` calibration showed that the Portal route must not receive
the optional OpenAI `response_format` extension: it returned empty content under
concurrency. JSON conformance is therefore enforced by the explicit prompt template
and the local strict validator, with malformed output retried as the same vote.

## Evidence and failure behavior

`sai-agent-data-candidate-v1` binds text to source and provenance. Each
`sai-nous-agent-label-receipt-v1` binds the model, endpoint, prompt rubric, request,
attempt statuses, provider response identity, token usage, normalized judgment, and
exact source evidence without persisting the credential. `sai-agent-data-aggregate-v1`
requires exactly three slots and conservatively resolves them.

Outputs are create-only and resumable at the candidate/slot level. Missing, duplicate,
malformed, ungrounded, or contradictory judgments fail closed. A completed label says
only how the document should be considered; `training_ready` remains false until
license, deduplication, decontamination, prerequisite ordering, tokenizer selection,
and packed-stream replay all pass.

## Stokes execution geometry

`jobs/sai-nous-label-worker-cpu.sbatch` groups 1,000 logical shards across a caller-set
number of physical CPU workers, each with a small bounded HTTP concurrency. Stokes
currently permits at most 250 submitted jobs for the account, so the prospective
starting point is 8 physical workers x 4 requests = 32 live requests, adjusted by the
100-document rate probe. No GPU is requested. The key is read inside the allocation
from an owner-only `0400`/`0600` file and is not exported by `sbatch`.
