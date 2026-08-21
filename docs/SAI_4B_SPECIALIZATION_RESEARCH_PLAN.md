# Sai 4B Specialization Research Plan

Status: prospective. This document authorizes no GPU work by itself. The
training hold in the README remains in force.

## Thesis

A 4B model cannot out-general a frontier model, but it can out-specialize one.
Sai spends its entire parameter, token, and vocabulary budget on English, code,
math, science, and technical reasoning, and deliberately does not spend it on
other languages, culture-specific knowledge, or multilingual chat. The bet is
that at fixed size, reclaimed capacity plus a harder, cleaner, verified data
mixture beats a general-purpose parent on the target domains without regressing
on the five-board public gate.

Every claimed advantage below is stated as a measurable quantity with a control,
because the Shohin falsification showed that plausible mechanisms lose to
matched controls more often than they win.

## Where the specialization budget actually comes from

### 1. Vocabulary capacity (measured, not assumed)

The parent tokenizer devotes a large fraction of its vocabulary to scripts Sai
will never emit. `src/sai/tokenizer/capacity.py` already measures this
conservatively: only zero-use, single-unsupported-script pieces are proposal
candidates, and every evaluation token is protected. The recovered embedding
rows are the only "free" parameters in this plan; everything else is a
reallocation of training tokens, not weights.

Open question to resolve before surgery: whether recovered rows are (a) simply
dropped (smaller checkpoint, faster softmax), or (b) reassigned to new merged
English/code pieces to lower fertility on the target corpus. Option (b) is the
real prize — lower tokens-per-character on code and LaTeX is effective context
extension and effective compute increase at inference — but it requires the
continued-pretraining repair the contract already demands. Decide with the
fertility numbers from the audit, not in advance.

### 2. Data mixture density

At 4B, data quality dominates data quantity. The strongest small-model results
(heavily filtered educational web text, textbook-style synthetic data, verified
reasoning distillation) all share one property: a higher fraction of tokens
that change model behavior. Sai's admitted-corpus rule makes this explicit —
every training row is English/code/math/science/technical, verified where
verification is mechanical (execution for code, answer checking for math,
rule-based checks for logic), and benchmark-decontaminated before admission.

The mixture to freeze (build-status item: prompt banks):

- **skill rows** — verified target-domain instruction/solution pairs;
- **deliberate rows** — teacher-generated long traces retained only after
  rule-based verification, per the reasoning curriculum in the contract;
- **direct rows** — the same problem classes answered without long traces, so
  brevity remains a learned option, not a lost ability;
- **replay rows** — broad parent behavior, including some general and
  conversational English, so the KL anchor sees the distribution Sai must not
  drift from.

Replay is the specialization safety valve: the narrower the skill mixture, the
more the frozen-parent KL matters. The equal-compute control (KL weight zero)
directly measures whether that is true rather than assuming it.

### 3. What is explicitly not claimed

- No claim that dropping multilingual data speeds up learning per se; the gate
  only tests that reallocation does not hurt and the reallocated tokens help.
- No architecture changes at 4B. The parent's architecture is assumed adequate;
  the changed factor stays narrow (adapter + data + optional tokenizer).
- No claim of frontier-general capability. "Frontier" for Sai means: best
  public five-board macro among ~4B single-pass checkpoints on the target
  domains, verified against the unchanged parent and equal-compute control.

## Risks the plan must retire

1. **English-only regression on instruction following.** IFEval is English, but
   general instruction diversity may live partly in data Sai excludes. The
   replay population and the gate's per-benchmark −1.0 floor cover this;
   monitor IFEval first in the pilot.
2. **Verification bias.** Keeping only verifiable rows skews toward problems
   with checkable answers. Direct rows and replay keep unverifiable-but-useful
   behavior in the mixture; track the verified/unverified ratio as a frozen
   mixture parameter, not an accident.
3. **Tokenizer surgery silently changing behavior.** The contract already
   requires lossless round-trip, exact retained rows, and matched continued
   pretraining. Add one more: the untouched-tokenizer candidate must run the
   full gate first, so tokenizer effects are never confounded with data
   effects.
4. **Decontamination leaks.** Teacher traces can reproduce benchmark items even
   when source corpora are clean. Decontaminate after trace generation, against
   all five boards, on n-gram and near-duplicate matches.

## Ordered plan (maps to README build status)

1. **Freeze the prompt banks.** Enumerate skill, direct, deliberate, replay,
   and RL sources with hashes; fix the verified/unverified and
   deliberate/direct ratios; run decontamination; commit the manifest.
2. **Qualify tokenizer candidates.** Run the capacity audit on the frozen
   admitted corpus plus all evaluation prompts; publish fertility numbers for
   untouched vs. reduced vs. reduced-plus-remerged; decide drop vs. reassign.
3. **Package the runtime.** Immutable parent hash, adapter geometry, seeds,
   optimizer, and the matched-control switch already prototyped in
   `src/sai/training/replay.py`.
4. **Low-token pilot.** Smallest run that can rank {candidate, control} on a
   held-out verified slice; go/no-go before any full run.
5. **Full five-board gate.** Unchanged conjunctive criteria from
   `docs/SAI_4B_BENCHMARK_FIRST_CONTRACT.md`; one serious regression vetoes.

Steps 4 and 5 remain behind the explicit official training order.

## Success definition

Sai v0 succeeds if, at equal tokens, updates, and inference cost, the
specialized candidate passes every gate conjunct against both the unchanged
parent and the equal-compute control. Anything less is a documented negative
result, kept in this repository the same way the Shohin falsification was.
