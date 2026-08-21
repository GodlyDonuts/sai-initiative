# Sai 4B Specialization Research Plan

Status: prospective specialization workstream inside the architecture
tournament. This document authorizes no GPU work by itself. The training hold
in the README remains in force.

## Thesis

A 4B model will not beat a frontier model merely by shrinking its vocabulary or
narrowing its data. Sai tests whether target-domain density, lower tokenizer
fertility, and evidence-selected architecture can jointly produce a stronger
small model. English, code, math, science, and technical reasoning receive the
primary parameter and token budget; arbitrary Unicode remains losslessly
encodable through byte fallback.

Specialization is one independently measured workstream, not the definition of
the model and not permission to skip the architecture ladder. The final 4B body
must first win the gated-GQA versus GDN-hybrid versus KDA/MLA-hybrid tournament
at 100M, 300M, and 1B. Qwen3.5-4B remains a baseline and fallback control, not an
assumed Sai architecture.

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

Data quality and density matter, but Sai will measure them against data quantity
and diversity rather than assert that either dominates. Base pretraining uses an
ordered, deduplicated, benchmark-disjoint stream of English/code/math/science/
technical documents. Exact UTF-8 prefix bytes and model FLOPs are separately
matched by the implemented token-stream freezer and experiment planner.

Post-training populations are frozen separately from base pretraining:

- **skill rows** — verified target-domain instruction/solution pairs;
- **deliberate rows** — teacher-generated long traces retained only after
  rule-based verification, per the reasoning curriculum in the contract;
- **direct rows** — the same problem classes answered without long traces, so
  brevity remains a learned option, not a lost ability;
- **replay rows** — broad parent behavior, including some general and
  conversational English, so the KL anchor sees the distribution Sai must not
  drift from.

Replay is a possible post-training safety valve only after a base architecture
wins. The equal-compute control (KL weight zero) directly measures whether replay
helps rather than assuming it. Prompt-bank rows are never mislabeled as the base
pretraining corpus.

### 3. What is explicitly not claimed

- No claim that dropping multilingual data speeds up learning per se; the gate
  only tests that reallocation does not hurt and the reallocated tokens help.
- No architecture component is introduced for the first time at 4B. The final
  body is the stack selected by the 100M → 300M → 1B tournament; Qwen remains a
  matched external baseline.
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
   requires lossless round-trip, exact retained rows, and matched training.
   Untouched-body tokenizer comparisons and fixed-total parameter-reallocation
   comparisons remain separate, so segmentation is never credited for added
   depth or FFN capacity.
4. **Decontamination leaks.** Teacher traces can reproduce benchmark items even
   when source corpora are clean. Decontaminate after trace generation, against
   all five boards, on n-gram and near-duplicate matches.

## Ordered plan (maps to README build status)

1. **Freeze base and post-training data separately.** Build the exact ordered
   pretraining stream with source hashes, UTF-8 prefix bytes, document-boundary
   masks, and benchmark-disjoint evidence. Separately freeze skill, direct,
   deliberate, replay, and RL-prompt banks for later stages.
2. **Qualify tokenizer candidates.** Publish byte-normalized fertility and
   round-trip results for untouched 64K, 48K, and 32K candidates. Compare the
   tokenizer itself at fixed body geometry before testing parameter
   reallocation.
3. **Select the base architecture.** Execute the frozen 100M three-family/
   three-seed iso-data and exact iso-FLOP screen, then evidence-gated 300M and 1B
   confirmation. No 4B architecture is chosen in advance.
4. **Train one selected 4B base.** Package only the winning body, tokenizer,
   ordered stream, optimizer, seeds, runtime, and matched controls. A low-token
   mechanics run precedes the full budget.
5. **Post-train only the surviving base.** Test verified instruction data and
   frozen-parent replay against an equal-compute control; retain direct and
   deliberate behavior without mandatory revision.
6. **Run the complete five-board gate.** Apply the unchanged conjunctive
   criteria from `docs/SAI_4B_BENCHMARK_FIRST_CONTRACT.md`; one serious
   regression vetoes the claim.

Steps 3 through 6 remain behind the explicit official training order.

## Success definition

Sai v0 succeeds only if the selected 4B stack beats the unchanged 4B baseline
and every applicable equal-compute control on the declared source-disjoint
development and public gates, without a serious domain regression. Tokenizer,
architecture, data, and post-training gains are credited only from their own
matched contrasts. Anything less is a documented negative result, retained the
same way the Shohin falsification was.
