# Sai Frontier Architecture Tournament

Status: prospective. Training is held. This document authorizes no GPU work.

## Decision

Sai will not be a DeepSeek-R1-era reasoning recipe attached to a Qwen checkpoint.
It will be a scale-gated search for the strongest practical dense model near four
billion total parameters. Qwen3.5-4B is a reference implementation and fallback
control, not the chosen architecture.

The research report supplied on 2026-08-21 contains several current and useful
signals:

- [Kimi Linear](https://arxiv.org/abs/2510.26692) reports that a KDA/MLA hybrid
  beats a full-MLA control under the same recipe, with large long-context cache
  and throughput gains.
- [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) deploys 69 KDA and 24 gated
  MLA layers, AttnRes, and SiTU-GLU at frontier scale.
- [Gated Attention](https://arxiv.org/abs/2505.06708) reports consistent gains
  from a head-specific sigmoid gate after SDPA across 1.7B dense and 15B MoE
  experiments totaling 3.5T training tokens.
- [Attention Residuals](https://arxiv.org/abs/2603.15031) reports consistent
  scaling gains and offers Block AttnRes as a lower-overhead form.
- [DeepSeek Engram](https://github.com/deepseek-ai/Engram) reports gains under
  iso-parameter and iso-FLOP constraints by adding deterministic O(1) n-gram
  lookup memory.
- [Multi-token prediction](https://arxiv.org/abs/2404.19737) reports sample
  efficiency, coding, and speculative-decoding gains.
- [Future-summary prediction](https://arxiv.org/abs/2510.14751) reports 3B/8B
  gains over NTP and MTP, but remains exploratory for Sai.

These are evidence for experiments, not evidence that stacking them wins at 4B.
DeepSeek V4-specific claims without a directly verifiable primary release are
excluded from the first tournament.

## Fixed target

- Dense, text-only, approximately 4B total parameters.
- English, code, math, science, and technical material are the primary domain.
- Arbitrary Unicode remains losslessly encodable through byte fallback.
- Tied input/output embeddings.
- One checkpoint with direct and deliberate behavior; no mandatory revision
  call or hidden external draft.
- Public claims require complete official source-disjoint benchmarks.

## Experimental ladder

### Stage 0: CPU and kernel qualification

No optimization updates. Verify parameter accounting, tokenizer round trips,
forward/backward equivalence, recurrent-state reset, causal masking, chunkwise
versus recurrent KDA equivalence, MLA cache reconstruction, deterministic MTP
targets, memory geometry, and exact configuration receipts.

### Stage 1: approximately 100M parameters

This is a mechanics and systems screen, not a capability claim. Under an
official order, measure stability, convergence, useful bytes/GPU-second, peak
memory, decoding throughput, and KV/state bytes per generated token. Any NaN,
unbounded state, kernel mismatch, or unexplained loss divergence rejects the
variant.

### Stage 2: approximately 300M parameters

Run three seeds per factor on frozen data. Change exactly one factor from its
control. Every factor has two declared contrasts: iso-data holds admitted UTF-8
bytes fixed and reports the compute difference; iso-FLOP holds model FLOPs fixed
and consumes a prefix of the same ordered data stream. Both retain the same
sequence curriculum, initialization policy, and evaluation decoding. Promotion
requires a positive paired 95% confidence bound on the declared primary metric
and no material domain regression.

### Stage 3: approximately 1B parameters

Confirm only 300M survivors. Test interactions one at a time. A component whose
benefit disappears, reverses, or depends on an unmatched budget is removed.
AttnRes may enter only here after the underlying mixer wins at 300M. Never combine
AttnRes and mHC in the same initial candidate.

### Stage 4: approximately 4B parameters

Train exactly one evidence-selected stack plus the controls required to support
the claim. The complete HumanEval+, MBPP+, IFEval, MuSR, and CorrectBench gate is
terminal. One serious regression rejects promotion regardless of macro average.

## Factor order

1. Tokenizer: 64K control versus 48K and 32K; 16K diagnostic only.
2. Core mixer: gated GQA versus 3:1 Gated DeltaNet/gated attention versus 3:1
   KDA/gated MLA.
3. Position: partial RoPE versus NoPE MLA, only after the MLA path is viable.
4. FFN: SwiGLU versus SiTU-GLU.
5. Optimizer: AdamW versus Muon for matrices plus AdamW for embeddings, norms,
   and scalars.
6. Objective: NTP versus NTP plus one or two MTP heads.
7. Static memory: no memory versus Engram, only after tokenizer selection.
8. Residual: standard PreNorm versus Block AttnRes, only after core selection.
9. Future-summary prediction: exploratory 300M branch only.

No factor can be credited from a comparison that changes architecture, data
identity/order, optimizer, and evaluation contract simultaneously. Iso-data and
iso-FLOP results must remain separately labeled; they are not interchangeable.

## Tokenizer accounting

At residual width 2,560 with tied embeddings, each vocabulary entry costs 2,560
parameters and 5,120 BF16 bytes. Relative to 248,320 entries:

| Vocabulary | Embedding parameters | Parameters freed |
| ---: | ---: | ---: |
| 64K | 163.84M | 471.86M |
| 48K | 122.88M | 512.82M |
| 32K | 81.92M | 553.78M |

Tokenizer-only tests keep body geometry fixed. Reallocation tests separately
hold total parameters near 4B and reinvest savings into depth or FFN width. This
prevents attributing a depth gain to segmentation or vice versa. Iso-data
tokenizer budgets use admitted UTF-8 bytes; iso-FLOP budgets use the exact same
ordered stream but may consume a different-length prefix. Validation loss is
normalized by source bytes, not tokens.

Protected behavior includes ASCII, byte fallback, English, source-code syntax,
whitespace/indentation, identifiers, URLs, numbers, units, Greek, math, LaTeX,
science notation, and all special/control tokens. Unsupported-language fertility
may regress; round-trip correctness may not.

## Development and public evidence

Development identities are disjoint from public benchmark identities and cover:

- English knowledge and instruction following;
- code generation and execution;
- quantitative math and formal reasoning;
- science and technical question answering;
- long-context retrieval and synthesis;
- direct-response retention; and
- throughput, latency, peak memory, and cache/state size.

Tokenizer comparisons use byte-normalized validation NLL. Capability comparisons
use fixed prompts, identical decoding, and official or executable scorers. Every
report binds ordered row identities, data, configuration, environment, source,
checkpoint, accounting, and output hashes.

The five public boards are not an iterative tuning set. They are run only at the
declared promotion boundary.

## Explicit exclusions

- No 4B run before smaller-scale evidence and an official user order.
- No always-revise architecture or mandatory long reasoning.
- No MoE merely to fit a 4B-total target.
- No AttnRes plus mHC bundle.
- No DeepSeek V4 compressed sparse attention unless ultra-long context becomes
  the primary project claim and a primary implementation is qualified.
- No FusedKV on a KDA/MLA winner; it is relevant only to a conventional-attention
  branch.
- No FP8 claim without measured stability, quality parity, and useful
  bytes/GPU-second improvement.
- No architecture promotion from perplexity alone.

## Post-training, after a base-model win

Build Base, then Instruct, then Reasoning. Use verified multi-teacher distillation,
unit-tested code, checked math, executable logic, and RL with verifiable outcomes.
Expose fast and deliberate modes. Measure adaptive reasoning against fixed-fast
and fixed-deliberate controls. MTP heads may be reused for self-speculative
decoding only after quality parity is established.

The goal is not to collect fashionable components. It is to produce a traceable
set of independently measured decisions that compound into a better 4B model.
