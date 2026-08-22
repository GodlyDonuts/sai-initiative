# Sai 2026 Data Research Synthesis

Status: current primary-source research boundary as of 2026-08-22. This is a
prospective prior for Sai experiments, not a copied mixture and not training
authorization.

## What current open recipes establish

### SmolLM3

Hugging Face reports that SmolLM3-3B was trained on 11.2T tokens using three
evolving pretraining mixtures. Its published proportions move from
`85% web / 12% code / 3% math`, to `75% / 15% / 10%`, and finally
`63% / 24% / 13%`. The team reports choosing mixture ratios through 3B
ablations trained on 50B–100B tokens. Code and math are therefore present in
the foundation stage rather than introduced as an abrupt late-domain shock,
but higher-quality and more difficult specialist data grows substantially
after broad capability is established.

SmolLM3 then separates base pretraining from long-context and reasoning
mid-training. Its long-context stage extends 4K→32K→64K, while reasoning
mid-training uses a separate reasoning-trace population. The recipe also
reports that merely upsampling long code repositories, books, and web pages did
not improve its long-context benchmarks. This argues for measuring data
function rather than assuming that long or specialized text is automatically
useful.

Primary source:
[Hugging Face, “SmolLM3: smol, multilingual, long-context reasoner”](https://huggingface.co/blog/smollm3).

### OLMo 3

Ai2 reports a similar stage separation. OLMo 3 starts with a broad 5.9T-token
Dolma 3 Mix, follows it with a 100B-token Dolmino mid-training mixture focused
on high-quality math, science, code, instruction following, and reading
comprehension, and uses a separate 50B-token Longmino stage for long-context
behavior. The underlying 9.3T-token pool includes web, science PDFs, code,
mathematics, and encyclopedic text, with stronger deduplication,
decontamination, quality filtering, and mixture control than its predecessor.

Primary source:
[Ai2, “Olmo 3: Charting a path through the model flow”](https://allenai.org/blog/olmo3).

### DataComp-LM

DataComp-LM ran controlled dataset experiments from roughly 400M through 7B
parameters and reports that small models can provide useful signal about which
data strategies transfer upward. Its strongest baseline relied on model-based
filtering and materially improved a standard 7B model. It also found that
details of the filtering model changed downstream performance substantially
and that human quality judgments alone had limited predictive value.

For Sai, this supports matched small-model source screens, but it also means a
human-looking “good document” label cannot admit a source by itself. Human
review remains necessary for semantics, licensing, extraction, and safety;
actual model evidence decides whether the admitted source improves capability.

Primary source:
[DataComp-LM paper](https://arxiv.org/abs/2406.11794).

### Vocabulary curricula

Recent small-scale work reports gains from progressively expanding token
granularity according to model entropy. It also emphasizes that inconsistent
numeric tokenization can harm arithmetic. The evidence is presently limited to
small GPT/enwiki8 experiments and cannot justify dynamic production
tokenization for Sai. It does justify testing the current individual-digit
policy against a frozen number-aware alternative before selecting the 4B
tokenizer.

Primary source:
[“Scaling LLM Pre-training with Vocabulary Curriculum”](https://arxiv.org/abs/2502.17910).

## Sai interpretation

Sai's curriculum hypothesis is **progressive composition with continuous
rehearsal**, not “only simple English first, then suddenly advanced domains.”

1. Grounding begins with broad English and direct concepts, plus bounded early
   examples of code, quantities, elementary mathematics, observation, and
   technical notation.
2. Integration increases worked examples and cross-representation links only
   after their primitives have measured coverage.
3. Reasoning increases verified proofs, algorithms, causal explanations, and
   multistep quantitative work while continuing grounding rehearsal.
4. Specialization increases high-quality code, math, science, and technical
   density only after prerequisite coverage is demonstrated.
5. Long-context adaptation and explicit reasoning traces remain separately
   measured stages rather than being mixed into base pretraining by intuition.

No published percentage is copied into Sai. SmolLM3 is evidence that staged
ratios can work, not evidence that its exact web/code/math mixture is optimal
for an English-focused 4B model. OLMo 3 is evidence for stage separation, not
permission to treat every Dolma component as licensed, clean, or useful for
Sai.

## Executable experiment order

The following factor isolation remains mandatory:

1. Complete the live same-record curriculum-order experiment. It changes only
   sequence order and cannot establish source quality or mixture ratios.
2. Admit each candidate source through exact provenance, license, extraction,
   safety, deduplication, and benchmark-decontamination evidence.
3. Compare every source addition against an equal-token selected-web control on
   held-out likelihood and real source-disjoint benchmarks.
4. Build a broad-domain tokenizer population only from sources that survive
   those gates; compare vocabulary capacity and numeric pre-tokenization with
   matched body and compute.
5. Compare staged mixture schedules only after their component sources win
   independently. Preserve general rehearsal in every later stage.
6. Select the 4B base mixture from sub-4B evidence. Reasoning mid-training,
   long-context extension, SFT, preference optimization, and RL remain separate
   later decisions.

## Current Sai evidence gaps

- The completed tokenizer tournament was trained and measured on an
  English-labeled FineWeb-Edu population only.
- FineMath rows remain candidates pending blinded human review and subsequent
  deduplication, decontamination, provenance, and prerequisite mapping.
- Stack-Edu Python rows remain metadata/content nominations pending exact
  license approval, opt-out replay, secret scanning, global deduplication,
  benchmark decontamination, and source-addition evidence.
- No science or technical source currently has complete admission evidence.
- The live 500M-token curriculum test has not reached its terminal held-out or
  benchmark result.

Until these gaps close, the correct output is better evidence—not a larger
training run.
