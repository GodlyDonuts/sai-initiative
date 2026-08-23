# Sai 2026 Data Research Synthesis

Status: current primary-source research boundary as of 2026-08-23. This is a
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

The current public identities resolved on 2026-08-22 are Dolma 3 150B Mix
revision `afa92bfb22366821c5e6cd427cdd036b34b713ef`, Dolmino 100B Mix revision
`f23942ae8a8114af6e992efe8188ce8c531acd16`, and reconstruction-code revision
`1a9daced81670e0fa768e47fbed32af6694a1865`. These freeze candidates for
inspection; they do not admit their bytes.
The first exact compressed-shard diagnostics show why this boundary matters:
three sampled components physically repeat most document IDs four to nine
times, and the sampled Stack-Edu shard is predominantly `no_license`. See
[`SAI_DOLMA3_BOUNDED_SHARD_AUDIT_20260822.md`](SAI_DOLMA3_BOUNDED_SHARD_AUDIT_20260822.md).

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

### Frequency- and length-aware subdocument deduplication

August 2026 work from Tencent's Hunyuan team separates duplicate detection from
copy retention. It segments at natural boundaries, globally counts normalized
exact subdocument groups, and assigns retention budgets from group frequency
and span length. In its FineWeb-Edu experiment, matched keep-one retention
scored `52.14` versus `52.90` for the full frequency/length-aware pipeline;
their subdocument-only variant scored `52.92`. The code-containing-web setting
improved all five reported benchmarks and moved the unweighted average from
`37.98` to `41.40`. These results are specific to the reported 30B-A3B training
setup, but they directly reject “keep one copy of every repeated paragraph” as
a consequence-free default.

Sai therefore separates three decisions:

1. whole-document normalized exact copies may use deterministic keep-one;
2. subdocument copies require global frequency, span length, and surrounding
   document context before deletion;
3. semantic near-duplicates remain a separately measured gate.

The prospective subdocument comparison must include unchanged, keep-one, and
frequency/length-aware controls under identical data and compute. It must also
preserve code structure and prevent isolated deletions from fragmenting prose.

Sai now has an executable implementation in
`sai.data.frequency_length_subdocument_deduplication`. It uses bounded
external-memory aggregation; exact source replay; the reported `tau_seg=32`,
`tau_del=100`, `L0=512`, `N=100/3` defaults; boundary-document protection; and
coherence-aware contiguous deletion. Fenced code is exact and indivisible.
Code-domain documents fail closed as indivisible pending a qualified
language-aware structural parser. Every transformed document receives a new
content-bound identity and a text-free parent/output receipt. Implementation is
not empirical validation: unchanged/keep-one/adaptive training evidence remains
required before promotion.

Primary source:
[“Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining”](https://arxiv.org/abs/2608.03089).

### Coherent source-grounded synthetic books

July 2026 controlled work isolates document organization from content and token
count. Its pipeline retrieves real source material, clusters related material,
plans hierarchical tables of contents, and writes source-grounded sections into
complete books. It produced 686,000 textbooks totaling 32B tokens across more
than 15,000 disciplines. The full-book representation beat a content-matched
split control by `+1.02` mean points and a retrieval-pool-matched rephrase
control by `+1.17`; random concatenation also remained below coherent books.

For Sai, synthetic volume is therefore not the objective. The admissible unit
is a source-bound knowledge work with exact anchor identities, a coherent
concept hierarchy, multiple useful representations, and explicit separation
from generic rephrasing. Cross-domain books should connect independently
grounded concepts rather than invite a generator to invent both premises and
conclusions.

Primary source:
[“Beyond Rephrasing: Book-Level Organization Improves Synthetic Textbook Data for Mid-Training”](https://arxiv.org/abs/2607.28109).

### Curriculum ordering evidence

EACL 2026 reports more than 200 pretraining runs up to 100B tokens. Curriculum
warmup reduced the steps needed to reach baseline by 18–45%, and returning to
mixed sampling retained improvements up to 3.5%. Compression ratio, MTLD
lexical diversity, and Flesch readability were the strongest tested surface
signals. These are useful observable axes, not semantic-prerequisite oracles;
Sai's concept graph and model-centric learnability measurements remain separate.

Primary source:
[“Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning”](https://aclanthology.org/2026.eacl-long.271/).

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

- The two pinned reservoirs reference 23,680,076,298,761 physical candidate
  bytes, but physical repository bytes are neither unique text nor training
  data.
- Two exact payload probes cover nine immutable members: 13,359,494,149
  physical bytes yielded 22,576,343,154 UTF-8 text bytes and 17,638,716,209
  bytes inside the current mechanical size window. This bounded sample cannot
  be extrapolated to the reservoir.
- Six immutable audit populations contain 2,103 rows. The source-disjoint
  Common Pile confirmation is complete at 224/224 compiler receipts, but no
  source-wide admission rate follows from those rows.
- Pressbooks and Public Domain Review completed bounded source pilots. Their
  3,353 raw rows yielded 3,301 benchmark-disjoint rows and 3,290
  within-source near-deduplicated rows; the full bounded cross-source replay
  found zero additional groups. Attribution and text-free external metadata
  lineage are exact. A live, no-HTML-persistence probe covered all 3,290 rows
  through 1,160 source/policy targets and observed expected declaration
  evidence for 1,719 records. It also measured 632 Pressbooks HTTP 403
  responses and 70 exhausted transport retries. A text-free per-identity queue
  now separates all seven observed/access/missing/transport adjudication paths
  without making an automated legal decision. Governing-scope adjudication,
  rights verification, representation verification, and full-reservoir
  deduplication remain open. Hermes compilation of the 3,290 survivors is in
  progress.
- Exact training-ready data remains zero bytes, and 4B training remains
  unauthorized.

Until these gaps close, the correct output is better evidence—not a larger
training run.
