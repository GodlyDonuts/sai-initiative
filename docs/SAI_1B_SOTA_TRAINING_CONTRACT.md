# Sai 1B SoTA Training Contract

Status: active preparation. Model training has not started.

## Objective

Sai now targets one dense billion parameters, not four billion. The objective
is the strongest practical English-first polymath at this scale. “SoTA” is an
evaluation target, never a pretraining claim: the resulting checkpoint must
beat prospectively frozen public baselines on complete, source-disjoint boards.

The production corpus is already signed at 92,929,510 rows and
2,050,874,912,388 logical UTF-8 bytes. This contract turns that corpus into a
deterministic teaching sequence without changing its admitted content.

## Token horizon

The full target is **4,000,000,000,000 tokens at sequence length 4,096**. It is
exactly 976,562,500 packed sequences. This is an ambitious but evidence-based
minimum for a frontier 1B attempt:

- AllenAI's transparent OLMo 2 1B recipe trains stage 1 for 4T tokens and then
  performs a 50B-token high-quality stage 2:
  <https://github.com/allenai/OLMo#pretraining>.
- The 2026 Mula-1B report independently trains its dense 1B model for 4T tokens:
  <https://arxiv.org/abs/2604.00785>.

The comparison establishes precedent, not guaranteed optimality. Sai records
public evaluation milestones at 1T, 2.4T, 3.4T, 3.8T, and 4T so training can be
stopped when measured marginal gain no longer justifies compute. The schedule
does not require blind completion of all 4T tokens.

## Moving-center spiral

The earlier 8T concept is scaled to the 1B horizon while preserving its stage
fractions and both difficulty tails.

| Stage | Token interval | Foundation | Intermediate | Advanced | Expert |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundation | 0–1.0T | 65% | 25% | 8% | 2% |
| Expansion | 1.0–2.4T | 40% | 40% | 15% | 5% |
| Depth | 2.4–3.4T | 20% | 40% | 30% | 10% |
| Synthesis | 3.4–3.8T | 10% | 25% | 40% | 25% |
| Annealing | 3.8–4.0T | 10% | 20% | 35% | 35% |

Foundation never reaches zero, expert material appears from the first stage,
and every stage boundary aligns with a 4,096-token sequence. Integer sequence
allocations use deterministic largest-remainder rounding.

The first complete index replay (`r1`) proved that treating length as a proxy
for conceptual difficulty was invalid: it assigned only 2,387,625,863 train
source tokens to foundation while assigning 460,466,425,296 to
advanced/expert. That release is rejected for training because the 65%
foundation stage would have replayed a tiny basic pool hundreds of times.

Production `r2` instead uses source-informed exposure tiers with a stable
identity-hash distribution inside each source family. The hash only balances
the pool; it never changes source custody, split, content, or rights. Broad
reference/narrative sources populate foundation and intermediate tiers;
research, patent, legal, and code sources remain concentrated in advanced and
expert tiers. Document length affects only within-tier ordering and never
promotes a document to a harder conceptual tier. The production index binds:

- Institutional Books topic, genre, length, OCR, and rights metadata;
- PleIAs collection, open type, word count, source-token count, license, and
  content identity;
- Stack-Edu language, educational score, length, licenses, and content identity;
- the verified connection overlay's explicit difficulty, prerequisites,
  domains, split, and transfer receipt.

This is an exposure curriculum, not a claim that every web document has a
perfect scalar pedagogical difficulty. Its first quantitative gate is that
every tier has enough distinct train text for the physical window without a
degenerate repetition ratio. The exact `r2` ledger must pass that gate before
packing.

## Exposure policy

Curriculum bands and source components are separate axes. Band weights move the
center of gravity; source weights preserve intellectual breadth. The sampler:

- draws without replacement inside a component/band epoch before repeating;
- reshuffles with a counter-based hash keyed by stage, epoch, and document;
- never admits the 168 held-out connection documents;
- holds out a deterministic 0.1% internal development partition from every bulk
  component before packing;
- caps the tiny verified connection overlay at 16 exposures per document over
  the entire run;
- records exact per-component and per-band exposure counts;
- prohibits silent reweighting when a requested stratum is exhausted; and
- keeps broad foundation rehearsal in the final annealing stage.

Repeated data is explicit. A 4T run necessarily revisits the roughly
half-trillion-token unique corpus. Repetition is organized across curriculum
epochs rather than pretending duplicated exposures are new information.

## Tokenizer

The production capacity is fixed at **48,000 lossless byte-level BPE tokens**
with untied input/output embeddings and explicit pad/BOS/EOS/think/code tokens.
The old 48K
tree is only a mechanically qualified default because its tournament population
underrepresented code, mathematics, science, and technical text. A new 1B
production tree must be trained on a stratified sample of all three released
components plus the complete connection overlay, then pass:

- byte-exact round trips;
- contiguous vocabulary IDs and exact special-token identity;
- the protected English/code/math/science/technical/Unicode suite;
- per-domain fertility accounting; and
- an exact tree/hash custody receipt.

This contract chooses capacity and pretokenization for launch. It does not claim
that 48K is a universal empirical winner.

## Production model and trainer

Sai uses the published OLMo 2 1B design as the conservative architecture
baseline: 16 reordered-norm Transformer blocks, width 2,048, 16 attention
heads, QK normalization, 500,000-theta RoPE, SwiGLU width 5,504, FlashAttention,
no bias or dropout, and untied embeddings. With Sai's 48K vocabulary this is
1,006,241,792 parameters. The architecture is deliberately not presented as a
Sai invention. It prevents another unvalidated mixer from confounding the data
and curriculum experiment.

The distributed training runtime is pinned to AllenAI OLMo commit
`090253dac6688f2532509daa7aa2eb5fae50e956` and OLMo-core commit
`b7e9671d7ea48af94838c4f124703c3ae36f0c70`. The optimizer follows the
published 1B stage-1 recipe: AdamW at 4e-4, betas 0.9/0.95, weight decay 0.1,
8,388,608,000 warmup tokens, cosine decay to 10% of peak, BF16 AMP, and FSDP
`SHARD_GRAD_OP`. The ordinary global batch is 512 sequences; exact stage and
terminal boundaries use a smaller deterministic final batch rather than
silently crossing a curriculum boundary.

## Launch-readiness gates

Training may be reported ready only after all of the following exist and replay:

1. the signed three-component data release;
2. the final 48K tokenizer tree and qualification receipt;
3. the complete source-text-free curriculum index with all 92,929,510 source
   identities accounted or explicitly held out;
4. an exact 4T counter-based exposure plan and stage ledger;
5. a packed-stream smoke receipt proving document-boundary isolation,
   losslessness, resume identity, and no development-row leakage;
6. a clean immutable runtime, environment hash, quota/accounting receipt, and
   measured H100 memory/throughput preflight that performs no optimizer update;
7. an explicit user order to start 1B training.

Readiness does not mean the model is already SoTA, that exhaustive benchmark
decontamination of the practical bulk corpus is complete, or that training has
started.
