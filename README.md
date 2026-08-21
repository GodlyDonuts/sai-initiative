# Sai Initiative

Sai is Project Shohin's return to its original objective: build the strongest
practical model near four billion parameters. This repository is the live
scratchpad and implementation surface for that effort.

Nothing is called an improvement until it beats the unchanged parent and an
equal-compute control on real, source-disjoint benchmarks.

**Execution status:** the user authorized sub-4B preparation and training on
2026-08-21. The 4B run remains explicitly prohibited until smaller-scale real
benchmark evidence selects an architecture.

## Live scratchpad — 2026-08-21

- Exact FLA 0.4.2 Gated DeltaNet and KDA chunk mechanics remain qualified by
  Newton job `768134`. The environment receipt file is SHA-256
  `778d137224671a44acdcc923270dc7478cded5437780a0ea37e19b764a219f29`.
- The benchmark-decontaminated training corpus, lossless 48K tokenizer, and
  exact binary streams are complete. The selected tokenizer tree is SHA-256
  `cf4879ee5b3914b4af187abcc93be5678e41ff942e0b0a14f6eeb1a089f6f76d`.
  The shared 48,828-sequence training stream identity is
  `b50bb94bc4ada3c5949430222d5551b6dc60423378cacd1f80a57641b1546b22`;
  the source-disjoint 1,024-sequence development stream identity is
  `ec533b1faadea0e0974bfce07923f126be5a2dfe3976b5ab3cf10cf0b43c6dd0`.
- KDA/MLA (`99,594,248` parameters) completed all 191 AdamW updates in job
  `768546` with zero restarts. It consumed 48,828 sequences and 99,831,130
  valid targets. Held-out NLL is `5.64448` per target (`282.73` perplexity,
  `1.20227` NLL/UTF-8 byte). Its immutable result file is SHA-256
  `7ee0fdc6ae229751976a579187e2d931f9c16a95294612a8bcca1de9e8a7c7e8`.
- GDN (`100,019,648` parameters) completed the identical work in job `768529`
  with zero restarts. Held-out NLL is lower at `5.58370` per target (`266.05`
  perplexity, `1.18932` NLL/UTF-8 byte). Its immutable result file is SHA-256
  `accba7dc5728ffa6317a08bd0d61271778d04f44a89568b7bc8b0cd4d60a601b`.
- On the complete 756-row source-disjoint MuSR development population, KDA
  scored `257/756 = 33.995%` and GDN scored `255/756 = 33.730%`. The paired
  GDN-minus-KDA delta is `-0.265 pp`, 95% paired normal interval
  `[-2.726, +2.197] pp`; both are below the `37.099%` uniform-choice baseline.
  This is evidence of no capability separation, not an architecture win.
- The original output-free monolithic MMLU-Pro jobs were stopped after exact
  workload measurement showed 113,990 independent choice forwards. Each full
  12,032-row population is now eight immutable 1,504-row shards: KDA jobs
  `768764–768771` merge through `768772`; GDN jobs `768773–768780` merge through
  `768781`. The shared shard manifest covers every source identity exactly once
  and is SHA-256 `35d714c6ba8f5f2509be3c71e8fc805b4caa54c2c49812315a05dbbcd2e7ba8b`.
  GQA job `768523` is still training; its benchmark fanout is dependency-staged.
  The final comparator reports paired wins/losses, domains, 95% intervals, and
  chance baselines.
- This is a one-seed, approximately 100M-token, iso-data short screen. It is
  not the frozen three-seed iso-data/iso-FLOP tournament and cannot authorize
  the 4B run. The user has authorized sub-4B training; actual 4B training
  remains prohibited pending smaller-scale real-benchmark evidence.

## Current target

- **Name:** Sai
- **Size:** approximately 4B parameters
- **Architecture:** not selected; it must win the scale-gated tournament
- **Reference model:**
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- **Final form:** approximately 4B total parameters, dense, text-only
- **Deployment:** one checkpoint, one pass, one H100 or smaller inference tier
- **Focus:** English, code, math, science, technical reasoning
- **Reasoning:** direct and deliberate behavior in one model; no mandatory
  hidden-draft/revision call

The exact Qwen reference metadata is frozen in
[`docs/SAI_PARENT_QWEN35_4B.json`](docs/SAI_PARENT_QWEN35_4B.json). No model
weights were downloaded or restored while preparing that receipt. Qwen is a
reference and fallback control, not a decision to inherit its full architecture.

## Why this repository exists

Always-revise Shohin failed its first broad public test. Across HumanEval+,
MBPP+, IFEval, MuSR, and CorrectBench, it scored `42.806%` macro versus
`54.022%` for the original and `49.911%` for an equal-compute control. The
`-33.201 pp` MuSR and `-20.839 pp` CorrectBench regressions close mandatory
revision as a route to general intelligence.

Sai starts from those negative results instead of hiding them.

## The SAI shift

The earlier plan was too centered on DeepSeek-R1-era post-training. Reasoning
distillation and RL with verifiable rewards remain useful later, but they cannot
repair a weak base architecture or compensate for lost general capability.

Sai now treats the complete model stack as an empirical tournament. No paper,
company, or fashionable mechanism is promoted directly into the 4B model. Every
change must first beat declared iso-data and iso-FLOP controls at smaller scales
and survive source-disjoint capability and retention tests.

The machine-readable plan is
[`docs/SAI_FRONTIER_ARCHITECTURE_TOURNAMENT.json`](docs/SAI_FRONTIER_ARCHITECTURE_TOURNAMENT.json),
and its rationale is documented in
[`docs/SAI_FRONTIER_ARCHITECTURE_TOURNAMENT.md`](docs/SAI_FRONTIER_ARCHITECTURE_TOURNAMENT.md).
The first executable CPU oracle and exact 48K scale geometries are documented in
[`docs/SAI_MODEL_GENERATOR_CONTRACT.md`](docs/SAI_MODEL_GENERATOR_CONTRACT.md)
and
[`docs/SAI_48K_SCALE_GEOMETRIES.json`](docs/SAI_48K_SCALE_GEOMETRIES.json).
The exact no-training 100M comparison planner is specified in
[`docs/SAI_100M_EXPERIMENT_PLAN.md`](docs/SAI_100M_EXPERIMENT_PLAN.md).
The tokenizer/data specialization workstream is reconciled with that ladder in
[`docs/SAI_4B_SPECIALIZATION_RESEARCH_PLAN.md`](docs/SAI_4B_SPECIALIZATION_RESEARCH_PLAN.md).
The exact lossless 64K/48K/32K measurement boundary is frozen in
[`docs/SAI_TOKENIZER_QUALIFICATION_CONTRACT.md`](docs/SAI_TOKENIZER_QUALIFICATION_CONTRACT.md).
The high-upside conditional-compute thesis and its ordered kill tests are in
[`docs/SAI_ADAPTIVE_COMPUTE_FALSIFICATION_PLAN.md`](docs/SAI_ADAPTIVE_COMPUTE_FALSIFICATION_PLAN.md).
Its exact Gate-0 workspace mechanics and oracle evidence boundary are in
[`docs/SAI_16_SLOT_WORKSPACE_CONTRACT.md`](docs/SAI_16_SLOT_WORKSPACE_CONTRACT.md)
and
[`docs/SAI_ORACLE_SLOW_PATH_CONTRACT.md`](docs/SAI_ORACLE_SLOW_PATH_CONTRACT.md).
Portable checkpoint/run replay and the truthful no-training performance boundary
are in
[`docs/SAI_COMPLETED_RUN_LINEAGE_CONTRACT.md`](docs/SAI_COMPLETED_RUN_LINEAGE_CONTRACT.md)
and
[`docs/SAI_WORKSPACE_PERFORMANCE_CONTRACT.md`](docs/SAI_WORKSPACE_PERFORMANCE_CONTRACT.md).

### Sequence-mixer tournament

The core contest is not “Transformer versus one grand invention.” It is:

1. gated GQA as the conventional reference;
2. a Qwen-inspired `3 Gated DeltaNet : 1 gated full-attention` hybrid; and
3. a Kimi-inspired `3 KDA : 1 gated MLA` hybrid.

KDA, MLA, gated attention, QK normalization, positional treatment, and kernel
efficiency are measured separately before they are combined. AttnRes, SiTU-GLU,
Engram, and multi-token prediction are second-stage ablations, not assumptions.

### Tokenizer and parameter reallocation

The 248,320-token Qwen reference vocabulary contains about 636 million tied
embedding/output parameters at width 2,560. A 32K vocabulary would use about 82
million, freeing roughly 554 million parameters; 48K would free roughly 513
million. Those are hypotheses, not free gains.

Sai will compare 64K, 48K, and 32K English/code/math/science/technical
tokenizers with byte fallback; 16K remains a stress-test only. The tournament
separates two questions:

- does the tokenizer itself improve byte-normalized compression and capability
  with identical body geometry; and
- does reinvesting the saved parameter budget into depth or FFN capacity improve
  the fixed-total-parameter system?

Cross-tokenizer comparisons include an iso-data contrast matched by admitted
UTF-8 bytes and an iso-FLOP contrast matched by the analytical and measured
compute ledger. Multilingual fluency may be deprioritized, but arbitrary Unicode,
identifiers, URLs, source code, math, and scientific notation must remain lossless.

### Conditional memory and objectives

DeepSeek Engram's deterministic n-gram lookup is a compelling partner for a
smaller vocabulary because it could move static phrase memory out of expensive
neural computation. It remains an isolated post-tokenizer ablation. NTP plus one
or two MTP heads is likewise tested independently; future-summary prediction is
exploratory only.

### Behavior-preserving skill learning

After a base architecture wins, Sai post-training may train a narrow adapter on
verified, benchmark-decontaminated math, code, logic, science, technical, and
instruction data. Every optimizer window also replays broad selected-base
behavior. The candidate minimizes task loss plus frozen-base token KL. The
equal-compute control executes identical forwards with KL weight zero.

### Reasoning without compulsory verbosity

After the base architecture wins, post-training may use verified multi-teacher
distillation and RL with verifiable rewards for math, code, formal logic, and
tool use. Direct-response examples remain in the same mixture. Only an SFT
checkpoint that survives the public gate may enter bounded outcome-based RL.

Long reasoning is a selectable inference mode, not a ritual imposed on every
prompt. Fixed-direct and fixed-deliberate controls must show that adaptive
compute actually helps.

## Scale ladder

The generator must instantiate comparable models at approximately 100M, 300M,
1B, and 4B parameters. After an official training order, the sequence is:

- 100M: mechanics, stability, kernel, memory, and throughput qualification;
- 300M: three-seed factor screens on frozen development data;
- 1B: confirmation of only the surviving factors and interaction checks; and
- 4B: one selected stack, followed by the complete public gate.

The 4B run is prohibited until the smaller-scale evidence exists. This is the
lesson from Shohin made executable: benchmark evidence chooses the architecture.

## First public gate

The complete official HumanEval+, MBPP+, IFEval, MuSR, and CorrectBench boards
are conjunctive. A candidate must:

- beat original and equal-compute macros by at least `1.0` point;
- remain within `1.0` point of both comparators on every benchmark;
- beat each comparator on at least four of five benchmarks; and
- be nonnegative against both comparators on MuSR and CorrectBench.

One serious regression vetoes a favorable average.

## Build status

- [x] retire always-revise compute and free every GPU request;
- [x] encode the five-board benchmark gate and historical falsification;
- [x] prototype frozen-parent replay KL with a matched zero-weight control;
- [x] begin a lossless tokenizer-capacity auditor;
- [x] implement deterministic, benchmark-disjoint freezing for skill, direct,
  deliberate, replay, and RL-prompt banks;
- [x] replace the R1-centered plan with a verified 2026 architecture tournament;
- [x] freeze the 100M → 300M → 1B → 4B promotion ladder and factor isolation;
- [x] implement causal CPU reference mixers and exact parameter ledgers;
- [x] freeze matched 48K geometries for all three families at every scale;
- [x] prove packed-document isolation across attention, RoPE, convolution, and
  recurrent state in the CPU architecture oracle;
- [x] implement an exact three-family, three-seed iso-data/iso-FLOP planner;
- [x] implement deterministic binary token packing with exact UTF-8 prefix and
  cross-document boundary receipts;
- [x] implement the exact 64K/48K/32K tokenizer qualifier and protected Unicode
  suite without building or selecting a candidate;
- [x] define an oracle-first falsification ladder for latent workspace,
  fixed-point recurrence, regret gating, and sparse semantic memory;
- [x] implement exact 16-slot workspace accounting, a bitwise fast bypass, and a
  row-level equal-FLOP oracle analyzer without training;
- [x] make oracle evidence reopen portable checkpoint/run lineage and add a
  mutation-free CPU workspace performance receipt without claiming H100 speed;
- [x] implement deterministic 64K/48K/32K tokenizer construction, exact stream
  loading, masked AdamW training, held-out NLL evaluation, and atomic resume;
- [x] qualify GDN/KDA FLA chunk forward/backward mechanics and complete stable
  full-model optimizer updates for both frozen 100M delta-family geometries;
- [x] implement replayable word/code benchmark decontamination and pin the
  FineWeb-Edu mechanics source prefix;
- [x] run that freezer on the exact admitted source populations;
- [x] qualify 64K/48K/32K tokenizer candidates on the admitted corpora;
- [ ] freeze matched data/FLOP/seed manifests for the 100M tournament;
- [ ] run the 100M mechanics tournament, then evidence-gated 300M/1B stages;
- [ ] package exactly one winning 4B architecture and matched controls;
- [ ] run all five complete public boards;
- [ ] promote only if every gate conjunct passes.

## Repository layout

- `src/sai/gates/` — real-benchmark promotion decisions
- `src/sai/model/` — scalable configurations, parameter ledgers, and CPU oracles
- `src/sai/data/` — verified role populations and contamination filtering
- `src/sai/training/` — behavior preservation and reasoning training
- `src/sai/tokenizer/` — vocabulary capacity measurement and surgery
- `tests/` — fail-closed regression coverage
- `docs/` — frozen contracts and experimental evidence

Historical Shohin evidence remains in
[`GodlyDonuts/shohin-ettr`](https://github.com/GodlyDonuts/shohin-ettr).
Sai-specific implementation and results live here.
