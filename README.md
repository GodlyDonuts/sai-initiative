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
- Gated GQA (`100,481,024` parameters) completed the same 191 updates,
  48,828 sequences, and 99,831,130 valid targets in job `768523` with zero
  restarts. It is slowest and has the worst held-out NLL: `5.65844` per target
  (`286.70` perplexity, `1.20524` NLL/UTF-8 byte). Its immutable result file is
  SHA-256 `592bf31ab880b532bb230016e17b77052578fd84da911add5fa86e9a8147afd6`.
  Its eight independent MMLU-Pro shards `768911–768918` and merge `768920`
  completed without retries: `1,101/12,032 = 9.1506%`, also below the exact
  `11.0877%` uniform baseline. The merged result file is SHA-256
  `0ba400ec78a89f86fcfd002897a45a8ad6529ff3efab32d1c506aad1544f9001`.
  MuSR job `768919` completed at `256/756 = 33.862%`, also below the exact
  `37.099%` baseline; its result file is SHA-256
  `40d5638157f3180f2e06ef61bcbca4a34215138ae498ad942cbdc3541239e8bd`.
- On the complete 756-row source-disjoint MuSR development population, KDA
  scored `257/756 = 33.995%` and GDN scored `255/756 = 33.730%`. The paired
  GDN-minus-KDA delta is `-0.265 pp`, 95% paired normal interval
  `[-2.726, +2.197] pp`; both are below the `37.099%` uniform-choice baseline.
  This is evidence of no capability separation, not an architecture win.
- The original output-free monolithic MMLU-Pro jobs were stopped after exact
  workload measurement showed 113,990 independent choice forwards. Eight
  immutable 1,504-row shards per model completed without retries: KDA jobs
  `768764–768771` merged through `768772`; GDN jobs `768773–768780` merged through
  `768781`. The shared manifest covers all 12,032 identities exactly once and is
  SHA-256 `35d714c6ba8f5f2509be3c71e8fc805b4caa54c2c49812315a05dbbcd2e7ba8b`.
  KDA scored `1,133/12,032 = 9.4166%`; GDN scored
  `1,113/12,032 = 9.2503%`. The exact variable-choice uniform baseline is
  `11.0877%`. Paired GDN-minus-KDA is `-0.166 pp`, 95% interval
  `[-0.533, +0.200] pp`. Both recurrent candidates fail the real-capability
  screen and are not promoted. Their immutable merged result files are SHA-256
  `06959cdbd5a038870fcc5a9da58e5ce49c8edc63a526f7265b3bd3e30d42b4d6`
  and `830261b871521297412b589d9ef20371449043bf0fa31292445c13babf3ae04e`.
  GQA is worse on the complete MMLU-Pro board; no family has cleared the
  capability floor at this token budget. The full comparison file is SHA-256
  `1558265380edeb701492585477fb16494b280e6c29aa8139d34650123dfef708`.
  The predeclared decision receipt is SHA-256
  `87b0a0c13eda2a1ab02ef898a48927af51d1dc7346f3d47791cf05ec889b6ed3`
  and returns `no_family_capability_qualified_data_extension_only`: no mixer is
  promoted, and the longer run is strictly a data-starvation diagnostic.
- A separate architecture-independent 30-file FineWeb-Edu prefix is now pinned
  for the next 300M factor screen. It covers exactly `64,562,434,300` raw
  upstream bytes at revision `87f09149…b8f9`. Attempt `768858` failed before
  download because ordinary Newton CPU allocations omit `SLURM_TMPDIR`; no data
  was written. Corrected job `768891` then scanned exactly 21,855,000 documents
  and admitted 2,661,644 into a 14,491,695,743-byte source whose SHA-256 is
  `2f908f5f225de109a21f66fb9fb31baa1f35b4a57f1d6d2a3f60fa95a98ea7e6`.
  Exact decontamination `768892` is running. At 19:30 EDT its live source file
  descriptor was at `4,442,947,584 / 14,491,695,743` bytes (`30.66%`), its
  maximum RSS remained flat at `65,195,780 KiB`, and it had zero restarts. A
  disk-read heuristic initially overstated progress; the process file
  descriptor is the authoritative scan measurement. Newton rejected a running
  wall-time extension, so conditional CPU fallback `769225` is held on
  `afternotok:768892` with compact, byte-equivalent SHA-256 shingle storage at
  commit `540d1080ebc5b1cd2b463d8c137e9d2c71567e82`. It cannot execute while
  `768892` remains healthy and will be invalidated if `768892` succeeds. Its
  four direct CPU descendants are also dependency-staged but ineligible:
  shared Sai stream `769226`, refreshed development populations `769227`, Qwen
  stream `769228`, and SmolLM3 stream `769229`. This prevents a manual recovery
  gap without requesting a GPU or duplicating healthy work. The mutually
  exclusive downstream recovery graph is also complete: 100M/250M launcher
  `769232`, parent benchmark launcher `769235`, workspace launcher `769236`,
  workspace evaluation stage `769237`, and comparison stage `769238`. Two
  malformed first launcher clones (`769233–769234`) were caught while still
  dependency-held and canceled with zero elapsed time and zero restarts; their
  corrected replacements bind the exact fallback population and stream job
  IDs. No recovery job is eligible while the primary path remains healthy.
  Successful decontamination is followed by 499,998,720-token stream freeze
  `768894`. Dependency-staged launcher `768932` will then run a fresh,
  matched three-family screen over the exact 249,999,360-token prefix using
  three independent one-H100 jobs. This tests whether the near-chance 100M
  results reflect data starvation or the mixers themselves; it neither selects
  a mixer in advance nor authorizes 4B.
- The exact pretrained capable-host control is now restored at
  `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` without using a
  GPU. CPU attempts `769141` and `769144` failed before publication on a Bash
  digit-separator bug and one transposed `README.md` Git-blob pin; both output
  roots remained absent. Corrected job `769148` completed in 33 seconds with
  zero restarts. Independent replay verifies all 13 upstream members, including
  the `1,746,942,600`-byte weight SHA-256
  `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
  The sealed 15-member local tree contains no links or writable members; its
  tree identity is
  `24457a397ecf57057d29636ace857c78f0983cb25647b24f10461bb4943e875d`
  and its receipt is
  `3c7d25ab4d4bcf4dec81b594f8919636483bd64607ec1ae506c76e6ba815e00b`.
  This is host preparation, not a Sai architecture result.
- The first capable-host Sai factor is now fully executable at commit
  `cc7039d1e5a0653f4581cbe1a7b3ce509fff58e6`: a `19,938,304`-parameter,
  16-slot recurrent workspace attached to the frozen Qwen3.5-0.8B text parent.
  Its matched `reset_average` control has identical parameters, initialization,
  optimizer, data, compiler/reactor/reader calls, and modeled workspace FLOPs;
  the sole change is whether reactor state carries across the two iterations.
  Each packed document is passed through the frozen parent exactly once, so no
  cross-document context leaks into the objective and all eight probe positions
  reuse the same detached causal hidden states.
- Exact parent H100 mechanics job `769161` and exact Qwen-tokenized 499,998,720-
  token stream job `769174` are the only unresolved prerequisites. Dependency
  launcher `769193` will request two independent one-H100 256-sequence canaries,
  then two independent one-H100 61,035-sequence full arms only if both canaries
  pass. No 4B model is involved.
- Parent development launcher `769171` and workspace evaluation stage `769194`
  are wired to the same refreshed, 500M-source-disjoint full MMLU-Pro and MuSR
  populations. Each workspace arm fans out as eight independent one-H100
  MMLU-Pro shards plus one independent one-H100 MuSR job. Comparison stage
  `769196` will compute exact paired row deltas and 10,000-replicate stratified
  intervals against both the unchanged parent and matched reset control. Its
  pass can authorize only another sub-4B confirmation; it cannot authorize the
  4B run. These are staged experiments, not positive architecture results.
- The cross-size/cross-family confirmation host is also pinned in advance, but
  remains unscheduled: `HuggingFaceTB/SmolLM3-3B` revision
  `a07cc9a04f16550a088caea529712d1d335b0ac1`, with `3,075,098,624` text
  parameters, a `6,167,865,576`-byte sealed tree, and tree SHA-256
  `6badcd593aee3052e3d66afb315b979e2cc62c4a61f9cef31c07203912478a0f`.
  Sai reopens its exact external manifest, receipt, and every weight member; a
  fresh CPU replay passed all 12 members and all `6,167,865,576` bytes. CPU-only
  first stream job `769203` was cancelled before allocation (zero elapsed,
  zero restarts) when preflight found that Hugging Face's base `vocab_size`
  excludes Smol's 256 added tokens, including EOS `128012`. The packer now
  binds the full tokenizer length `128256`; corrected CPU job `769208` is
  staged after decontamination to freeze the same 499,998,720-token source
  under the exact Smol tokenizer. The one-H100
  no-training mechanics entry point remains unscheduled. Commit `d58937c`
  additionally prepares—but does not launch—the same recurrent-vs-reset factor
  on this host: a proportional `79,722,496`-parameter, 16-slot workspace with
  identical initialization, optimizer, source prefix, objective, calls, and
  modeled workspace FLOPs across arms. The sole changed factor remains reactor
  state carry. This host will be used only if the 0.8B recurrent factor passes;
  preparing it is not a result and does not consume the terminal 4B boundary.
- Thirteen obsolete Q36 score jobs (`759843`, `759860`, `760174`, `760180`,
  `760185`, `760187`, `760194`, `760201`, `760206`, `760208`, `760215`,
  `760216`, and `760217`) were terminally cancelled after each was proven held
  on already-failed August 14–15 dependencies. Every job had zero elapsed time
  and zero restarts. This released submission slots for Sai's independent
  benchmark shards without cancelling or changing any live Sai allocation.
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
