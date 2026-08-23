# Sai Initiative

Sai is Project Shohin's return to its original objective: build the strongest
practical model near four billion parameters. This repository is the live
scratchpad and implementation surface for that effort.

Nothing is called an improvement until it beats the unchanged parent and an
equal-compute control on real, source-disjoint benchmarks.

**Data precedes architecture.** Sai first earns a trustworthy learning
sequence: verified source bytes, quality and duplication controls, explicit
prerequisite coverage, gradual difficulty, later rehearsal of fundamentals,
and source-disjoint evaluation. A sophisticated mixer cannot compensate for
bad examples or for teaching dependent concepts before their primitives. A 4B
run may now be used as the decisive architecture experiment once its data is
qualified, but the architecture is not called successful until that checkpoint
beats the unchanged and equal-compute baselines on real benchmarks.

Sai treats data admission as three separate scientific questions:

1. **Quality:** is each example accurate, coherent, useful, legally and
   operationally admissible, and free of benchmark contamination, spam, answer
   farms, and high-confidence duplication?
2. **Coverage:** does the mixture teach the English, mathematics, code, science,
   technology, reasoning, and communication capabilities we intend to measure,
   without allowing one abundant source to crowd out the rest?
3. **Pedagogy:** are primitives taught before dependent concepts, are new ideas
   composed gradually, and are foundations rehearsed after difficulty rises?

These gates cannot substitute for one another. Clean advanced text is still a
bad first lesson when its prerequisites are absent. A readable document is not
necessarily foundational. A source with a high upstream quality score is not
automatically a Sai training source. The target is not merely an easy-to-hard
sort; it is a measured semantic learning progression whose exact records and
ordering beat a same-record order control before scaling.

### Sai data constitution

Data decisions precede tokenizer, architecture, and scale decisions.

The latest open-recipe evidence is reconciled in
[`docs/SAI_2026_DATA_RESEARCH_SYNTHESIS.md`](docs/SAI_2026_DATA_RESEARCH_SYNTHESIS.md).
Sai uses progressive composition with continuous broad rehearsal, retains
low-dose foundational code/math/science/technical exposure early, and measures
specialist upsampling, reasoning mid-training, and long-context adaptation as
separate factors rather than copying another model's ratios.

The corpus objective is stated more directly in
[`docs/SAI_POLYMATH_DATA_THESIS.md`](docs/SAI_POLYMATH_DATA_THESIS.md). Web is a
candidate reservoir rather than a protected percentage. The final pool must
cover human expression, literature, arts, history, philosophy, society,
everyday life, reference knowledge, mathematics, science, engineering, and
software, and explicitly reward accurate cross-domain bridges. Selection is
driven by marginal concept, style, and capability coverage under a spiraled
prerequisite curriculum--not by copying another model's 75--85% web ratio.

The executable replacement for dataset concatenation is described in
[`docs/SAI_DATA_COMPILER.md`](docs/SAI_DATA_COMPILER.md). It treats global raw
sources as reality anchors, chooses preservation and English-translation policy,
derives grounded representations, protects form-bearing human expression,
constructs verified cross-domain and procedural reasoning, and leaves final
sampling to a coverage- and model-responsive curriculum controller.

The first large book reservoir is now pinned in
[`docs/SAI_INSTITUTIONAL_BOOKS_COMPILER.md`](docs/SAI_INSTITUTIONAL_BOOKS_COMPILER.md).
Its 983,004 Harvard Library volumes are screened metadata-first rather than
blindly downloaded. Sai separates archive facts from model inferences, measures
linguistic/conceptual/reasoning complexity independently, builds cited
prerequisite edges, and uses a spiral schedule that never stops rehearsing
fundamentals. Valuable non-English technical works route to English; literature
must use a reputable human translation or separately labeled literal and
literary synthetic translations. Dataset terms and per-volume rights evidence
remain independent admission gates.

The first metadata-only book queue now contains 10,000 duplicate-safe volumes
across 772 language×subject cells: 9,409 non-English and 591 English. This is a
translation-discovery workload, not a fixed training ratio. Authenticated access
has now replayed the first exact enriched-text shard and built a separate
185-volume compiler pilot. The gated text remains local under the pinned IDI
terms; it is not redistributed through GitHub or Hugging Face.

The durable data catalog is
[`Godlydonuts/Sai`](https://huggingface.co/datasets/Godlydonuts/Sai). It separates
upstream source references, model judgments, verified representations,
curriculum artifacts, and final training shards. The registry never treats a
downloaded dataset as training-ready and records reference-only sources without
copying bytes when their terms prohibit redistribution.

### The eight-trillion data program

Sai is now executing two related but deliberately separate programs:

1. **Hash-bound source-candidate reservoirs.** Two immutable inventories now
   reference 21.537 physical TiB of reality anchors and candidate material.
   They provide breadth and filtering headroom; neither is itself a training
   set.
2. **A prospective 8T-token training curriculum.** This is the maximum-horizon
   schedule for turning qualified material into a moving-center spiral. It is a
   curriculum contract, not authorization to train and not a claim that eight
   trillion accepted tokens already exist.

Conflating these two quantities would hide the most important work. Raw bytes
become training tokens only after rights checks, normalization, exact and
near-duplicate removal, benchmark decontamination, quality judgment, concept
and prerequisite annotation, grounded transformation, and final replay. A
source can be excellent and still require a different representation or a
later curriculum position.

#### Exact source-reservoir checkpoint

Reservoir v2 was sealed on 2026-08-23 from exact Hugging Face revisions. It
contains **16,001 files and 8,796,890,808,426 bytes**
(**8.0007255823 TiB**), exceeding the exact 8 TiB target by 797,786,218
bytes. The selection includes every file from the specialist sources and only
the minimum deterministic, path-ordered FineWeb-Edu prefix needed to cross the
target.

| Source | Exact revision | Files | Selected bytes | Function |
| --- | --- | ---: | ---: | --- |
| FinePDFs | `220bac3acbf07789502c621d2d33952f51ac7f86` | 3,573 | 5,375,642,953,643 | Global PDF reality anchors |
| Institutional Books enriched text | `92fcdf938eb87edfe0fbf09d4f692fa3d8bc9bcd` | 4,916 | 870,263,633,412 | Books, human expression, and historical knowledge |
| FineMath | `e92b25a616738fe95dc186b64dfb19f9c8525594` | 288 | 149,447,371,427 | Mathematical reality anchors |
| Dolma 3 mix-150B | `afa92bfb22366821c5e6cd427cdd036b34b713ef` | 6,081 | 110,586,325,507 | Broad multidomain reality anchors |
| SmolLM corpus | `3ba9d605774198c5868892d7a8deda78031a781f` | 338 | 672,430,417,560 | Curated educational web and synthetic textbooks |
| OpenWebMath | `fde8ef8de2300f5e778f56261843dab89f230815` | 114 | 27,431,041,597 | Mathematical exposition |
| FineWeb-Edu deterministic fill | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | 691 | 1,591,089,065,280 | Broad educational-web coverage after specialists |

The manifest is 7,808,445 bytes with file SHA-256
`36d41d579511af2479281b3199f242d5d039ec8a261c2e3d37b07354ea6d7ccf`
and ordered-row SHA-256
`3e1e95121ef44b0c87d430bbd46356cd978ede8719328a7416923a95a64c4e18`.
The receipt SHA-256 is
`38e777da2a81e90919d4404d00d2a8e17531e8e0aa1405424ec645e4cdaddf44`.
The manifest and receipt were replayed byte-for-byte after publication in
Hugging Face dataset commit
[`8199183b8064ffd0c0b3748bdb40a90a10da2b23`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/8199183b8064ffd0c0b3748bdb40a90a10da2b23).
Every source revision was resolved exactly, an object from every source was
access-probed, and every selected file is bound to its upstream LFS SHA-256.

The v2 correction came from actual schema replay: SmolLM's two `python-edu`
files contain blob identifiers, repository names, paths, lengths, and scores,
but no code text. Sai excludes those metadata-only bytes instead of counting
them as code training material. Code coverage remains present through Dolma's
content-bearing Stack-Edu shards; it is not fabricated from an index.

This checkpoint does **not** claim that 8 TiB is locally downloaded, unique,
licensed as one combined corpus, quality-approved, translated, decontaminated,
or training-ready. Institutional Books remains authenticated,
reference-only material under its pinned early-access terms. The Stack v2 and
manual-gated sources are excluded until their exact bytes can be accessed
lawfully and reproducibly. Source inclusion also does not establish a final
training percentage: origin is metadata, not an epistemic function or a fixed
mixture allocation.

#### Modern-source augmentation checkpoint

The first semantic results exposed an unavoidable arithmetic fact: an 8 TiB
raw reservoir cannot yield 8 TiB of finished data after rejecting damaged,
duplicated, unsafe, low-value, or rights-incompatible material. Sai therefore
sealed a second, source-only frontier inventory on 2026-08-24. Frontier v3
contains **26,599 files and 14,883,185,490,335 physical bytes
(13.5361783490 TiB)** from modern, independently curated source families.
Together, the two reservoirs reference **23,680,076,298,761 physical bytes
(21.5369039313 TiB)** before cross-reservoir deduplication and quality
compilation.

Those bytes are now independently accounted under source-safe conversion
ledger release r6 receipt
`1970d1ecdb70ccc6f554ee9927df49eb2b42d15ed3a516bc955862748a545fb6`.
The ledger hash-verifies both reservoir manifests, all six immutable audit
populations containing 2,103 rows, corrected rights-inventory v2, and both exact
bounded text-payload probes. Duplicate audit or probe receipts are rejected. Its
current funnel is deliberately blunt: 21.5369 TiB referenced candidates, 2,103
acquired audit rows, 17,638,716,209 mechanically useful bytes measured in nine
bounded members, two completed source pilots containing 3,290 bounded
near-deduplicated rows, and **0 training-ready bytes**.
The bounded measurement is not extrapolated to the reservoir. Of the candidate bytes,
7,899,196,133,417 require declared-license
obligation handling, 5,027,859,142,584 require per-row license evidence, and
10,753,021,022,760 require source-terms resolution. Cross-inventory overlap and
full-reservoir text-payload yield remains unresolved, so the candidate-byte sum cannot
be used as a training-data claim. The text-free r6 receipt (file SHA-256
`36b2dcc3542b9a282eb4836fcf1d64883d3c75624c33ab25464998bca007cb86`)
was uploaded and replayed byte-for-byte in Hugging Face commit
[`f5ec9e07e987f008c52a29b31922c2e361c8472a`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f5ec9e07e987f008c52a29b31922c2e361c8472a).
Earlier ledger releases remain immutable historical evidence.

| Candidate slice | Exact revision | Files | Physical bytes | Intended comparison |
| --- | --- | ---: | ---: | --- |
| Ultra-FineWeb current English L2 (2026-08-20 slice) | `02c85641e3d19a854be2e09139c25adaa9518063` | 6,000 | 477,974,475,357 | Newest model-selected English web |
| Ultra-FineWeb benchmark-validated English L2 | `02c85641e3d19a854be2e09139c25adaa9518063` | 2,048 | 2,661,358,122,836 | Earlier L2 generation with published proxy evidence |
| FineWeb2-HQ multilingual | `c0c06e94fd3a44ae9e802b2b0fc533817601eb5e` | 5,891 | 6,042,406,965,380 | Twenty-language high-value translation discovery |
| Nemotron specialized reasoning | `9ed3718b5f2ae29074c5e34e64115432b7c4320f` | 219 | 244,286,609,368 | RQA, InfiniByte, math textbooks, and scientific coding |
| UltraData-Math L1 selected slice | `fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87` | 1,485 | 366,622,518,811 | Filtered and deduplicated mathematical source material |
| PleIAs Common Corpus | `307910e4c5d040d6f318e6edf2a2b97849155771` | 10,000 | 4,489,486,652,558 | Traceable open-license and public-domain global reality anchors |
| Nemotron Specialized v1.2 | `807afc1fa65c441d46ebc7d9b95295a35499a527` | 90 | 53,621,158,028 | Fact seeking, generative tasks, moral scenarios, and multiple choice |
| Nemotron Legal v1 | `3d91d58a5c0c46fe9944300ec46719f97a385b13` | 21 | 6,990,697,508 | Primary law and legal reasoning |
| Common Pile filtered collection | 31 exact repository revisions | 845 | 540,438,290,489 | Courts, government, patents, science, books, education, code, reference, and culture |

#### UltraData Math L2/L3 quality audit

Sai does not treat a provider's tier name as evidence of quality. On 2026-08-25
it therefore acquired a deterministic **160-row** screen from the exact
[`openbmb/UltraData-Math`](https://huggingface.co/datasets/openbmb/UltraData-Math)
revision `fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87`: 32 rows from the
33.7B-token L2 quality-selected tier and 32 rows from each of the four L3
synthetic formats (conversation, multi-style, question-answer, and
textbook-exercise). Every official dataset-server response returned that exact
revision in `X-Revision`; 20 response hashes and 160 source identities are
sealed under population receipt
`d62c4bf9711135f8d3d2aabbbeca4891c7e02eba3ec799974d3dbc65197cd138`.

The active official-public benchmark boundary rejected **12/160 rows** before
quality scoring: 3/32 L2, 0/32 conversation, and 3/32 in each remaining L3
format. The benchmark-disjoint population contains **148 rows** under receipt
`54cfe887c504bb1415fc5fd803eaea7438cc40a4ee0f7882b279f07e31d3afe3`.
Hermes compilation is dependency-staged behind the two already-running source
populations, so these 121.7B published upstream tokens remain source candidates,
not accepted or training-ready Sai tokens. Raw sampled text and individual
contamination decisions remain local. The source-safe publication envelope,
population receipt, and 20 batch-custody receipts were uploaded and replayed
byte-for-byte in Hugging Face dataset commit
[`28227ed9ba5a22887c2a0bb3bee20502e0982253`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/28227ed9ba5a22887c2a0bb3bee20502e0982253).

#### Complete Python Enhancement Proposal census

The 32-row source-disjoint Common Pile confirmation recovered the Python
Enhancement Proposal lane as a narrow but unusually traceable reality anchor:
32/32 rows received `retain` judgments, 27/32 were routed to representation
verification, 2/32 were quarantined, all 32 carried the recognized
`LicenseRef-Public-Domain` declaration, and none overlapped the active
benchmark boundary. That evidence authorizes a complete filtered census of
one pinned parent; it does not authorize source-wide quality or training.

Sai downloaded the exact 3,723,467-byte
`common-pile/python_enhancement_proposals_filtered` parent at revision
`582170907dd303c207770fceacd38e6abf133edc` and verified compressed SHA-256
`4bb61eded5168ac7f0059a92ed242577c67e4fced8c0d019c84bfaca5596c791`.
The exhaustive pass scanned **655 rows**, removed 36 identities already used
in audit populations and one mechanically short row, then screened all 618
remaining rows against the official-public benchmark boundary. It rejected
50 overlapping rows, near-deduplicated the 568 clean rows into **567 unique
candidates**, and joined every survivor to an exact public-domain attribution
record. The compressed parent was removed after the one-host census.

All 567 survivors now form a create-only Hermès compiler population with
8,822,685 excerpt bytes and complete text-free lineage. Compilation is
dependency-staged after the already-running UltraData tier audit so the proxy
is not overloaded and no identity is scored twice. The census receipt is
`c1f18f641a31672cda7d2b10caf60769df766aa7edea62418d1089645319c92b`,
the compiler-population receipt is
`255e9aa09ec8d2f00c01db05b8eabb6bd06d9f93c77d59b1ed6e8ce2caf7a5ba`,
and the source-safe publication envelope is
`3eeb07c28d575542d87a670396452a155f0c97aaaeaa15f19d526f210568168e`.
Source text, individual decontamination decisions, and machine-local paths
remain unpublished. The 567 candidates remain non-training-ready until
Hermès quality compilation and representation verification close.

The corrected reservoir rights inventory now binds **46 source lanes, 45 exact
repository revisions, 42,600 files, and all 23,680,076,298,761 candidate
bytes**. Five lanes have an exact manifest declaration with obligations, 31
Common Pile lanes require per-row evidence, and ten require source-terms
resolution. A permissive wrapper card can no longer override manifest labels
such as “with upstream terms” or “generator terms.” StackV2 HTML has no
`README.md` in its exact pinned tree; that absence is recorded rather than
substituted from another revision. Corrected receipt
`8e72391081af17323aa1e1b8d0480ddbe70dcb232006e6cf37ed7228d34d3d80`
contains no source text and establishes no legal clearance. Its remote bytes
replayed exactly in Hugging Face commit
[`b7b60404ab737b9fd1e44740f6f781dc8d56da38`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/b7b60404ab737b9fd1e44740f6f781dc8d56da38).
The earlier 11/31/4 routing receipt is retained as superseded audit history.

The 16,200,072-byte v3 manifest has SHA-256
`0a59e8a24208f8593f806b919a65a3e3e64d911936f137286235c31627f56ebd`
and ordered-row SHA-256
`510ddd35f23a00474d1ba5e6468f65bfd44e82cf33b98c8ed9870e85bf744819`.
Its canonical receipt is
`5c3423c8d473a6155f6c402deeef298f95e89cc3769c968ced6da79f24d488a1`
and the receipt file SHA-256 is
`3a4a3169a8fbb75bb805dc8726c2e15e96bd21a0958f1b9bf1e2585156b09468`.
Every selected revision is exact, every file is upstream-LFS-hash-bound, and
at least one selected object per slice passed a byte-access probe.
The source-safe manifest and receipt were uploaded and re-downloaded
byte-for-byte in Hugging Face dataset commit
[`65729b3e32cc5f86aa440cb2ff2f6e3bc8d64611`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/65729b3e32cc5f86aa440cb2ff2f6e3bc8d64611).

The Common Pile expansion deliberately supplies epistemic functions missing
from a web-heavy reservoir: case law, regulations, Hansard, US government
publications, patents, arXiv and PubMed, biodiversity archives, public-domain
and open-access books, open textbooks, Stack-Edu code, technical discussion,
Wikipedia-family reference material, and spoken explanation. Sai uses the
filtered releases as source candidates, retains each component's exact
revision, and still requires per-record rights and quality review.

Five deterministic coverage populations now bind **1,879 source candidates**
across the original, weighted, frontier, Common Pile, and v3-expansion screens.
The cross-population exact-content replay found zero duplicate pairs; its
receipt is
`e31954f5bd2b220004c6b19c0dd35949052f74a464f0ad009af476e2f6dff0be`.
This is an exact result for the screened candidates, not a claim about the full
21.537 TiB reservoirs and not a near-duplicate result. The source-safe lineage,
population receipts, duplicate reports, and combined report were uploaded and
re-downloaded byte-for-byte in Hugging Face dataset commit
[`de17529bd3ba9ea67355c26985b70350e6b8377f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/de17529bd3ba9ea67355c26985b70350e6b8377f).
Raw candidates and evidence-bearing compiler judgments were deliberately not
redistributed through that public commit.

The first complete 128-row compiler screen materially changed the acquisition
strategy. Although Hermes marked 98 rows `retain`, the independent conservative
routing layer sent only 22 rows (17.2%) directly to representation verification;
the other 106 still require quarantine, rights, cleanup, factual-grounding, or
translation work. This is exactly why model preference is never an admission
decision.

| Original-reservoir source | Screen rows | Representation verification | Main measured obstacle | Current work decision |
| --- | ---: | ---: | --- | --- |
| FineWeb-Edu fill | 24 | 9 (37.5%) | 7 factual, 5 cleanup, 3 quarantine | Priority targeted verification |
| Dolma 3 mix-150B | 24 | 8 (33.3%) | 8 quarantine, 5 cleanup, 3 factual | Targeted recovery and verification |
| FineMath | 16 | 3 (18.8%) | 7 cleanup, 5 factual | Targeted recovery and verification |
| FinePDFs | 40 | 2 (5.0%) | 21 quarantine, 9 translation, 6 cleanup | Bulk expansion paused |
| SmolLM corpus | 16 | 0 | 9 factual, 5 cleanup, 2 quarantine | Targeted recovery and verification |
| OpenWebMath | 8 | 0 | 7 rights holds, 1 quarantine | Rights-blocked pending resolution |

These are exact descriptive results for a coverage-first screen, not estimated
full-source acceptance rates. They are still decisive for resource allocation:
Sai will not build its center of gravity around raw FinePDF volume, will not
silently count unresolved OpenWebMath bytes, and will spend verification effort
first where the screen found recoverable signal. The aggregate receipt is
`c0706f92535aded29c679fff5c35798a6380c01b58dc9bdf95ffd155f9a76359`;
the deterministic source-work ledger receipt is
`7cd1a6b040eaa00a40eb37f2578045780815931d6f712a43d5bd33848a4e250e`.

The 31-source Common Pile breadth audit has also completed all 124/124 compiler
judgments under aggregate receipt
`d79749882b8e306e87997a2e0f13bd558e0bef268356b696e6d140eab656bd22`.
At four rows per source it is a discovery screen, not an acceptance-rate
estimate. ArXiv Abstracts and Public Domain Review each routed 4/4 rows to
representation verification with no quarantine; Python Enhancement Proposals
and StackExchange each routed 3/4 there with one cleanup review. Those are
high-priority candidates for a larger source-specific confirmation screen.
Wikiteam routed 4/4 rows to quarantine, while arXiv Papers, peS2o, and Wikimedia
each routed 3/4 there, so bulk expansion of those representations is paused.

No compiler-only result can bypass an independent gate. USGPO illustrates why:
Hermes routed 4/4 rows to representation verification, but the corrected exact
word boundary independently found benchmark overlap in 3/4 rows. It is not a
priority clean lane. The complete aggregate and conservative work ledger were
published and byte-replayed in Hugging Face commit
[`90a87727f9b5e88b0268153001f19d47c091101d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/90a87727f9b5e88b0268153001f19d47c091101d).
Raw candidates and evidence-bearing judgments were not redistributed.

The next Common Pile gate is now frozen rather than selected by intuition.
Receipt `350e96f2c1bbffa473eb7801fcd43548b03141754622c1ba0cd55a1e7bb9e625`
combines the completed compiler aggregate with the independent v2
contamination screen. Promotion requires at least four observed rows, at least
50% representation-verification routing, zero quarantine or rights routes, and
zero benchmark-overlap rows. It selects ArXiv Abstracts, GitHub Archive,
LibreTexts, Pressbooks, Public Domain Review, Python Enhancement Proposals, and
StackExchange for a 224-row confirmation. Confirmation rows must be exact-row
and exact-content disjoint from discovery; acquisition uses a different pinned
parent whenever one exists and otherwise reuses the only pinned parent with
fail-closed discovery-line and content-hash exclusions. This is confirmation
workload selection, not training admission. The executable plan was
byte-replayed from Hugging Face commit
[`77cb201f68dab8f447f3d3a6e81b63a9ee4407f5`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/77cb201f68dab8f447f3d3a6e81b63a9ee4407f5).
The earlier receipt
`a48d9860193460e037c095f5483eb18b4b5199ec6b7be05eba8c6ebcfe562676`
and dataset commit
[`6618216352dbecfae8e3c92eef53d4e14e1e24f1`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6618216352dbecfae8e3c92eef53d4e14e1e24f1)
are retained as superseded audit history: their universal different-parent
requirement is infeasible for selected sources with only one pinned parent.

The v2 confirmation population is now sealed under receipt
`40e72050e1c5a44d0e7618413d6e731de23232be9982f8f4be5d13eada44b6a5`:
224/224 rows, exactly 32 from each selected source, three different-parent
lanes, and four single-parent lanes with enforced discovery exclusions. It
verified 2,637,343,362 compressed parent bytes while holding at most one parent
locally. An independent duplicate replay covered all 60,378 possible pairs
across the 124-row discovery and 224-row confirmation populations and found
zero byte-exact or normalized-token duplicates (receipt
`6fd6b8491a627021d2cfd2db75c6eb8b495bcd48044254452583037eff2f8785`).
The corrected benchmark boundary found 223/224 clean rows: six lanes were
32/32 clean, while one GitHub Archive row contained two exact word-shingle
hits. Therefore GitHub Archive cannot receive blanket source promotion, and no
lane is training-ready until the remaining compiler, rights, full-corpus
deduplication, and transformation gates close. Screen receipt:
`02fa2ead3bd14689fb6f46bf7eaca4f1518342aea8e3c08393d44aac1eb9acba`.

The bounded production-pilot path is now executable but evidence-locked.
`sai.data.confirmation_promotion` combines the confirmation compiler aggregate,
corrected benchmark screen, discovery/confirmation duplicate report, and exact
population receipt. A source must have at least 32 confirmation rows, at least
50% representation-verification routing, zero quarantine, zero rights holds,
zero benchmark-contaminated rows, zero exact/normalized duplicate pairs, and
exact identity/content disjointness. A pass authorizes only a bounded streaming
pilot, never bulk ingestion or training.

`sai.data.common_pile_streaming_pilot` consumes that promotion receipt. It
chooses the smallest hash-pinned parent not used by discovery or confirmation
when available, downloads only that parent, verifies the full compressed hash,
excludes every audit line and content identity, and chooses deterministic
bottom-k rows in a text-free first pass. A second pass replays the exact rows,
writes provenance-complete raw candidates, applies the corrected official
benchmark boundary and exact normalized deduplication, and then runs an
exhaustive bounded near-duplicate join over every surviving pilot pair. The
join uses exact SHA-256 identities of five-word shingles, the frozen reservoir
Jaccard/containment thresholds, deterministic canonical survivors, and a
source-safe receipt; it does not claim that cross-source global deduplication
is complete. The pilot then seals receipts and removes the downloaded parent.
Documents outside 200 bytes to 128 KiB are counted rather than silently
truncated. Pilot rows still remain non-training data until full rights,
cross-source deduplication, and representation verification close.

Every surviving pilot document also receives a text-free attribution companion
record. The replay reconstructs the exact upstream repository revision,
source file, and row index from the raw population; reclassifies the original
license declaration; verifies that its canonical license matches the cleaned
document; and preserves attribution/share-alike obligations. This closes
internal lineage loss without claiming external provenance verification or
legal clearance, both of which remain explicit open fields.

`sai.data.cross_source_pilot_duplicates` is the next no-idle-gap gate. Once at
least two source pilots exist, it preserves a deterministic per-source floor,
fills unused capacity by a global bottom-k key, and exhaustively compares every
unordered pair in the resulting sample with the same exact sparse shingle
join. Its receipt distinguishes cross-source duplicate components and says
whether the sample happened to cover every pilot row. It never upgrades a
sample result into full-reservoir deduplication or training admission.

That gate has now completed over the entire bounded pilot population. The
Pressbooks lane selected 2,000 rows, rejected 42 against the public benchmark
boundary, and removed ten more through the within-pilot near-duplicate filter,
leaving 1,948. Public Domain Review had 1,353 eligible rows, rejected ten at
the benchmark boundary, and removed one near duplicate, leaving 1,342. The
combined cross-source pass covered all 3,290 survivors and all 5,410,405
unordered pairs logically; it found zero additional duplicate groups and
dropped zero rows. Its receipt is
`f489ab77ec8c8cd930e8b7b7dfafb17e36f4c8936d3d8f6f7630ce33421729ba`.
This closes the cross-source gate for these two bounded populations only. It
does not establish reservoir-wide deduplication, rights verification,
representation verification, or training readiness.

The two pilot receipts, nested filter receipts, text-free attribution
manifests, cross-source receipt, and conversion ledger r6 were all replayed
byte-for-byte from Hugging Face dataset commit
[`f5ec9e07e987f008c52a29b31922c2e361c8472a`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f5ec9e07e987f008c52a29b31922c2e361c8472a).
Raw and transformed source text was deliberately not published.

All 3,290 cross-source survivors are now joined back to their exact source
revision, file, row, declaration, decontamination evidence, and pilot receipt
in a compiler population. It contains 1,948 Pressbooks candidates and 1,342
Public Domain Review candidates under receipt
`b87e7c864fec79de60dc90576777b347dfd66bdbe40af40c42db9f91f422a372`.
The candidate file SHA-256 is
`9e2da348914c6c175bcb9d2fe3272caaa891923bdc27f211118b8e849f33f93a`;
the text-free lineage SHA-256 is
`b68ff632fa6c5fda2677a289569679387d29157771a1f382851e0ef983fe76a0`.
Hermes compilation is running across 128 immutable identity shards with
resumable create-only receipts. The compiler can recommend or reject
representations, but its output remains a judgment rather than independent
verification or admission.

Only the source-safe receipt and text-free lineage were published and remotely
replayed in Hugging Face dataset commit
[`bb34c47c1cf77f3bb9b3603ccdfa8c61ac6d2caf`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/bb34c47c1cf77f3bb9b3603ccdfa8c61ac6d2caf).
The evidence-bearing candidate text remains local.

The original compressed parents were then replayed once more to recover the
source metadata that a language-model judgment cannot reconstruct. Every one
of the 3,290 retained identities now has a text-free binding to its original
parent row, native ID, declared license, bibliographic metadata, and source
URLs. Pressbooks contributes 1,948 complete records and 2,844 unique chapter or
book URLs; Public Domain Review contributes 1,342 complete records and 1,342
unique article URLs. The exact metadata-manifest SHA-256 values are
`253e433031600aae7a5b1155122d2f93afd276f8e61d8c61663a5b6b87dfef40`
and
`d4228cc643ae4797b294cc1b697ee8f97c61082d7f00cbf7469f7af57e8eb4d1`.
Both compressed parents were size/hash verified and removed after replay.

This closes internal metadata lineage, not external rights provenance. The
pinned Common Pile source cards explicitly warn that inaccurate metadata or
license laundering can mislabel documents. Live source-page and work-level
rights verification therefore remains mandatory. The text-free manifests and
receipts were uploaded and byte-replayed in Hugging Face dataset commit
[`35efe5b49e62a44dbd430f2c238116acbc571e82`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/35efe5b49e62a44dbd430f2c238116acbc571e82).

A bounded live page probe has now measured that next boundary without storing
source HTML. It froze 901 unique Pressbooks work pages, 258 retained Public
Domain Review essay pages, and the official Public Domain Review reuse-policy
page that covers the 1,084 retained collection/conjecture records. Those 1,160
targets cover every one of the 3,290 pilot records exactly. Public Domain
Review returned HTTP 200 plus the expected CC BY-SA evidence on all 259
targets, covering 1,342 records. Pressbooks returned 193 HTTP 200 responses;
182 contained the expected declaration, covering 377 records. Its remaining
targets produced 632 HTTP 403, four HTTP 401, two HTTP 404, and 70 exhausted
transport retries. Across both sources, matching evidence was observed for
1,719 records. No response was truncated and no source page body was written
to disk.

This is temporal page evidence, not legal clearance. In particular, a license
string appearing on a page does not prove that it governs every retained text
span, while a blocked page cannot be treated as negative rights evidence. The
receipt therefore deliberately keeps `rights_provenance_verified=false`,
`legal_clearance_established=false`, and `training_ready=false` for the full
population. Receipt
`2483d76d4c596541044cd45eda8c73ad0b6539c5e5767f6f3529865f8ee5b5de`
and results SHA-256
`5f85c155881a820153136bfb2c4c774cd9f16a468f6fb1a0be207893db1966e8`
were uploaded and byte-replayed from Hugging Face dataset commit
[`be6becc19c2e2e1bccfa12640b9f5ca4368da43c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/be6becc19c2e2e1bccfa12640b9f5ca4368da43c).

The probe has also been joined back to every exact identity as a fail-closed
adjudication queue. It routes 377 records to Pressbooks book/section scope
review, 258 to PDR essay license/exception review, and 1,084 to PDR policy and
embedded-third-party-material review. The unresolved Pressbooks population is
separated into 52 records whose HTTP 200 page lacked the expected canonical
pattern, 1,300 behind HTTP 401/403, two on missing pages, and 217 attached to
exhausted transport outcomes. Access controls were not bypassed and no route is
an automated legal decision. Queue receipt
`afc84f09628ca9153c626d1f527f715b23bb967a644a19ca3f719db5825fe7c3`
and queue SHA-256
`8273f615b97e13a7cd7815078acc4b60666a1874e2b32355eed507ab1e638335`
were uploaded and byte-replayed from Hugging Face dataset commit
[`e1a2f00a121cbfec417cabe657111e5cb6a2de30`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e1a2f00a121cbfec417cabe657111e5cb6a2de30).

Public Domain Review has now received a stricter, per-identity scope replay.
Inspection of the pinned Common Pile collector at commit
`9457f04a14cb2355ab00023420369d46ffd4a395` found that its permissive-license
checker was defined but not applied in record construction, while quoted
`blockquote`/`q` material could enter the extracted text. The audit therefore
re-fetched every one of the 1,342 frozen PDR pages, reproduced the pinned
selector geometry, required the applicable official-policy or page-specific
CC BY-SA evidence, and constructed a quotation-excluded hash without retaining
HTML, source text, or scoped text.

The first immutable replay (`20260825-r1`) exposed a live license-footer class
collision and is retained only as superseded audit history. The corrected
active replay (`20260825-r2`) deterministically excludes the exact license UI
before comparison. It accounts for all 1,342 identities: 1,253 exactly replay
the frozen candidate, 85 require source-page drift review, and four returned
unavailable responses. Across the inspected pages, 961 quotation elements
containing 446,625 codepoints were excluded from the scoped hashes. The active
receipt is
`779eeef0a192dcd73744f68aa47af46305c5a45391601a244a87be3cdbf0f40a`;
the results SHA-256 is
`3d9b8793a2696dfc072ab226262c577ceb76212a22794fcd312d8e254a7d6271`.
Both r1 and r2 source-safe evidence were force-downloaded from and replayed
byte-for-byte at Hugging Face dataset commit
[`c2dbb5dfe68a85b06e85c5d1962162d12a62c68f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/c2dbb5dfe68a85b06e85c5d1962162d12a62c68f).
This measures scope evidence only: every row remains non-cleared and
non-training-ready.

The active scope was then materialized into actual candidate data rather than
left as a hash-only plan. A third live replay reproduced all 1,253 eligible
pages without drift and emitted 5,919,449 UTF-8 text bytes across 995
Collections, 243 Essays, and 15 Conjectures. It excluded 883 quoted elements
containing 412,039 codepoints; the other 89 pilot identities remain absent.
Because quotation deletion creates new token adjacencies, the exact transformed
text was re-screened against the pinned official benchmark boundary. All 1,253
rows remained clean with zero word or eligible-code shingle hits. Receipt
`52484c5f8b22d79b231e71d2d03962fd10ea18b29c6740c02b86afd25ebd7741`
binds the scoped candidates; receipt
`9a051d33874a8515938d072914dbe3888e7cde52ed7eff7754c12d7efd528097`
binds the post-transformation screen. This is a replayable open candidate
population, not content-quality verification or training admission. The card,
candidate bytes, receipts, and text-free decisions were force-downloaded and
replayed from Hugging Face dataset commit
[`6885a18a0a98eb10c3d5d0e73ad276dd49a99a0d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6885a18a0a98eb10c3d5d0e73ad276dd49a99a0d).

The next PDR compiler stage is now executable but has not been launched. Once
the complete 3,290-row compiler population seals, it will join the clean PDR
texts to their exact content and rights lanes, retain only identities routed to
representation verification, and freeze at most six compiler-requested
derivative types per source. The generation contract requires one exact source
citation for every representation, preserves CC BY-SA attribution and
share-alike obligations, and treats prerequisite edges and cross-domain
connections as unverified candidates. Generated text is emitted separately
from source text, with source citations represented by hashes in the candidate
corpus. It remains nontraining until post-generation benchmark screening,
global deduplication, source-claim verification, and independent representation
verification complete. No representation generation or model training is
authorized by the code-only preparation.

A single high-throughput verification pass is also dependency-staged after the
post-generation screen. It compares every generated representation with its
exact source in a fresh request, requires literal evidence from both texts, and
routes outputs into retain, revise, or reject lanes using strict entailment,
factual-fidelity, uncertainty, cultural-specificity, prose-quality, and copying
checks. Because the verifier uses the same model family as the generator, a
retained row records `same_model_family_verification_complete=true` while
keeping `independent_model_family_verification_complete=false`,
`representation_verified=false`, and `training_ready=false`. This gives Sai a
fast quality filter without mislabeling same-family agreement as independent
truth.

Rights are independently fail-closed. The exact pinned Hugging Face cards for
all seven confirmation candidates currently expose no top-level `license`
field; source-specific READMEs instead describe their collection policy and,
for several sources, point to per-document license metadata. The 224
confirmation rows contain concrete CC0, Public Domain, CC BY, CC BY-SA,
Apache-2.0, MIT, BSD-2-Clause, WTFPL, and two unversioned “GNU Free
Documentation License” declarations. `sai.data.license_policy` canonicalizes
only exact recognized aliases and attaches attribution/share-alike obligations.
The unversioned GFDL label and every unknown value enter `rights_hold`; they are
excluded from pilot selection. A recognized declaration still records
`source_provenance_verified=false` and `legal_clearance_established=false`.
The sealed 224-row audit found 222 recognized declarations and two rights holds,
both in LibreTexts. Its receipt is
`357414811d687921225830732feae6f45508707f126c01cf7b01624eaed0df40`;
the text-free artifact was remotely byte-replayed in Hugging Face commit
[`e6b1210f26a7fb7e06e45c193131aa71d2c574df`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e6b1210f26a7fb7e06e45c193131aa71d2c574df).
The v2 promotion decision now requires this independent rights receipt in
addition to compiler, contamination, duplicate, and disjointness evidence.
Forward conversion uses exact-declaration policy schema v2: the complete alias
table is hash-bound, and the observed reservoir declarations `ODC-By-1.0` and
`CC-BY-2.0` are recognized with attribution obligations. This does not
retroactively change the immutable 224-row audit receipt.

The official public-benchmark contamination boundary is executable, but its
first code-shingle policy has been superseded. It
projects 18,235 rows from MMLU-Pro, HumanEval+, MBPP+, CorrectBench,
LiveCodeBench release v6, LongBench Pro, LiveBench 2024-11-25, IFEval, and MuSR
without retaining benchmark text. The strictly ordered binary indexes contain
27,979,728 unique 13-token word shingles (895,351,296 bytes) and 1,907,051
unique 8-token code shingles (61,025,632 bytes). Those r1 artifacts passed
their byte-level replay, but the code index also admitted punctuation-only
windows. Its receipt,
`073bb9f8a9ab9954ed3913b2414ff718e8f86a5020b2eb1feb18069cd75510f1`,
and non-reversible index are mirrored in Hugging Face commit
[`ad178281de02625f043359a89070e905944452b9`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/ad178281de02625f043359a89070e905944452b9).
It remains immutable audit history but is not an active admission gate.
RULER remains an explicit gap until its generator is pinned to Sai's exact
tokenizer and length geometry. Building the boundary makes contamination
testing possible; it does not retroactively decontaminate any source bytes.

The active v2 boundary keeps the exact r1 word index byte-for-byte and admits an
exact 8-token code window only when it has at least four alphanumeric-bearing
tokens, three distinct alphanumeric-bearing tokens, and 16 total characters.
The corrected code index contains 475,804 unique shingles (15,225,728 bytes),
down from 1,907,051. Its receipt is
`9fee65cb9f99813407ea4d5e4c35b4bc0bb7659c1720342f0f50bd1a8c237667`;
the receipt file SHA-256 is
`cd985016d5a301b4a1d17e9ee0f5290edda956f0434ba7293475cc187037d20a`.
Both indexes passed an independent hash, byte-count, and strict-order replay.

All five v2 population screens are complete. They found 69/1,879 flags: 42 rows
with word overlap and 27 additional eligible code-only rows. The population
counts are 6/128 original, 26/1,024 weighted, 28/512 frontier, 7/124 Common
Pile, and 2/91 frontier expansion. Nemotron specialized reasoning is 25/96,
comprising five word-overlap rows and 20 additional code-only rows. Neither an
individual flag nor a clean screen licenses bulk source admission or rejection.
The boundary and all five source-safe screen receipts were uploaded and
byte-replayed in Hugging Face commit
[`43ae57ee4981c78ae23c111436b1fc9b6aa27023`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/43ae57ee4981c78ae23c111436b1fc9b6aa27023).

The 91-row modern-source expansion compiler pass is now complete under
aggregate receipt
`afd82b43ac66f3a485d167b97f79fccc75bc67026c94e182d846a6923f9dea23`.
The result again separates a model's `retain` verdict from actual source
readiness: Hermes retained 60/91 rows, but conservative routing sent only 14/91
to representation verification. Nemotron Legal contributed 8/21 such rows and
is the only priority targeted-verification lane; Nemotron Specialized v1.2 sent
22/30 rows to factual-grounding review and none directly to representation
verification; PleIAs Common Corpus sent 16/40 to quarantine, 11/40 to cleanup,
6/40 to translation, and 6/40 to representation verification. Independent v2
benchmark screening found both contaminated rows in PleIAs (2/40), while the
Nemotron Legal and Specialized samples were clean. These are coverage-screen
results, not source-wide yield estimates. The aggregate and fail-closed work
ledger were remotely byte-replayed in Hugging Face commit
[`2a085eacf1479293e3c369d7eaa8e476d7f84054`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/2a085eacf1479293e3c369d7eaa8e476d7f84054).

The r1 286/1,879 overall and 77/96 Nemotron conclusions are retracted because
they were materially inflated by nonsubstantive code windows. Those screens
remain immutable evidence of the discovered policy failure in
Hugging Face dataset commit
[`5dc89bfeceadf56663a8f00c479f5d41d5229671`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/5dc89bfeceadf56663a8f00c479f5d41d5229671).
They are explicitly superseded and are not active contamination decisions.

[PleIAs Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus)
adds a distinct 2.27T-token, traceable open corpus rather than another opaque
web mixture. Its rows expose collection, open-status, license, language,
creator, and date metadata alongside text. Valuable non-English material is a
translation-discovery pool, not automatic English training data; literary form
and cultural context remain protected from indiscriminate rewriting.

These are **physical source-object bytes, not measured text-payload bytes**.
FineWeb2-HQ, for example, includes large embedding columns, so treating its
repository size as English training text would be materially false. The two
Ultra-FineWeb generations may overlap, and every web-derived source may overlap
the original reservoir. None of those bytes are counted as unique or
training-ready until text-column extraction and global semantic deduplication
complete.

Two exact text-payload probes now measure that distinction instead of guessing
it. One member per source was selected by a frozen SHA-256 rank before size or
content inspection. Eight selected members fit the first probe's 4 GiB parent
cap. FinePDF's independently selected 4.84 GB member was blocked rather than
replaced by a conveniently smaller shard, then measured in a second prospective
probe with a 6 GiB cap. Every measured member was fully downloaded, matched to
its pinned size and SHA-256, streamed one at a time, and deleted afterward.
“Useful” below means only the mechanical 200 B–128 KiB size window; it is not a
quality, rights, uniqueness, or admission judgment.

| Source | Exact physical bytes | Text UTF-8 bytes | Useful UTF-8 bytes | Useful/physical |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu fill | 2,378,402,603 | 3,832,560,263 | 3,741,009,274 | 1.572908× |
| Dolma 3 mix-150B | 5,349,719 | 62,584,337 | 59,977,943 | 11.211419× |
| FineMath | 733,726,864 | 1,199,827,734 | 1,165,858,399 | 1.588954× |
| SmolLM corpus | 2,391,060,328 | 3,915,215,823 | 3,822,727,059 | 1.598758× |
| PleIAs Common Corpus | 430,252,575 | 795,883,158 | 750,836,358 | 1.745106× |
| FineWeb2-HQ multilingual | 1,203,684,113 | 447,018,551 | 384,827,189 | 0.319707× |
| Ultra-FineWeb current L2 | 82,007,099 | 138,259,448 | 136,403,753 | 1.663316× |
| Ultra-FineWeb earlier L2 | 1,298,592,398 | 2,240,142,912 | 2,190,701,659 | 1.686981× |
| FinePDFs | 4,836,418,450 | 9,944,850,928 | 5,386,374,575 | 1.113711× |

Across only the nine exact measured members, 13,359,494,149 physical bytes
contained 22,576,343,154 text bytes and 17,638,716,209 mechanically useful
bytes. Ratios above one are expected for compressed members. FinePDF contained
397,166 useful rows, 4,821 short rows, and 12,013 rows above 128 KiB. Those long
documents are a structure-aware segmentation queue, not automatic rejects. The
roughly 35× spread between the observed useful/physical ratios proves that
repository size is not a defensible acquisition objective, but these bounded
member probes are not source-wide yield estimates and cannot be extrapolated.
The first plan receipt
`4f5312f7d9ae86b3fbe8998c7e780c7238eae9394fc767fdbedad2affbacc66c`
and measurement receipt
`1d550e0abc513c5b4e61f0ce5890155bfff01bcdbd2a6896f9a078c26952f848`
were remotely replayed in Hugging Face commit
[`fecd9d596c18dd63ab6ea7a89dda7b2544eca4a1`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/fecd9d596c18dd63ab6ea7a89dda7b2544eca4a1).
The FinePDF plan receipt
`325382746db5836ccffa12ea437fcfdfaf12ee0f29e469ac47cf0e43c0559017`
and measurement receipt
`5947564751b941b18d8a025abd3451c2e81cfa6e6357c0cad28213561d372919`
were remotely replayed in Hugging Face commit
[`e15ca127c695d2d42df04e15738e56525f0bb3ce`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e15ca127c695d2d42df04e15738e56525f0bb3ce).

Sai now has a create-only long-document recovery path rather than a truncation
policy. `sai-segment-long-documents` splits only over-budget raw documents,
preferring paragraph, line, sentence, clause, and word boundaries before a
lossless Unicode-character fallback. It never normalizes or rewrites source
text, proves exact parent reconstruction, issues collision-free child row
identities, and emits a text-free segment-lineage manifest. Segment geometry is
carried through benchmark decontamination and the attribution manifest. This is
preparatory compiler infrastructure, not a FinePDF admission result: segmented
documents still require contamination screening, global deduplication, rights
verification, representation verification, and curriculum placement.

The first global deduplication layer is also executable as
`sai-deduplicate-global-exact`. It external-sorts compact text-free indexes in
bounded fan-in passes, groups documents by NFKC/casefold/whitespace-normalized
SHA-256, selects the minimum immutable document identity, and replays every
apparent collision against the full normalized source text before dropping it.
Outputs are deterministic across input order, temporary indexes are removed,
and the duplicate manifest contains identities and byte locators rather than
source text. This closes normalized exact duplicates only; scalable semantic
near-duplicate filtering remains a separate unresolved gate.

The second exact layer is now executable as
`sai-deduplicate-subdocuments`. It follows the August 2026
[frequency/length-aware method](https://arxiv.org/abs/2608.03089): natural-boundary
segmentation with short forward merges, normalized global exact counting, the
explicit `T(C,L)` copy budget, document-identity ordering, whole-boundary-document
retention, and deletion only for sufficiently long contiguous candidate runs.
It uses bounded external-sort fan-in, replays every indexed occurrence against
the immutable source before trusting a hash, recalculates identities for changed
documents, and writes a text-free parent-to-output transformation manifest.
Numeric template normalization is restricted to natural-language chunks;
fenced code is indivisible and exact, while full code-domain documents
currently fail closed as indivisible. This is deliberately safer than
pretending a language-agnostic brace parser preserves every programming
language.

The implementation does not turn the paper's result into a Sai result. Corpus
promotion still requires an identical-token, identical-compute,
source-disjoint comparison of unchanged, keep-one, and adaptive retention.
The CLI freezes both executable transformed arms with
`--retention-policy keep_one_control` and the default
`adaptive_frequency_length`; the immutable input is the unchanged arm.
Semantic near-duplicates remain a separate measured gate, and every receipt
keeps `training_ready=false` and `four_b_training_authorized=false`.

NVIDIA's organic/translated Nemotron-CC v2.1, CC-Code v1, and Code v2
repositories were investigated but are not counted: metadata is visible while
the current user token receives HTTP 403 on the actual gated objects. The
public Nemotron specialized-reasoning slices are counted and will still face
the same quality, novelty, generator-lineage, contamination, and benchmark
gates as every other synthetic source. In particular, Sai does not import the
2.1T-token medium-high synthetic-rephrase bulk merely to inflate volume.
Nemotron Pretraining Code v3 is also excluded from byte counts because its
published Parquet schema contains repository, path, language, and commit
metadata but no code text. Those locators may support a future rights-aware
source fetch, but an index is not training data.

#### Moving-center spiral for an 8T-token run

The prospective schedule moves its center of gravity from foundations to
synthesis and expertise while preserving both tails. Expert material begins in
the first token stage, and foundational rehearsal remains present through the
last 400B tokens.

| Stage | Token interval | Stage tokens | Foundation | Intermediate | Advanced | Expert | Minimum cross-domain material |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Foundation | 0–2.0T | 2.0T | 55% | 30% | 12% | 3% | 1% |
| Expansion | 2.0–4.8T | 2.8T | 25% | 45% | 23% | 7% | 4% |
| Depth | 4.8–6.8T | 2.0T | 12% | 28% | 42% | 18% | 10% |
| Synthesis | 6.8–7.6T | 0.8T | 10% | 20% | 35% | 35% | 30% |
| Annealing | 7.6–8.0T | 0.4T | 10% | 18% | 30% | 42% | 20% |

These percentages are the current prospective difficulty-band allocation, not
a frozen source mixture. Within each stage, the compiler and proxy experiments
must still discover which concept, style, source, and reasoning regions buy the
largest source-disjoint marginal capability gain without causing retention
loss. Annealing uses the highest-value mixture supported by that evidence; it
does not simply become “all expert data.”

The policy is executable in `sai.data.eight_trillion_spiral`. It binds exactly
8,000,000,000,000 tokens, contiguous stage boundaries, exact per-band token
allocations, nonzero early expertise, nonzero late foundations, and receipt
SHA-256
`ffa85e065bb7a3895af55bfa9ffdb7f65e236d05722e1ca8abea6161bf259bd2`.
Both `training_authorized` and `four_b_training_authorized` remain false in the
artifact: a long-horizon policy cannot authorize a run whose accepted stream
does not yet exist.

#### Synthetic data is a bridge compiler, not a prose factory

Sai's synthetic advantage is intended to be **knowledge composition**. The
model should not merely know biology and information theory independently; it
should learn when and how their structures connect. Candidate pairings include
biology × information theory, music × Fourier analysis, law × logic,
architecture × structural engineering, history × economics, literature ×
psychology, computer systems × thermodynamics, chemistry × quantum mechanics,
and art × geometry. The pairing list is not a quota and novelty is not assumed
merely because two domain labels appear in one prompt.

Every admitted synthetic bridge must carry:

- at least two genuinely distinct domain identities;
- exact source anchors and immutable source hashes;
- the concepts and prerequisite edges required to understand the bridge;
- evidence that prerequisites were taught earlier or are explicitly rehearsed;
- a relationship that is not a paraphrase or a superficial word collision;
- an independently solved answer or deterministic verifier;
- translation and transformation lineage; and
- benchmark-overlap evidence sufficient to keep evaluation prompts out of
  training.

Generic ungrounded generation is forbidden. The compiler should prefer
representations whose truth can be checked: executable code and tests,
symbolic solvers, simulations, constraint systems, multiple independent
solutions, cited synthesis from primary sources, and contradiction-seeking
review. A model-generated explanation can improve presentation, but it cannot
create a reality anchor or verify its own unsupported claim.

The cross-domain generation loop is therefore:

```text
qualified reality anchors + concept/prerequisite graph
    -> identify distant but structurally meaningful concept pairs
    -> generate a task, derivation, explanation, or worked example
    -> solve independently or execute a deterministic verifier
    -> reject unsupported, duplicate, stylistically collapsed, or contaminated work
    -> assign difficulty on linguistic, conceptual, and reasoning axes
    -> place into the spiral only after prerequisite and retention checks
```

Gradient-space or capability-gap targeting can later choose underrepresented
bridges, but it may not weaken those grounding requirements. Synthetic volume
is never the target by itself; verified marginal learning is.

#### Hermes and Institutional Books operating state

Hermes is the compiler workforce, not an oracle whose output is accepted by
default. The current Institutional Books program starts from 983,004 Harvard
Library volumes totaling 242,051,626,500 upstream tokens. A metadata-first
10,000-volume review queue spans 115 languages and 772 language×subject cells;
9,409 rows are non-English translation-discovery candidates and 591 are
English controls. This is coverage-first sampling, not a desired final language
ratio.

The first authenticated enriched-text pilot contains 185 candidates from 200
source rows: 164 English and 21 non-English, with 14 OCR rejects and one token
bound reject. Hermes has now completed the first production-schema judgment for
an advanced 1917 geology volume. It retained the source as a historical
scientific anchor, marked outdated claims as a risk, extracted 34 concepts, 11
prerequisites, and six evidence-backed edges, and left the raw archive source
explicitly non-training-ready. The successful request used 12,314 prompt and
1,454 completion tokens after one invalid first response and one schema-bound
repair. This single result proves the worker and repair path function; it does
not estimate corpus-wide acceptance quality.

A separate reservoir-wide coverage audit now freezes 128 generic compiler
candidates across six content-bearing source families: 40 FinePDFs rows (16
English and 24 named non-English language strata), 16 FineMath rows across all
four quality/subset bands, 24 Dolma rows spanning reference, papers, math,
Python/Rust/Java, and 18 PDF/web topics, 16 SmolLM educational-web and
Cosmopedia rows, eight OpenWebMath rows, and 24 FineWeb-Edu rows across six
crawl years. Institutional Books remains on its richer book-specific schema and
is not flattened into this generic audit.

The population file contains 128 unique identities and 1,059,279 bytes, with
SHA-256
`d3f24cd2855400a00e16f0bcf6dca63190a0f6653ee64031d77af9b35e83d823`.
Its 138,940-byte lineage file has SHA-256
`dc4cab1b3a0cea23b12841afd830970e588f3e0a17a52cd5d50849e8dbe8207b`.
Twenty-four compressed Dolma parent files were fully downloaded and matched to
their upstream SHA-256; 104 Parquet parents were read by exact-revision range
requests and remain bound to the upstream LFS hash without falsely claiming a
local whole-file rehash. The population receipt is
`4c5a179bb6863969850cc7d70133650211a445eb124df6dd28948053cb817ed4`.
This is a coverage-first diagnostic, not a statistically weighted corpus
acceptance estimate.

The exact/near-duplicate replay compared all **8,128 unordered pairs** in this
128-row audit population. It found zero byte-identical, normalized-token
identical, five-word-shingle Jaccard, or five-word-shingle containment matches
at the frozen conservative thresholds. The report's canonical receipt is
`ecd131b92a708d0cf002004b62a4c69e86b208a96afa2fea2631ca54e511fc2d`
and its file SHA-256 is
`0cd6e93fa4008728a409de82de29fda9e5f9ce5960316c91981d6f62b523568c`.
This establishes that the diagnostic excerpts are not obvious copies of one
another; it does **not** establish that the 8 TiB reservoir is globally
deduplicated.

Compiler judgments also pass through deterministic conservative routing. A
model verdict of `retain` is quarantined for personal/secret data, held for
rights ambiguity, sent to factual-grounding review for weak reliability, sent
to translation review for non-English material, and sent through cleanup and
transformation review before representation verification. Even
`representation_verification` is not training admission. This prevents a
confident semantic reviewer from silently overriding objective data gates.

Hermes routing depends on content type:

- Preserve high-value English literature, rhetoric, letters, essays, and other
  form-bearing expression instead of flattening it into generic model prose.
- For non-English technical and factual work, create English representations
  with exact source and translation lineage, while preserving source metadata.
- For non-English literature, prefer a reputable admissible human translation.
  If none exists, keep separately labeled literal and literary synthetic
  translations and do not represent either as the original voice.
- For papers, standards, reference works, and technical books, preserve the
  source and derive multiple grounded representations such as prerequisite
  maps, concise references, textbook explanations, worked examples, FAQs, and
  misconception/correction pairs.
- Reject or quarantine OCR damage, bibliographic ambiguity, missing rights
  evidence, duplicative editions, unsupported factual claims, benchmark
  material, and synthetic voice collapse.

The exact Common Pile confirmation is now complete: all 224 source-disjoint
rows have compiler receipts and all 32 shards have summaries. Hermes returned
207 `retain`, 11 `review`, and six `reject` verdicts, but deterministic routing
is deliberately stricter: 141 rows require representation verification, 49
cleanup review, 15 factual-grounding review, 17 quarantine, and two
transformation review. The sample spans seven sources; 187 rows identify at
least one cross-domain bridge. The run consumed 958,783 model tokens and 48
rows required bounded schema repair. These are diagnostic results, not an
acceptance rate or training admission.

Only `common_pile_pressbooks` and
`common_pile_public_domain_review` cleared the frozen zero-quarantine,
zero-rights-hold, zero-benchmark-contamination, identity/content-disjoint, and
minimum-representation-verification checks for a bounded streaming pilot. The
other five sources remain held, and neither promoted source has bulk ingestion
or training authorization. Source-safe aggregate, decision, and promotion
receipts were uploaded and byte-replayed in Hugging Face dataset commit
[`44fbdd30cedc89ac908057929468d3162651d645`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/44fbdd30cedc89ac908057929468d3162651d645).

The next operational work is to expand sustainable, stratified compiler lanes;
build cross-source exact and semantic duplicate families; populate the concept
prerequisite graph; create verified English translations and grounded
representations; and measure accepted bytes and tokens by domain, culture,
style, complexity axis, and epistemic function. Only the accepted, replayable
output of those steps can become a curriculum shard.

#### Evidence and status vocabulary

To keep progress legible, Sai uses these states consistently:

- **Referenced:** exact upstream repository, revision, path, size, and object
  hash are known.
- **Locally present:** bytes were actually fetched and their content hash was
  verified.
- **Candidate:** the source passed mechanical parsing into a compiler queue.
- **Judged:** a model or human produced a schema-valid assessment with a sealed
  request/response receipt.
- **Verified:** rights, content, transformations, evidence, and contamination
  checks passed independently.
- **Curriculum-ready:** concepts, prerequisites, difficulty axes, duplicate
  family, and spiral placement are complete.
- **Training-ready:** the final packed stream replays exactly and every required
  gate is closed.

The current truthful status is: the 8 TiB reservoir is **referenced and
hash-bound**; the 8T-token spiral is **prospective**; one production book and
224 Common Pile confirmation rows are **judged**; two bounded source pilots
have 3,290 benchmark-screened, near-deduplicated rows; the new UltraData Math
L2/L3 screen has 148/160 benchmark-disjoint rows awaiting Hermes compilation;
a complete PEP parent census has 567 benchmark-disjoint, near-deduplicated
survivors dependency-staged for Hermès compilation;
a live source-page probe observed matching declarations for 1,719 pilot rows
without establishing governing scope; and the reservoir as a whole is **not
training-ready**.

Every training population must pass these gates in order:

1. **Source truth:** reopen exact source bytes; reject corruption, spam,
   benchmark overlap, unsupported claims, and high-confidence duplicates.
2. **Semantic foundations:** bind each lesson to the concepts it teaches and
   assumes. A dependent concept cannot appear confidently before its declared
   prerequisites have accumulated enough independent prior exposure.
3. **Learnable progression:** move from grounding to composition, reasoning,
   and specialization. Surface complexity may break ties, but it cannot
   overrule the semantic prerequisite graph.
4. **Rehearsal and retention:** keep foundational material in every later
   phase. Evaluate the same phase-stratified held-out population at
   prospectively fixed phase boundaries so acquisition and forgetting are both
   visible.
5. **Matched falsification:** compare the proposed order with the identical
   record multiset under a frozen order control, with the same tokenizer,
   initialization, optimizer, token budget, and observation schedule. Require
   held-out likelihood and real source-disjoint capability evidence; any
   prerequisite-phase regression vetoes promotion.

Specialized data is therefore not automatically "better" data. It becomes
useful only after the learner has the language, symbols, primitives, and
compositions needed to extract its signal. Sai will not compensate for a bad
curriculum by adding parameters, inference-time reasoning, or architectural
machinery.

The executable milestone evaluator now enforces the retention part of this
constitution for future matched runs. It scores the same phase-stratified
development population at initialization, every prospectively declared phase
boundary, and termination, then compares curriculum and order-control
acquisition and forgetting phase by phase. The current live 500M-token run was
frozen before milestone snapshots were added; it retains valid final
phase-stratified evidence but will not be misrepresented as a full dynamic
learning-curve experiment.

**Execution status:** on 2026-08-22 the user authorized Sai to proceed directly
to the 4B architecture once the data is ready. Small-scale architecture
tournaments are no longer a launch prerequisite. The remaining scientific
launch gate is the exact data artifact: canonical unique documents, provenance
and licensing decisions, benchmark decontamination, frozen exposure weights,
semantic-prerequisite curriculum order, tokenizer qualification, and a replayed
packed stream. A bounded one-update 4B execution canary remains a technical
requirement so the full job does not fail on memory, kernel, or checkpointing
mechanics. Neither the authorization nor a successful training run is evidence
of improvement; only matched real-benchmark results can establish that claim.

## Data-first reset — 2026-08-22

- Data quality and presentation order are now the primary Sai admission gate.
  The 30-file FineWeb-Edu prefix remains useful raw material, but passing its
  upstream educational score, basic hygiene filter, and benchmark
  decontamination is not sufficient to make it a training curriculum.
- Audit found that the previous 500M-token freezer preserved upstream file/row
  order. It had no prerequisite progression, difficulty strata, domain pacing,
  or near-duplicate-cluster gate. Dependency-held stream job `769226` and GQA
  launcher `769687` were therefore cancelled before execution: both recorded
  zero elapsed time, zero restarts, no node, no child job, and no scientific
  output. Parallel decontamination recovery `769626` continues because its
  benchmark-disjoint admitted corpus is still required input evidence.
- `sai.data.curriculum` now defines a create-only four-band, four-phase data
  contract. It performs a second quality pass, removes high-confidence
  five-word-shingle near duplicates, measures every document's surface
  difficulty, excludes specialization from grounding, backloads advanced
  material, and rehearses foundational material in every later phase. Its
  validator reopens source/decontamination evidence and replays every output
  row, band, duplicate decision, phase mean, and identity fingerprint.
- Recovery job `769626` completed the exact twenty-boundary decontamination
  pass without retries. It published a `9,541,423,202`-byte admitted corpus
  with SHA-256
  `5e234e56d3df101c668beb49d4582740d6bc3fbe723448231c73a6d4e6e57dda`;
  its receipt file is SHA-256
  `a396590cc253fb208c276fb004c53b5dd50bca651811c9f1c315cd69c7c8cb5f`.
  Curriculum builder `769780` atomically published a `9,469,603,720`-byte
  curriculum and receipt before entering its original single-core replay. It
  scanned `2,153,160` documents, admitted `2,125,835`, rejected `27,325`, and
  emitted every admitted identity exactly once. The admitted bands contain
  `33,220` foundation, `360,044` composition, `1,660,568` reasoning, and
  `72,003` specialization documents. Phase mean difficulty rises monotonically
  from `0.23614` to `0.27084`, `0.29141`, and `0.30431`; grounding contains no
  specialization and later phases retain foundation rehearsal. Independent
  eight-worker replay `770001` completed in `1,351` seconds with zero restarts,
  empty stderr, exact receipt SHA-256
  `b575bebb18509c13848ac34146ac9b8a7d4be54e83e4d66f43100eb717545c1b`,
  and status `qualified`. The redundant single-core replay was then cancelled;
  no published artifact was removed or changed.
- A second audit found that the legacy development stream was produced from an
  earlier corpus slice and could not prove disjointness from the new admitted
  population. Dependency-held freeze/control jobs `769787` and `769788` were
  cancelled with zero elapsed time, zero restarts, and no node before they could
  publish a stream. `sai.data.curriculum_split` now performs the train/development
  separation only after the global quality and high-confidence near-duplicate
  pass. It preserves the four phase boundaries in training, assigns every
  accepted identity exactly once by a frozen hash modulus, and replays the full
  source against both outputs. New training and development token streams must
  bind that split receipt; the legacy development stream is not admissible for
  the curriculum experiment.
- A valid curriculum receipt is still not proof that the order helps. Before
  using it for a larger model, Sai will compare it against a same-document
  deterministic order control at small scale, with matched tokenizer,
  initialization, optimizer, updates, and compute. Held-out NLL/UTF-8 byte and
  source-disjoint real capability must improve or remain nonnegative without a
  domain regression. Tokenizer, filtering, and ordering remain separate factors.
- The current four-phase schedule remains explicitly a surface-complexity
  falsification experiment, not the final semantic curriculum. It cannot prove
  that a concept's prerequisites were learned. The next data gate must bind an
  acyclic concept graph, evidence spans for every exposure, minimum prior
  coverage before dependent exposure, first-exposure positions, phase-local
  rehearsal, and unresolved violations. This is the executable version of the
  rule that a model should encounter the concepts of yellow and blue before it
  is expected to learn their composition into green.
- A separate model-centric scheduler is now executable for web-scale pacing.
  `sai-score-learnability` first compares a predeclared earlier milestone with
  the terminal state from the same independent probe trajectory on an
  exact-record-disjoint target stream. Its immutable receipt binds both model
  states, the target and probe streams, tokenizer, runtime, and every normalized
  loss row. `sai-build-learnability-curriculum` then consumes that receipt and a
  prospectively frozen weak/strong-checkpoint policy. It reorders the identical
  token-and-boundary record
  multiset into `ready`, `developing`, `challenging`, and `stretch` bands,
  preserves ready-record rehearsal in every later phase, and hash-randomizes
  within each phase so continuous loss order is not a hidden factor. The full
  permutation is replayable and treatment checkpoints and terminal benchmark
  feedback are prohibited. This is a future matched factor, not a semantic
  prerequisite claim; no qualifying production score population or training
  run exists yet. Sai therefore treats difficulty as two independent axes:
  model-relative learnability and audited semantic prerequisites. Neither can
  silently substitute for the other.
- `sai-compose-semantic-learnability` is the production composition boundary.
  It replays the complete semantic taxonomy, curriculum, annotations, and
  progression report; requires zero skipped or truncated semantic documents;
  and locks every packed record to its audited semantic phase. Weak/strong
  learnability evidence may only determine bands and order *inside* that phase.
  An advanced low-loss record therefore cannot jump ahead of missing
  prerequisites. The output retains the identical token-and-boundary multiset
  and remains unauthorized for training or 4B until a matched comparison wins.
- The semantic gate now freezes exactly 120 review documents: eight from each
  of the 15 pedagogically valid phase/band cells before any labels are produced.
  `grounding:specialization` is excluded because qualified progression requires
  that cell to remain empty; demanding examples there would contradict the
  curriculum being audited. A
  separate replay compares the prospective annotator with an independently
  identified reviewer, validates every cited evidence span against immutable
  source text, and computes concept-set disagreement directly. The taxonomy
  cannot be built unless at least 100 documents were reviewed and disagreement
  is at most the prospectively frozen five-percent ceiling; callers can no
  longer satisfy this gate with an unattached arithmetic-only receipt.
- The first held-out selector attempt, CPU job `770519`, correctly failed before
  publishing an artifact because its legacy 16-cell geometry demanded examples
  from `grounding:specialization`, a stratum that the qualified curriculum
  requires to be empty. The corrected selector was pushed at commit
  `89fda9b76d3898e10a150b25557d7b14768be7d3` and executed as CPU job `770526`.
  It completed in 151 seconds on `evc21`, with zero restarts and empty stderr.
  Independent replay verified exactly 120 immutable rows, eight in every one of
  the 15 valid phase/band strata, and no grounding-specialization row. The
  population file SHA-256 is
  `85a7e804b0622f85b2f45c3edf5a37a1a200fdd3e9c833a8c65edf09a804cce8`;
  its canonical receipt SHA-256 is
  `34ca4ca64acccaf3bc1ae04152156ea879efed47d85dcfdac3242ef5ee8b171a`.
  The population is selected but deliberately unreviewed, and its receipt keeps
  both training and 4B authorization false. Independent blinded annotation and
  disagreement review remain mandatory before this sample can qualify the
  semantic taxonomy or any training curriculum.
- `sai-build-prerequisite-blind-review` closes the remaining label-leakage
  boundary before that annotation begins. It shuffles the 120 rows by a salted
  review identity and exposes only immutable text and the requested evidence
  format. Curriculum phase, surface band, source identity, and original order
  are held in a separately sealed key. Reviewers therefore cannot infer the
  answer from the schedule they are auditing, and no packet can authorize
  training or 4B scaling.
  CPU job `770533` executed the exact pushed implementation at commit
  `d798e4edc48d06ff112c95d44872b0d123ad01ab`, completed in 224 seconds on
  `evc21` with zero restarts, and replayed the packet twice. The blinded packet
  SHA-256 is
  `ae4ebe9721f2b2e156cb72aba9f46f73396f0cdcecd0480754784b32c0efba2d`,
  the withheld key SHA-256 is
  `f2986e52cedec2a5db8802c43505539d76b17210d7e1fac026e79ef2aa8aa3fa`,
  and the canonical receipt SHA-256 is
  `6a07e0e1fa8e5832f2b196cd65e1547f563a47d0f896278fcbb364a3626e347d`.
  Independent inspection confirmed all 120 packet rows omit phase, surface
  band, source, original index, and document identity; the separate key retains
  all mappings. The packet is now ready for two independently identified blind
  reviews, but contains no completed labels and authorizes no training.
  The same tool now compiles a frozen blind response only after reopening the
  sealed key: it validates the candidate concept vocabulary and every cited
  character span, restores phase and document identity in canonical population
  order, and emits a separate lineage receipt. A single compiled review remains
  explicitly insufficient; qualification still requires two independent
  identities and a passing disagreement replay.
  The replay now additionally requires each independent side to label at least
  100 of the 120 documents with evidence-backed concepts. Two empty annotation
  files therefore fail even if their nominal disagreement is zero.
- Source-disjoint split `770039` is now running from that independently
  qualified receipt with four ordered workers across both receipt validation
  and exact split reconstruction, plus a four-hour fail-closed wall limit.
  Earlier split `770024` was cancelled after 542 seconds, before creating any
  output, when byte-equivalence tests proved the ordered parallel
  reconstruction. Development stream `770040`, update-aligned 500M-token
  training stream `770041`, exact sequence-multiset order control `770042`, and
  matched GQA launcher `770050` are dependency-staged. Launcher `770043`
  was canceled before allocation (zero elapsed and zero outputs) when the
  frozen runtime was advanced solely to give terminal replay four hours.
  The first three
  curriculum boundaries occur exactly after optimizer updates 191, 429, and
  715, so no gradient accumulation window mixes adjacent difficulty phases.
- Split construction has atomically published all `2,125,835` admitted
  identities exactly once: `2,104,726` training documents in
  `9,375,399,692` bytes (SHA-256
  `6a43417411f886336632f2ad1abbf539504043f28180cdc4a6fa792e7de6241b`)
  and `21,109` development documents in `94,204,028` bytes (SHA-256
  `56ebbf200bae4ce21c454bd80e91328ab9a486798c83a0b729477f57e0122289`).
  Its qualified receipt self-hash is
  `0580b683427ecae5943cc0a706e9fb31686c46051283b154e1aebc09b78eb0aa`;
  job `770039` completed the independent full replay at `0:0` with zero
  restarts. Its stdout contains the same qualified receipt identity for build
  and replay, and stderr is empty.
- A downstream audit caught a validation confound before allocation: freezing
  the first 1,024 sequences of the phase-ordered development file would test
  almost only grounding data. The development freezer now requires exactly 256
  sequences from each of grounding, integration, reasoning, and specialization
  and binds those four token/byte strata in the stream receipt. Original jobs
  `770040` and `770050` were canceled at zero elapsed with no node, logs, or
  outputs. Training-stream job `770041` completed at `0:0` in `5,386` seconds
  with zero restarts and empty stderr. Its `60` shards contain exactly `244,140`
  packed sequences and `499,998,720` tokens. The four emitted phase budgets are
  `48,896 / 60,928 / 73,216 / 61,100` sequences, and their boundaries match the
  declared optimizer updates. Receipt file SHA-256 is
  `8de9780d4b5b873e668260ee2423c1536912163f9fb6696597663ac0c1e026b1`;
  ordered-stream identity is
  `c4c271f38b55ab277c7660719e3d36bc485063d440a3745b8f4d532545d51636`.
  Dependent job `770042` completed the exact sequence-multiset order control at
  `0:0` with zero restarts. It contains the identical `244,140` packed records,
  `499,998,720` tokens, boundary masks, tokenizer, source, and admitted UTF-8
  bytes, but applies frozen permutation seed `2026082201` with zero fixed
  points. Its receipt file is SHA-256
  `a8481d767a6e468a5c46a69331ebe33fca80ff3d382bfd014b6e7ffa62c05ad0`;
  ordered-stream identity is
  `0d40a828e83b7cda52fcf77489dbe2a223761fe70f8972f2d3e66297d4439513`.
- The likelihood evaluator and terminal order comparator now retain all four
  development strata separately. A lower aggregate NLL cannot pass if
  grounding, integration, reasoning, or specialization regresses against the
  exact-order control. This closes the remaining possibility that easier rows
  could hide curriculum damage to advanced material. Development job `770086`
  exposed a zero-work runtime packaging defect: eight
  historical executable bits had been stripped, so the runtime correctly
  failed its clean-tree check before Python or data access. Its dependent
  launcher `770088` canceled without allocation. Both output roots remained
  absent. Replacement job `770105` completed at `0:0` with zero restarts and
  empty stderr against clean immutable runtime `78f9d7b`. It published exactly
  `1,024` packed sequences: `256` and `524,288` tokens from each of grounding,
  integration, reasoning, and specialization. The stream receipt file is
  SHA-256
  `6ae403c18f683fa3ddd7536989c38f7c5c4c14a3b448993e020cf84cabb8eb9a`;
  its ordered-stream identity is
  `232f68b380db1cbfa75aeda8c8bb3a878f9afe1551528b6efcbccbb4c6e6a34a`.
  Its exact train/development split receipt and development source hashes match
  the independently qualified lineage. Launcher `770106` then failed before
  submission because its export omitted the literal `_stream_` component of the
  completed training-stream path. It ran for four seconds on CPU, emitted no
  output, and submitted no GPU child; continuation `770127` was dependency-
  cancelled without work. Recovery launcher `770136` uses the corrected exact
  path and is currently replaying the split and all three streams with zero
  restarts. It has not yet submitted a GPU child. Continuation `770137` remains
  dependency-held. The trainer, evaluator, comparator, data bytes, model
  geometry, seed, optimizer, and evaluation rows are unchanged.
  Pre-allocation runtime audit then found that the prior 100M-token GQA arm
  required `9,905` seconds; a linear 500M-token projection is `49,525` seconds
  and the predeclared 25% safety margin is `61,907` seconds. The original
  eight-hour child limit was therefore deterministically insufficient. Future
  launcher bytes request 18 hours for each full arm. A create-only live guard
  will apply the same limit to the two still-pending children immediately after
  `770136` publishes their dispatch, before either training arm starts; it
  changes no data, model, seed, optimizer, update, or evaluation identity.
- Source-disjoint MMLU-Pro and MuSR evaluation now admits curriculum-derived
  streams only through the exact completed lineage from the benchmark-audited
  decontaminated source, through the qualified curriculum and split receipts,
  to the frozen train stream. Direct-source evaluation remains unchanged;
  missing, partial, parent-drifted, or train-drifted lineage fails before
  benchmark GPU submission.
- Population refresh now publishes the same canonical aggregate contract as
  the original population builder. The immutable 12,032-row MMLU-Pro and
  756-row MuSR sources are reconverted against the current decontamination
  receipt; a refresh-only schema can no longer strand an otherwise valid
  evaluation before submission. Recovery job `770126` completed at `0:0` in
  60 seconds with zero restarts and empty stderr. Its canonical aggregate file
  is SHA-256
  `dbbbeb2904a7d6d9c5e9fdc06017b97cdf5befaa70c0dba3622bd179386f43f1`
  with self-hash
  `a6df423a114c2611ce6b3af16df1303278225c354fd9bd14e6a9db5d36a93f68`.
- The real-development decision is frozen before scores exist. Both matched
  checkpoints must complete all 12,032 MMLU-Pro rows and all 756 MuSR rows.
  Curriculum order is retained only with nonnegative deltas on both boards, a
  positive unweighted macro, a strictly positive paired 95% bootstrap lower
  bound, and no domain regression worse than one percentage point. This
  development-only receipt cannot authorize architecture promotion or 4B.
- The positive-NLL handoff is now executable rather than manual. A CPU-only
  stager reopens the exact order receipt and both checkpoints, then submits two
  matched fan-outs: eight independent one-H100 MMLU-Pro shards plus one MuSR
  job for each arm. Two CPU merges and a terminal CPU comparator bind all 18
  H100 jobs to `COMPLETED|0:0|0` with zero restarts before applying the frozen
  benchmark decision. Partial submission is cancelled, while a negative NLL
  decision submits no benchmark work.
- A dependency-held continuation reads the live training dispatch and stages
  that handoff against the exact comparison and canonical population jobs.
  This removes the manual gap between a clean NLL decision and real-board
  measurement without reserving an idle GPU or pre-authorizing a favorable
  result. Job `770127` is held on launcher `770106`; it requests no GPU and
  submits the real-board graph only through the frozen positive-NLL condition.
- Architecture promotion remains behind this boundary. Two matched 100M GQA
  jobs now measure curriculum ordering versus a deterministic order control on
  the identical record multiset; they are a data-order falsification, not an
  architecture result. The 4B prohibition remains unchanged.
- A deterministic forty-document qualitative audit of the exact development
  population confirms that the frozen surface score is not semantic pedagogy:
  grounding can include electrostatics and engineering, reasoning can include
  introductory rocks or biographies, and specialization can include basic
  traffic-light explanations or mythology. The exact sample rule, source
  hashes, observations, and resulting 4B-data prohibition are recorded in
  `docs/SAI_CURRICULUM_QUALITATIVE_AUDIT_20260822.md`. The current comparison
  remains an order falsification experiment; it cannot waive a later semantic-
  prerequisite or source-mixture gate.
- The first public math-source audit rejected blind use of FineMath `4plus`.
  Exact shard `5d0b2611...1fe5` contains `104,680` unique rows, but only
  `34.7583%` set `found_math=true`; direct evidence includes incoherent score-5
  algebra with essay-service links plus answer-farm, SEO, gambling, and
  commercial-homework material. `docs/SAI_FINEMATH_SHARD_AUDIT_20260822.md`
  freezes the input and findings. FineMath remains a candidate only after a new
  Sai quality, provenance, deduplication, and decontamination filter.
- FineMath filter V1 then applied a prospectively frozen high-precision policy
  to all `104,680` rows and accepted zero. The dominant cause was a miscalibrated
  upstream language-confidence floor of 0.98: only `367` total rows reached it,
  while `3,114` rows passed every non-language criterion. V1 remains an
  immutable `filter_empty_no_candidate` result; it was not relaxed after the
  outcome. `docs/SAI_FINEMATH_FILTER_V1_RESULT_20260822.md` binds the receipt,
  rejection counts, and exact funnel. The next prospective step is a blind
  human-review ladder at no language floor, 0.90, and 0.95—not training.
- That prospective ladder has now frozen `3,114` non-language-qualified
  candidates and a 192-row blind packet: exactly 64 rows from each of `<0.90`,
  `0.90–0.95`, and `>=0.95`. The packet hides score and stratum; its separate
  key remains closed until labels are complete. Receipt
  `a17dcf57…d6d0` authorizes no training.
- An offline workspace now exposes only the 192 row identities and texts, with
  resumable evidence-backed labels and no external requests, URLs, language
  scores, strata, or hidden key. The post-review decision was frozen before
  labels: two independent reviewers, at least 90% consensus acceptance and an
  80% Wilson 95% lower bound in every included stratum, then the lowest passing
  floor among none, 0.90, and 0.95. If the `>=0.95` stratum fails, FineMath is
  rejected rather than retrospectively relaxed. Any selected rows remain
  non-training candidates pending global deduplication, benchmark
  decontamination, provenance replay, and semantic prerequisite placement.
  `docs/SAI_FINEMATH_HUMAN_REVIEW_WORKSPACE.md` freezes this boundary.
- Bulk code admission now begins with an executable Stack-Edu metadata audit,
  not a download. It pins revision `eeec5caa…814c`, rejects every unlicensed or
  mixed-unallowlisted row, and measures quality score, encoding, length,
  provenance, and duplicate identities before content retrieval. Metadata can
  only nominate blobs for a later content audit; missing current opt-out replay,
  secret scanning, legal review, deduplication, or benchmark decontamination
  keeps training authorization false.
  The complete pinned five-shard Python population measured 25,286,019 rows,
  but only 514,566 (2.0349%) survived this preliminary filter; 20,722,635 were
  marked `no_license`, and 18,536 more were marked permissive without a detected
  license. The independently replayed complete-language evidence is recorded
  in `docs/SAI_STACK_EDU_PYTHON_LANGUAGE_AUDIT_20260822.md`.
- The complete Stack-Edu candidate identities are now frozen separately from
  source content. CPU job `770639` completed in `3,100` seconds with exit
  `0:0`, zero restarts, and empty stderr. It froze 514,566 unique blob
  identities spanning 127,672 repositories and 514,559 unique repository/path
  pairs; the candidate JSONL SHA-256 is `7429c9d4…07c5` and canonical receipt
  identity is `ec6b0aa9…0762`. Exact execution, hash, and population evidence is
  in `docs/SAI_STACK_EDU_CANDIDATE_IDENTITY_AGGREGATE_20260822.md`. This remains
  metadata-only and authorizes no content retention or training.
  Current-release alignment is executable. The new
  `sai-stack-v2-current-python-snapshot-v1` boundary requires the complete
  Python metadata shard set from `bigcode/the-stack-v2` revision
  `e565caa3…90e47` (`v2.2.0`, opt-outs enacted through `2026-07-29`) plus the
  exact dataset card and self-hashed access evidence. The freezer queries the
  authenticated Hub API at that commit and verifies every local file against
  its remote Git or LFS identity; caller-chosen local hashes are insufficient.
  Alignment retains an old candidate only when the exact
  `(repo_name, path, blob_id)` still exists and the current row is permissive,
  non-vendor, and non-generated. Missing rows are treated as removed. This
  closes current opt-out drift only; content-byte
  verification, attribution, secret/PII/malware scanning, global exact and near
  deduplication, benchmark decontamination, and semantic curriculum review all
  remain mandatory. See `docs/SAI_STACK_V2_CURRENT_ALIGNMENT_CONTRACT.md`.
- Authorized acquired code must then pass the separate exact-content bundle
  verifier. It requires a sealed contiguous bundle and ordered index, recomputes
  the Git/SWH blob SHA-1 over `blob <length>\0<bytes>`, checks independent
  SHA-256 and declared length, proves strict UTF-8 round trips, and rejects any
  gap, overlap, missing row, or trailing byte. A valid byte receipt still keeps
  training and 4B authorization false; quality, secrets, global duplication,
  contamination, and semantic placement remain downstream gates.
- Verified bytes now feed a separate bounded safety/quality findings pass.
  High-confidence private-key and credential formats plus invalid control bytes
  are vetoes; personal-email, high-entropy/JWT-like strings, generated markers,
  extreme repetition/minification, and Python-version parse failures require
  review. A row with no bounded finding is still only a candidate because this
  scanner cannot prove the absence of novel secrets, malware, dependency
  hazards, or subtle benchmark-derived code. The exact policy and limitations
  are frozen in `docs/SAI_STACK_EDU_CONTENT_SAFETY_CONTRACT.md`.
- A separate create-only selector now resolves those bounded findings without
  pretending they constitute source admission. High-confidence rejects cannot
  be overridden; every manual-review row requires one exact hashed
  adjudication; bounded-clean rows remain candidates. The selected population
  retains training and 4B authorization false until accuracy, usefulness,
  duplication, contamination, semantic placement, and matched source-addition
  evidence pass. Its contract is
  `docs/SAI_STACK_EDU_SAFETY_SELECTION_CONTRACT.md`.
- An authored programming-curriculum candidate now preserves 111 Rust Book
  chapters and 16 CPython tutorial chapters at exact pinned revisions, in
  publisher order with byte-exact code and license evidence. Its 127-row
  candidate receipt is `80de7bef…08e`; it authorizes no training. Python's
  tutorial explicitly assumes prior programming knowledge, so every Python row
  requires `programming_foundations` instead of being mislabeled as grounding.
  The authored sequence is a prospective pedagogical spine, not a complete
  corpus; semantic review, global deduplication, decontamination, source-addition
  controls, and identical-document order controls remain mandatory. Exact
  evidence is in `docs/SAI_AUTHORED_CURRICULUM_CANDIDATE_20260822.md`. All 127
  rows are now frozen in a salted blind-review packet (`f052ff87…b906`) that
  hides our provisional order/stage key until independent concept labels and
  evidence spans are complete. The two-reviewer adjudicator now verifies all
  evidence spans and preserves separate concept, prerequisite, quality,
  admission, and defect disagreement; no completed labels or PASS exist yet.
- The final 4B mixture boundary can no longer be satisfied by plausible-looking
  64-character hashes. The new relocation-safe v3 validator reopens every
  source manifest, selection policy, license decision, quality audit,
  decontamination receipt, and pedagogical progression receipt; validates exact
  file bytes plus canonical receipt schema/status/self-hash; and rejects links,
  missing evidence, and re-signed drift. Each receipt must also carry its exact
  role-specific positive decision; a generic `status: qualified` cannot admit a
  source. Each decision must also name the exact source-manifest hash it covers,
  preventing cross-source receipt reuse. There is no structure-only v3 mode;
  every validation reopens the evidence root. No v3 mixture passes yet.
- The source-addition gate is now executable rather than a prospective table.
  `sai-compare-source-addition` requires equal training tokens and compute,
  identical model/initialization/optimizer/tokenizer/development evidence, and
  distinct replayed source qualifications. It compares target-normalized and
  UTF-8-byte-normalized held-out likelihood in every development stratum; any
  stratum regression vetoes the source. A likelihood pass still retains
  nothing until real source-disjoint benchmark confirmation completes. The NLL
  receipt now binds each terminal checkpoint plus manifest using the exact
  bundle identity consumed by evaluation. `sai-confirm-source-addition-benchmarks`
  then requires paired complete MMLU-Pro and MuSR development evidence, exact
  checkpoint lineage, a positive 95-percent paired macro lower bound, no
  negative benchmark delta, and no domain regression below one percentage
  point. Only that terminal receipt can retain a source, and it still cannot
  promote an architecture or authorize 4B training.
- The matched curriculum-order experiment is now live. CPU launcher `770136`
  completed at `0:0` after replaying the split and all three streams and
  published dispatch SHA-256 `4d682933…7ee`. Its create-only wall-time receipt
  `895fd3e3…2cc` extended both full arms to the predeclared 18-hour bound before
  release. Exact-geometry canary `770153` completed on one H100 in 294 seconds,
  with zero restarts and empty stderr; its terminal run receipt is
  `f484fcf8…68b1`. Curriculum job `770154` and identical-record deterministic-
  order control `770155` then started independently on `evc22` and `evc24`.
  Both train exactly 499,998,720 tokens with the same model, initialization,
  optimizer, seed, update count, tokenizer, and admitted record multiset.
  Both independently published their first durable checkpoints at optimizer
  step 10: exactly 2,560 sequences each. Their checkpoint manifests bind the
  same code (`16439922…410`), configuration (`a34e96f8…f11`), environment
  (`778d1372…f29`), and model (`ef67fa5d…574`) identities; only the ordered-
  stream and resulting run identities differ. The curriculum checkpoint hash
  is `02fe57b4…e2d`, and the control checkpoint hash is `1fef6028…996`.
- Continuation `770137` failed in five CPU seconds before submitting work
  because Slurm would not accept a new dependency on completed population job
  `770126` after it aged out of the live controller table. The population and
  split jobs remain exactly `COMPLETED|0:0|0` in accounting. A later audit found
  that pending comparator `770156` would have published a raw checkpoint-file
  hash where the benchmark evaluator requires the canonical checkpoint plus
  manifest bundle hash. Pending benchmark stage `770159` inherited that
  deterministic lineage mismatch. Both jobs had zero elapsed time and zero
  restarts, created no outputs, and were cancelled without touching live
  training. The exact scientific inputs and destinations are now staged behind
  corrected CPU comparator `770503` and benchmark stage `770505`, using sealed,
  clean runtime commit `c4b3b011ddee4d5e8aad3b60a30219b799c5b686`.
  Comparator `770503` still depends only on training jobs `770154` and `770155`;
  stage `770505` depends only on `770503` and will independently reopen all
  completed population, split, stream, checkpoint, and manifest evidence before
  it may submit benchmark work. No GPU job was duplicated and no scientific
  identity, record order, token budget, model byte, or evaluation population
  changed. Source now avoids both the aged-dependency and checkpoint-identity
  failures in future continuations.
- Semantic annotation policy v2 and prerequisite taxonomy v3 now require every
  accepted positive concept label to bind at least 16 exact source Unicode
  codepoints. A bare term mention therefore cannot establish that a prerequisite
  was taught. This is only a minimum evidence guard; independent annotation
  review, prerequisite order, concept-density, and later-rehearsal gates remain
  conjunctive. No semantic curriculum has passed yet.
- Semantic audit selection can now run against the already-qualified,
  source-disjoint development split. The selector preserves the same frozen
  salt, four phases, four surface bands, and eight documents per stratum while
  re-reading 94 MB rather than repeatedly replaying the 9.5 GB training
  curriculum. Its 120 selected documents remain an unreviewed audit packet; the
  speedup changes no label, threshold, or training byte.
- The authored-curriculum prerequisite lane now has two exact blind candidate
  reviewers. CPU context jobs `770444` (Qwen3.5-9B) and `770445` (SmolLM3-3B)
  completed with zero restarts and verified all 127 prompts against the same
  hidden-key-free packet. Qwen observed 966–11,620 input tokens and SmolLM3
  952–11,098, both below the frozen 24,576-token ceiling. Independent one-H100
  review jobs `770450` and `770451` both loaded their models but failed on row
  zero after exhausting three invalid structured-response attempts; no label,
  draft, or review receipt was published, and comparison `770471` never ran.
  The repaired runner now preserves every rejected response, constrains output
  complexity, canonicalizes list order and unique whitespace-equivalent source
  quotes, and retains every original evidence threshold. Fresh collision-safe
  jobs `770735` and `770736` preserved exact failures: Qwen repeatedly placed
  evidence-backed taught concepts in both semantic roles; SmolLM3 completed two
  candidate rows before repeatedly recommending `admit` with an empty taught
  set. Neither published a complete result and comparison `770738` never ran.
  Pushed commit `a14e6eae…d387` now gives explicit quoted taught evidence
  precedence over an ungrounded duplicate assumption and conservatively maps
  empty-taught `admit` to `revise`. All evidence thresholds remain unchanged.
  Fresh jobs `770761` and `770762` each completed three replayable rows before
  failing on row 3; comparison `770763` never ran. Their immutable failure
  artifacts show only redundant nested quality fields and unsupported,
  ambiguous, or sub-minimum evidence strings. Pushed commit `2aaf0f3…af5bd9`
  conservatively discards those unsupported strings, retains only unchanged
  unique literal 16-codepoint evidence, and never upgrades a recommendation.
  Fresh collision-safe jobs `770785` (Qwen) and `770786` (SmolLM3) are staged
  independently, with CPU comparison `770787` dependent on both. They may rank
  cross-family disagreements for human attention, but model output still
  cannot qualify labels.
  Both reviewers crossed the former row-3 boundary but terminated at rows 4
  and 5 respectively: Qwen could not ground defect quotes, while SmolLM3
  exceeded evidence bounds and then returned non-JSON text. Comparison
  `770787` never ran. This ends parser iteration rather than weakening the
  evidence contract; the packet now proceeds only through the frozen two-human
  review path.
  `sai-build-authored-review-workspace` now makes that path operational through
  a self-contained offline form. It exposes only salted review identities,
  exact chapter text, the candidate vocabulary, and frozen evidence rules;
  requires every row to be explicitly reviewed; supports packet-bound local
  progress export/import; and emits exactly the existing compiler's JSONL
  schema. It performs no network request and includes neither the hidden key nor
  provisional phase labels. Its receipt still records human review, training,
  and 4B authorization as false.
  Exact failure and recovery evidence is in
  `docs/SAI_AUTHORED_MODEL_REVIEW_RECOVERY_20260822.md`. The final adjudicator now
  accepts neither arbitrary identity strings nor model-review identities: each
  side must bind all 127 completed rows to a distinct human identity
  attestation, exact packet and policy hashes, no model-generated labels, and
  no hidden-key access before label freeze. Model-model agreement can therefore
  accelerate review but cannot admit training data.

## Live scratchpad — 2026-08-21

- Measured full curriculum/split/stream replay exceeded two CPU wall-hours on
  Newton before dispatch. Future curriculum-order launchers therefore reserve
  six CPU hours for the evidence replay; the downstream comparison remains a
  separate four-hour CPU job, and both 500M-token H100 arms retain their
  independently measured 18-hour limits. This changes no data, seed, model, or
  scientific result.

- Exact FLA 0.4.2 Gated DeltaNet and KDA chunk mechanics remain qualified by
  Newton job `768134`. The environment receipt file is SHA-256
  `778d137224671a44acdcc923270dc7478cded5437780a0ea37e19b764a219f29`.
- The benchmark-decontaminated training corpus, mechanically qualified lossless
  48K fixed-geometry tokenizer, and exact binary streams are complete. The
  qualified default tokenizer tree is SHA-256
  `cf4879ee5b3914b4af187abcc93be5678e41ff942e0b0a14f6eeb1a089f6f76d`.
  It is not an empirical tokenizer-capacity winner: the completed tournament
  measured `459,376` English-labeled documents but no representative code,
  math, science, or technical corpus strata. The exact compression tradeoffs
  and the required capability selection boundary are documented in
  [`docs/SAI_TOKENIZER_EVIDENCE_AUDIT.md`](docs/SAI_TOKENIZER_EVIDENCE_AUDIT.md).
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
  Exact decontamination `768892` reached a live source position of at least
  `6,257,876,992 / 14,491,695,743` bytes (`43.18%`) with zero restarts, but its
  maximum RSS had risen to `66,368,664 KiB` inside a 64-GiB allocation. Exact
  source replay proved that implementation stored each SHA-256 shingle as a
  64-character Python string; measured RSS growth left only about 740 MiB and
  projected an unavoidable OOM near halfway. It was therefore cancelled at
  20:16 EDT before infrastructure termination, after `12,457` seconds and
  before publishing an output or receipt. Its sole abandoned partial was
  explicitly resolved as PID `3211846`, measured at `4,150,497,097` bytes,
  permanently removed, and confirmed absent. No scientific decision changed. A
  disk-read heuristic initially overstated progress; the process file
  descriptor is the authoritative scan measurement. Newton rejected a running
  wall-time extension. Conditional CPU fallback `769225` was originally held
  on `afternotok:768892` with compact, byte-equivalent SHA-256 shingle storage at
  commit `540d1080ebc5b1cd2b463d8c137e9d2c71567e82`. It started on evc1 with zero
  restarts at 20:16 EDT. The compact representation held resident memory near
  `48,812,000 KiB`, but exact process-file-descriptor replay at 22:51 EDT
  measured only `3,701,448,704 / 14,491,695,743` source bytes (`25.54%`) after
  2h34m, projecting beyond its immutable eight-hour limit. A deterministic
  ordered fork implementation was therefore built at commit
  `cc5a7fd4c938e66166c1a9fd7aef91b2f51f2877`. A live 20,000-row, one-boundary
  A/B proved that sequential and eight-worker builds produced the exact same
  `101,009,890`-byte output with SHA-256
  `d86c43c32f4a0a6649fbd5045515c798f232ea1829d26fce2f85b374d2fc1212`.
  Parallel job `769636` completed in 134 seconds versus 311 seconds for
  sequential job `769635`, an exact measured `2.32x` speedup. The slow primary
  was then cancelled deliberately at 22:55 EDT after 9,516 seconds, before any
  final output or receipt existed, and parallel recovery `769626` started
  immediately on evc1 with zero restarts and a 14-hour limit. Its source,
  twenty ordered boundaries, final output path, and receipt path are identical.
  The abandoned `2,521,335,906`-byte partial from PID `3216825` was proven
  unreferenced, atomically quarantined, permanently removed, and confirmed
  absent; the new job owns a distinct staging file. The three stream descendants
  use an OR dependency on successful completion of `769225` or `769626`, so the
  parallel lineage now exclusively releases shared Sai stream `769226`, exact
  125M-token Qwen stream `769437`, and exact 125M-token SmolLM3 stream `769455`.
  A pre-execution audit found that the first refreshed-population
  clone `769227` still checked terminal state for cancelled original job
  `768892` inside its script even though its scheduler dependency had been
  repaired. That clone and its three bound descendants `769235`, `769237`, and
  `769238` were cancelled with zero elapsed time and zero restarts before they
  could create any output. Exact mutually exclusive replacements now bind both
  scheduler and in-script custody to the same lineage: population jobs
  `769355`/`769627` follow primary/recovery corpus jobs `769225`/`769626`;
  parent benchmark launchers are `769425`/`769426`; workspace evaluation stages
  are `769440`/`769441`; and terminal comparisons are `769442`/`769443`. Each
  pair targets the same collision-safe artifact and only the branch whose
  corpus lineage completes can run. Workspace launcher `769439` remains
  staged. Two
  malformed first launcher clones (`769233–769234`) were caught while still
  dependency-held and canceled with zero elapsed time and zero restarts; their
  corrected replacements bind the exact fallback population and stream job
  IDs. No recovery job executed concurrently with the primary scan.
  Recovery-bound parent/workspace/benchmark stagers have been rewired to
  population job `769627`; primary-bound branches are now dependency-dead and
  cannot race the recovery lineage. Successful decontamination is followed by
  shared 499,998,720-token stream freeze `769226`. The original dependency-held
  100M launcher `769232` and benchmark stager `769649` were cancelled with zero
  elapsed time and zero restarts after end-to-end DeltaMixer qualification found
  that their older immutable runtime checked only operator-level FLA parity.
  They submitted no child jobs and created no screen result. Current code now
  requires an explicit, non-downgrading scope: `gqa_only` submits exactly one
  reference GQA baseline over the same 249,999,360-token prefix and labels it
  non-tournament; `three_family` fails before output creation or submission
  unless an exact full-DeltaMixer receipt is hash-pinned and independently
  revalidated by both launcher and GPU job. Every selected family now first
  executes an exact B=8, T=2,048, one-update H100 canary; its longer screen is
  released only by that canary's successful completion, and both job identities
  are bound downstream. This neither selects a mixer in advance nor authorizes
  4B. Once a three-family screen is qualified, the
  real-benchmark continuation is
  executable rather than aspirational: after those three checkpoint jobs and
  the refreshed decontamination population close, a CPU-only stager submits
  eight independent one-H100 MMLU-Pro shards plus one independent one-H100
  MuSR job per family, followed by one deterministic CPU merge per family's
  MMLU-Pro shards. That is 27 single-H100 evaluation jobs, three CPU merges,
  and six terminal comparison dependencies across the three families. A final
  CPU job reopens the six row-complete terminal receipts, verifies all 27 H100
  jobs completed cleanly, and emits exact paired family deltas. The benchmark
  launchers require the population's
  admitted-source SHA-256 to equal the sole source SHA-256 recorded by the
  trained checkpoint's token stream. The chain contains no arrays, retries,
  requeues, implicit substitutions, architecture promotion, or 4B authority.
  A red-team replay caught the prior six-job draft before execution: its
  monolithic MMLU-Pro arms repeated a known output-free four-hour failure mode,
  and its comparison stager could race an already-published fast result. Both
  are fixed at commit `f128faaf507ab35b7782225e8aea273d5b7beea8`; the complete
  tree passes 383 tests. The sealed Newton runtime
  `/lustre/fs1/home/sa305415/sai-initiative-runtime-f128faa-r1` contains 213
  files at tree `ff7fba7473785368c2e5274e37c29edfdd02c343`. Stale pending
  stager `769619` was cancelled with zero elapsed time and zero restarts.
  Replacement `769649` was also cancelled dependency-held, with zero elapsed
  time and zero restarts, when its unqualified upstream launcher was withdrawn.
- Full-model FLA qualification is an evidence-bearing NO-GO, not an assumed
  pass. Jobs `769650–769654`, `769658`, and `769659` used independent one-H100,
  no-requeue requests and executed zero optimizer steps. Across fixed seeds,
  all structural mappings, row resets, `scale=1`, and every one of 24 direct
  packed causal-convolution comparisons passed. The direct recurrence isolation
  at seed `20260823` produced GDN normalized-RMSE ratios
  `0.00413/0.00462/0.00457/0.00447` and KDA ratios
  `0.00213/0.00465/0.00565/0.00434` for lengths `1/63/64/65`. Pinned FLA 0.4.2
  requires strict forward ratio below `0.005`, so KDA length 64 is a real
  failure and the existing v1 receipt remains unqualified. All direct
  convolution checks passed; full-layer ratios were about `0.35–1.09%`, with
  sparse maximum-element outliers, so neither threshold relaxation nor a
  mapping-bug claim is justified. Commit `e27a8d723a19cea3c0790568667dee06e5e67b15`
  preserves the failed evidence and gates hybrid screens on a future qualified
  receipt. The family-separated v2 primitive gate is now implemented with the
  upstream FLA forward/backward ratio limits and prospectively frozen seeds
  `20260824–20260826`; calibration seeds cannot be reused, every tensor/case is
  a veto, and GDN/KDA receive separate statuses. It runs no optimizer and does
  not authorize training. GQA-only reference training remains independently
  admissible behind its exact-geometry canary.
  The three immutable v2 executions `769716`, `769718`, and `769720` then closed
  on those exact held-out seeds. Every direct GDN/KDA recurrence forward and
  backward metric passed, but every seed failed the shared BF16 causal-
  convolution gate: observed ratios ranged from `0.00137` to `0.00560` against
  the declared `0.001` limit. Receipt file hashes are respectively
  `2d022fe508d68327a94f62f79bdf1be0b1eda4957fc14dac8a3822e81b65992e`,
  `005f6c880d44dc23cb018d794d266bf86d7c75ada344a1e5169a247468a1284b`,
  and `fa915883f6009fdef430967470626d9b084b365ee49ea45327004e77fed861eb`.
  These receipts remain permanent FAILs. A source audit also established that
  upstream FLA's Torch-reference `0.001` convolution tests cover FP32/FP16, not
  BF16, while its relevant varlen check compares two executions of the same FLA
  operator. Therefore v2 does not establish a Sai mapping failure either: its
  claimed BF16 threshold grounding is invalid and must be replaced
  prospectively with fresh seeds, never relaxed or re-signed post hoc.
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
- First parent-mechanics allocation `769161` ran once on evc37 for 227 seconds,
  with zero restarts, and failed before model-weight loading, inference,
  training, or output publication. Its exact stderr SHA-256 is
  `f1a91a4e0d5f7511f8c1f6607f7586995e6e2be7519fc399c7c537c5e88735bf`.
  Read-only replay established that the checkpoint intentionally separates a
  `248,320`-row padded model embedding/logit vocabulary from a tokenizer with
  base vocabulary `248,044`, 33 added tokens, exact length `248,077`, and EOS
  token `248,046`. The first mechanics code incorrectly equated tokenizer
  length with model rows. The repaired implementation now binds all four
  identities independently in mechanics, stream-freeze, training, and replay
  validation. Second mechanics allocation `769410` loaded all 320 weight
  tensors on evc23, then failed in 49 seconds because Transformers 5.15 returns
  its empty loading-key collections as sets while the validator required the
  `unexpected_keys` container itself to be a list. CPU replay `769420` proved
  exact empty sets for missing, unexpected, and mismatched keys and an empty
  error list. Independent CPU geometry replay `769435` then proved the exact
  loaded text host is `Qwen3_5ForCausalLM` with hidden size `1,024`, padded
  model vocabulary `248,320`, 320 parameter tensors containing exactly
  `752,393,024` parameters, two buffers, and no loading discrepancies. The
  exact CPU forward replay `769445` also completed in 38 seconds: the frozen
  ten-token mechanics prompt produced finite logits of shape
  `[1, 1, 248,320]`, FP32 sum `-324265.375`, maximum `18.125`, and argmax token
  `279`. Thus tokenizer, loading, model-call, and logits contracts are all
  exercised before the fresh H100 allocation; only CUDA residency and the
  sealed mechanics receipt remain allocation-specific. The
  validator now normalizes only list/set/tuple container form,
  still rejects any non-string or disallowed key, and continues to require no
  missing, mismatched, or error entries. Full local regression is 354 passed;
  these are admission-code corrections before any Sai model result, not
  scientific retries. Exact repaired commit
  `d2cdf5f1fc314fd424c0316df278496a8675c157` is sealed on Newton with zero
  writable members. Fresh mechanics `769422` is an independent one-H100,
  no-requeue request with a measured-safe eight-minute limit. Its corrected
  graph is dependency-staged without occupying GPUs: primary/recovery parent
  launchers `769425`/`769426`, accelerated Qwen stream `769437`, workspace
  launcher `769439`, primary/recovery workspace evaluation stages
  `769440`/`769441`, and terminal comparisons `769442`/`769443`. Superseded
  generic-stream jobs `769411`, `769427`–`769429`, and `769431`–`769432` were
  cancelled with zero elapsed time and zero restarts. Every target was absent at
  submission; a mechanics failure cancels all scientific descendants before
  allocation.
- Fresh parent mechanics `769422` completed 0:0 on known-good evc44 in 97
  seconds with zero restarts. The 1,845-byte receipt file SHA-256 is
  `e1767a706d3e7aafefb2707cdfcdd8b4af55699efb81c9997f0ea0ee13268ffb`;
  its canonical internal receipt is
  `d94e62c84be0625b3479b671fcd63a264b3358d7713975fb0718c6bfc25ee8a0`.
  Independent replay against the sealed snapshot passes. It binds
  Transformers `5.15.0.dev0`, Torch `2.6.0+cu124`, CUDA 12.4, H100 capability
  9.0, all 320 parameter tensors and two buffers on CUDA:0, finite
  `[1,1,248320]` logits with argmax 279, peak allocation 1,807,927,296 bytes,
  unchanged model state, zero backward/optimizer calls, and no 4B execution.
- The Qwen factor consumes exactly `61,035 × 2,048 = 124,999,680` ordered
  training tokens. A dedicated 125M-token freezer now materializes only that
  exact prefix plus the 256-sequence canary prefix instead of spending CPU,
  storage, and walltime on the unused 375M-token tail of the generic 500M
  stream. It preserves tokenizer, source order, boundary masking, and every
  byte consumed by training; its receipt explicitly rejects any extra prefix
  or wrong total. Corpus hashing is streaming rather than a 14.49-GB
  `read_bytes()` allocation. Full regression after this acceleration is 355
  passed. Exact acceleration commit
  `88275bd9550bd1789f783389c218da188369805b` is sealed on Newton with zero
  writable members.
- The conditional SmolLM3-3B cross-family confirmation consumes the same exact
  `124,999,680`-token prefix. It now has its own 125M-token freezer with the
  identical no-unused-tail rule, streaming corpus hashing, restored-model
  manifest/receipt replay, exact Smol tokenizer identity (`128,256` vocabulary,
  EOS `128,012`), and no GPU exposure. This removes another 375M tokens of
  unnecessary preparation if Qwen earns promotion. Full regression is 356
  passed. Exact commit `a74e1e6e54ad1c57842eafcedb1882348cb6d095`
  is sealed on Newton with zero writable members. Superseded generic stream
  `769229` was cancelled with zero elapsed time and zero restarts; accelerated
  Smol stream `769455` is dependency-held on the same mutually exclusive
  primary/recovery corpus lineage. Conditional release jobs `769457`/`769458`
  are now staged behind the matching primary/recovery Qwen comparison,
  population, and Smol-stream branches. They target one collision-safe run
  root and submit the 32-job, independent-one-H100 cross-family graph only if
  the reopened Qwen receipt passes its predeclared gate; a measured Qwen fail
  exits before any Smol H100 request.
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
  Commits `29c1e1c`, `dfa4150`, and `b1980fa` now complete the unscheduled Smol
  execution path: two independent canaries, two matched full training arms,
  eight independent MMLU-Pro shards plus MuSR per arm, deterministic merges,
  and a paired terminal comparison. The cross-family comparator must reopen a
  passing, hash-valid Qwen factor receipt before it can describe a Smol pass as
  cross-family confirmation. The graph is fail-closed against partial launcher
  submission, uses one H100 per scientific job, and still records both
  `four_b_training_executed=false` and `four_b_training_authorized=false`.
  This is executable preparation only; no Smol GPU job has been submitted.
  Commit `72289ca` additionally provides one fail-closed CPU release that can
  open the passing Qwen comparison and submit the complete Smol mechanics,
  parent evaluation, matched training, workspace evaluation, and comparison
  hierarchy. The hierarchy contains 32 eventual independent one-H100 jobs and
  cannot release on a failed or re-signed Qwen receipt. A clean read-only
  Newton checkout of this exact commit is sealed at
  `/lustre/fs1/home/sa305415/sai-initiative-72289ca` with 226 regular files,
  zero symlinks, and zero writable members. No release job has been submitted.
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
The data-first source, mixture, curriculum, and promotion boundary is specified
in
[`docs/SAI_4B_DATA_MIXTURE_PLAN.md`](docs/SAI_4B_DATA_MIXTURE_PLAN.md)
and
[`docs/SAI_DATA_CURRICULUM_CONTRACT.md`](docs/SAI_DATA_CURRICULUM_CONTRACT.md).
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
