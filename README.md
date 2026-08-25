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

The final-corpus size is now a **quality ceiling, not a volume target**. Sai may
retain at most 2,000,000,000,000 exact UTF-8 text bytes at the final ledger, and
it may retain substantially less. No source, duplicate, synthetic example, or
low-confidence row is admitted merely to fill that ceiling. Exact post-rewrite
bytes, not upstream archive sizes or token estimates, determine compliance.

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

A complete metadata census now replaces extrapolation from that coverage
queue. Across all 983,004 records, the duplicate graph yields 941,691 connected
components. After choosing the best metadata representative per component,
requiring `pd`, `pdus`, or `cc-zero` rights evidence, and enforcing 2,000--2M
tokens per work, the nested English tiers contain **382,961 works / 78.690B
tokens at OCR≥95**, 418,604 / 87.029B at OCR≥90, and 438,861 / 91.898B at
OCR≥80. The high-confidence translation-discovery tier adds 91,272 non-English
works / 23.628B tokens at OCR≥95. These counts identify what is worth selective
materialization; they do not admit the raw 870GB archive or flatten literature
through automatic translation. The source-text-free receipt has canonical hash
`11c0426b07a478fecf790a6622bc5e1d4cd2bedd809db25f1996a3905b03a43e`
and file SHA-256
`1b86b34450038d2ae72e7ee4f35a9fc07739ed5ad347b5b0ef4a89096c1ab61a`.
It was downloaded back and byte-replayed from Hugging Face commit
[`b9ea6ce404438b5917cad3caee05fe9c6b0bcf55`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/b9ea6ce404438b5917cad3caee05fe9c6b0bcf55)
and is hash-matched in the authorized Stokes evidence root.

The strict OCR≥95 English tier is now frozen as an exact **382,961-row,
78,689,684,117-token** materialization manifest. It contains no book text; its
208,401,103 bytes bind each barcode to its metadata hash, token count, OCR,
subject, genre, and rights evidence. Manifest SHA-256 is
`986e87999763fef2e6271f869a81b7a23fbe8fc2b7c3e77fbf38da1d4d6fa02c`
and canonical receipt is
`c32619e44bad583781e6e5280c2fe3031ab215fbc94a81e73d8cd20279e07e15`.
The receipt, but not the gated manifest or text, was downloaded back and
byte-replayed from Hugging Face commit
[`717a1a4f21618177cdb76e41e7e89d804f7dccb0`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/717a1a4f21618177cdb76e41e7e89d804f7dccb0).
The complete manifest and receipt are hash-matched on private Stokes storage.

Selective private materialization has passed both local and Stokes one-parent
pilots: each found the same 107 selected works, excluded zero, and materialized
16,527,888 enriched tokens. The Stokes pilot compressed those works to 22.6MB
from a 127.7MB source parent. Full jobs `818439_[0-63]` now process all 4,916
pinned parents as 64 independent CPU shards, with aggregate job `818440`
verifying every source hash, selected barcode, lineage receipt, and private
output hash. The output remains non-training-ready pending global benchmark
decontamination, semantic deduplication, and representation checks, and its
early-access terms prohibit Hugging Face redistribution.

A complete second-pass mechanical gate is now dependency-staged behind that
aggregate. CPU array `818505_[0-63]` will hash-verify every private Parquet and
scan all materialized rows for contextless answer keys and score sheets,
control/Unicode corruption, repeated OCR gibberish, link or markup fragments,
metadata-only forms, and duplicated boilerplate. Aggregate job `818506` then
requires exact barcode coverage and decision-stream hashes across all 64
shards. The gate copies no source text; every non-pass row is withheld from
direct admission, while even a pass remains non-training-ready until semantic
quality, decontamination, and global deduplication also close.

The mechanical decisions now feed a physically separate private candidate
corpus rather than relying on a downstream flag. Dependency-staged CPU array
`818507_[0-63]` will revalidate each materializer and gate receipt, copy only
`pass_mechanical_gate` rows, and hash every resulting Parquet. Aggregate job
`818508` then requires exact source, retained, and excluded row accounting,
unique barcodes, and byte-identical outputs across all 64 shards. Non-pass text
is never copied into this candidate corpus. This is still only a mechanically
filtered source: it deliberately remains `training_ready=false` until benchmark
decontamination, global semantic deduplication, and semantic admission finish.

For bulk pretraining, the user has now replaced that publication-grade blocking
policy with a practical two-tier contract. The base corpus requires pinned
English metadata, OCR score at least 95, `pd`/`pdus`/`cc-zero` rights evidence,
the connected-work duplicate policy, and a `pass_mechanical_gate` decision.
`institutional_books_practical_admission.py` replays those exact receipts and
emits a private allowlist over every surviving filtered Parquet row, excluding
any residual exact-content duplicate deterministically. It requires no per-book
LLM judgment and marks that private allowlist practical-pretraining-ready while
keeping redistribution and 4B training unauthorized. Official-benchmark
decontamination continues in parallel and blocks benchmark/evaluation claims,
not construction of the bulk English/non-slop stream. Hermès explanations,
concept graphs, independent model review, and cross-domain synthesis are now a
premium overlay instead of a prerequisite for every ordinary source book.

The same fast path now covers PleIAs Common Corpus. The practical locator scan
reopens every hash-pinned parent directly, accepts only rows labeled English
with an explicit reusable license, rejects short/empty/malformed rows and every
deterministic junk signature, and writes source-safe locators instead of a
second multi-terabyte text copy. Each fixed identity partition now orders its
assigned parents by canonical source hash, retains 100% of eligible rows, and
stops reading only after its 15,625,000,000-byte allocation is full. This
cross-directory byte-cap sampling replaces the slower full-reservoir/20%-row
pass: it reads several times fewer source bytes while supplying more usable
candidate mass. All 128 source-disjoint CPU shards can run concurrently and do
not wait for Hermès labels or the metadata-audit policy. The locator scan is a
candidate pass; practical readiness is declared only after exact-content
deduplication and final byte balancing. The admission pass uses an on-disk exact
hash index, keeps the lexicographically smallest source identity per identical
content hash, verifies the signed 1,548-identity global quarantine registry,
removes every matching known-bad content hash before deduplication, and
subtracts exact admitted Books UTF-8 bytes from the shared
2,000,000,000,000-byte ceiling before writing final PleIAs locators. Byte-cap
selection follows canonical content-hash order globally and only then routes
each winner into its source-local output shard. The Books reservation therefore
cannot erase a contiguous tail of source partitions, and SQLite can stream its
primary-key order without an additional output-shard sort. This denylist and
balancing work reads only compact locator hashes and does not slow or restart
the live source scan. Admission also reconstructs each shard's deterministic
scanned-parent prefix and rejects any locator whose repository, revision, path,
or parent SHA differs from that exact manifest assignment. Official benchmark
cleanliness remains a separate evaluation-claim axis.

Execution snapshot at 2026-08-25T07:21Z: 122 PleIAs practical scan shards have
closed with zero scientific errors. Their signed receipts alone account for
**1,906,249,975,423 selected UTF-8 bytes** across 4,880 complete parents. Six
remaining original tasks were separately proven to be infrastructure failures:
their CPU counters stopped advancing and each process retained a dead HTTPS
socket in `CLOSE-WAIT`. They had no receipt or final locator. Only those six
tasks were canceled, and their six unclosed partial Parquets (437,552,767 bytes
total) were permanently removed; those partials are not locally recoverable.
Identity-preserving replacements `822087`--`822092` now run from immutable
commit `f4d27458b942b8bd4414b472a512db847d4f8bbc` with bounded connect/read
timeouts, closed response custody, exact-hash retries, and unchanged source and
sampling bytes. The replacement downloader independently replayed exact 8,710-
and 104,183,282-byte Hugging Face objects on Stokes before launch. Shard
allocations are still open, so this
is measured work in progress rather than admitted corpus size. Institutional
Books practical admission has
now closed successfully: **382,072 rows, 222,099,976,155 logical UTF-8 text
bytes, and 69,795,954,639 enriched tokens** are admitted for private practical
pretraining. Its 284,845,639-byte manifest independently rehashes to
`8cd4981dcfc2349ef36e4f53b4ee5ff6555df8909d917f389e44deebc9d8e992`,
and its canonical receipt rehashes to
`904b4f64801239fe492d025990245c6773ff2cbf3de5ce64017fd25e6ff9bc83`.
This puts measured Books plus live PleIAs candidate mass at
**2,128,349,951,578 bytes** before the six replacements, pending cross-source exact-dedup, and
two-trillion-byte balance pass. The final admission still must prove the signed
post-dedup total is within 1.9--2.0 trillion bytes; crossing the raw candidate
floor is not represented as completion. To concentrate scheduler slots and shared storage
bandwidth on this critical path, 122 nonblocking metadata-recovery workers were deferred after
their exact identities, progress tails, 86 completed-receipt hashes, and resume
rule were preserved in durable Stokes evidence. An automatic redispatch was
also stopped, leaving zero metadata-recovery tasks active and the completed
receipts intact.

The repaired premium Books path now runs independently of bulk admission. A
pathological archive record with more than 64 ISBNs exposed a prompt-metadata
bound in the first 8,192-work population attempt. Sai now preserves the full
raw metadata-row hash while passing Hermès a stable first-unique bounded view
of identifiers and duplicate barcodes. Twenty-two focused tests cover this
normalization. Replacement population job `821500` completed successfully in
19m44s with all 8,192 selected works and 1,206,849,389 candidate tokens; the
single measured-safe Hermès lane is live while all later conservative decision,
independent Nemotron, agreement,
decontamination, subdocument-signature, and evidence jobs are dependency-staged.
The remaining 31 Hermès array lanes are now released behind an exact serial
`afterok` chain from lane 0 through lane 31. This preserves the measured
one-lane provider limit and every frozen identity while removing manual gaps;
it cannot increase concurrent API pressure or duplicate a completed judgment.

The practical corpus now also has a separate educational-code overlay instead
of misclassifying programming-language rows as non-English prose. Common Pile
Stack-Edu revision `c354dbe8…54de` contributes 95 exact compressed parents
(83,037,672,896 physical bytes) spanning Markdown and fourteen programming
languages. The locator policy accepts only integer educational scores 3–4,
UTF-8 source rows, non-vendor and non-generated files, and complete permissive
license sets drawn from the frozen Stack-Edu allowlist. It excludes bounded
secret/credential matches, personal-email candidates, JWT/high-entropy tokens,
extreme minification or repetition, generated-code markers, and Python that
does not parse as Python 3. Like the prose path, it preserves only exact parent,
row, repository, license, language, score, byte-count, and content-hash
locators—not source text. Array `821719_[0-94%95]` is staged behind the live
PleIAs scan so it reuses released CPU slots rather than competing with the 2 TB
critical path. A separate on-disk exact-dedup admission then enforces the global
quarantine registry and a 150,000,000,000-byte code ceiling. Code admission is
an overlay candidate until its scan and signed admission receipt close; it does
not change the running PleIAs bytes or authorize the 4B run.

The admitted-code publication path is now implemented at commit
`b345d3c96853fa1671377a442fdeb917e07b30b7`. It validates every signed
Stack-Edu output descriptor, refuses Parquet containing source text, publishes
only the 32 compact locator partitions, and replays every remote LFS size and
SHA-256 before publishing the admission metadata. The full repository passes
**1,340 tests**. Dependency-bound jobs `821826_[0-31%16]`, `821827`, and
`821828` are staged after admission job `821820`; their launch receipt is
`f0f7b525cd99ce78201fbc1e50357e75c55f3751959a31c59e22b4e960945c9a`.
No publication job can start before its exact prerequisite succeeds.

Final locator shards preserve source locality: after global content-hash dedup,
every winning row is assigned by canonical source-path hash, so all retained
rows from one upstream Parquet stay together. The transient stream reader uses
that property to download each pinned parent once, verify its full size/SHA and
every selected row's metadata/content hash, emit ordinary `text` JSONL directly
to the tokenizer or trainer, and delete the temporary parent before moving on.
It writes only a text-free replay receipt. This avoids both a second 2 TB copy
and the pathological alternative where every worker would touch thousands of
remote parent files.

The practical graph is live. Initial full-replay array `820240` was stopped
before any final locator or receipt existed after measured network throughput
proved it inefficient; its 2.85 MB of unclosed partial files were permanently
removed. The first byte-cap staging `820410` was also stopped before a final
locator existed when Slurm denied extension of its unsafe 12-hour limit; its
0.50 MB of unclosed partials were removed. Final PleIAs locator array
`820530_[0-127%128]` launched all 128 fixed source partitions with the same
scientific bytes and a 24-hour termination margin; all 128 are now running
concurrently across eight Stokes nodes. Book practical admission `820358` is
complete, and combined
PleIAs exact-dedup/byte-balance admission `820649` remains dependency-staged
after all 128 `820530` identities close. The complete repository regression
suite passes: **1,338 tests, 2 dependency warnings**.

Publication is dependency-staged rather than manual. After combined admission,
128 independently retryable workers validate that each final PleIAs Parquet is
a text-free locator table, upload it beneath
`training/practical/pleias/20260826-r3/` in `Godlydonuts/Sai`, and verify exact
remote LFS size and SHA-256. A remote aggregate replays all shard receipts and
the current repository head. The final metadata job then publishes the Books
allowlist, both practical admission receipts, and the locator publication
receipt beneath `training/practical/metadata/20260826-r3/`. No private book text
is uploaded; the public dataset holds the exact indexes needed to reconstruct
PleIAs from its pinned upstream revision while the Books text remains in its
authorized private Stokes root.

The final readiness audit does not treat a merely successful pipeline as a
2 TB result. It recomputes Books + PleIAs rows, source tokens, and logical UTF-8
bytes from the two signed admissions; reconciles every PleIAs collection and
rights count; requires canonical content-hash cap selection and complete
row/byte/token reconciliation across every logical output partition; binds the
Hugging Face publication receipt; and requires the
combined corpus to contain **1.9–2.0 trillion exact text bytes**. An underfilled
corpus fails instead of being relabeled complete. Benchmark decontamination
remains false and blocks evaluation claims, while 4B training remains explicitly
unauthorized.

The next book stage is now code-frozen as a bounded **8,192-work semantic
population**, not a bulk promotion of every mechanical pass. After `818508`, it
replays the exact private filter and pinned 983,004-row metadata file, then uses
a stable round-robin over normalized subject × genre × four length bands. This
forces broad intellectual and stylistic coverage while preventing the largest
archive categories from dominating by frequency. Each chosen work contributes
only a deterministic beginning/middle/end excerpt to the private Hermès review
queue; all full text remains on Stokes. The population is explicitly
non-training, non-publishable, and still requires Hermès semantic judgment,
rights review, benchmark decontamination, and global semantic deduplication.
Unselected books are not inferred to be good or bad—they simply receive no
admission claim. This scope is now machine-enforced in both the population
receipt and aggregate validator: semantic decisions can apply only to selected
identities, and any receipt that implies a judgment about unsampled books fails
closed. The repository implementation and tamper-aware replay entry point are
`institutional_books_semantic_population.py` and
`sai-build-institutional-books-semantic-population`.

This stage is dependency-staged on Stokes as CPU job `818511`. When its exact
population receipt closes, Hermès array `818512_[0-31%16]` will cover all 256
identity shards with at most 16 simultaneous single-CPU lanes and resumable
per-work receipts; it cannot run before the private population exists.
Aggregate job `818513` then requires one valid judgment per selected identity,
the exact nonempty shard-summary set, matching model/rubric hashes, and full
usage accounting. Invalid, malformed, or inconsistent model outputs are
retried and never converted into admissions. The aggregate deliberately keeps
`training_ready=false`: Hermès supplies structured semantic triage, not final
authority.

Conservative decision job `818517` is staged behind that aggregate. Its frozen
policy advances only allowed-rights, English, zero-risk `retain` judgments with
at least 900,000 ppm confidence, quality 4/4 for both overall value and OCR, and
4/4 in at least one of knowledge density, literary value, or historical value.
Science, mathematics, engineering, medicine, law, reference, and textbook works
also require factual reliability 4/4. All other rows receive an explicit
quarantine, rights, risk, translation, or quality hold. Even the survivors are
only candidates for independent semantic and representation verification; no
Hermès judgment directly admits training text.

Job `818518` is staged behind the conservative decision receipt and constructs
a second private population containing only those strict survivors. It replays
every decision hash against the original candidate identity and preserves the
bounded excerpts without publishing them. That subset is the sole input
eligible for an independent model-family pass; a survivor count of zero remains
a valid negative result and must not be repaired by relaxing the policy.

Independent array `818519_[0-15%8]` is dependency-staged behind that subset and
replays the same source-bound book rubric with
`nvidia/nemotron-3-ultra-550b-a55b`, not the Hermès/OpenRouter model family.
Aggregate `818520` verifies every independent receipt, model identity, rubric
hash, shard summary, and usage total. This remains a verification measurement;
only later agreement and decontamination logic may advance bytes.

Cross-family comparison job `818521` is staged behind the independent
aggregate. A work becomes only a `consensus_candidate` when both model families
independently satisfy the full conservative quality policy, agree on the exact
genre, and share at least one domain. Any quality, genre, or domain disagreement
becomes an explicit hold. Consensus still does not bypass benchmark
decontamination or global semantic deduplication and remains
`training_ready=false`.

Full-source decontamination job `818523` is staged behind cross-family
agreement. It reopens the exact mechanically filtered Parquet, verifies each
selected full-text SHA-256, and screens the entire work against the pinned
official benchmark boundary—not just the review excerpt. Exact 13-word or
eligible code-shingle overlap is held. Its output contains only identities,
hashes, byte/token counts, and overlap counts; global semantic deduplication is
still required before any admission.

Evidence job `818524` is staged behind the full-text screen. It copies a fixed
allowlist of text-free aggregates, decisions, agreements, and clean identity
manifests into the authorized durable evidence root, then seals every byte in a
hash manifest. Candidate excerpts, compiler evidence quotes, and full book text
are explicitly excluded. A separate local collector is already waiting for the
same source-safe files so no temporary shard is the sole evidence copy.

The benchmark-disjoint survivors now also feed source-safe subdocument signing
instead of bypassing corpus-wide boilerplate control. CPU array
`818571_[0-63%32]`, dependency-staged after full-source screen `818523`, reopens
each private filtered shard, selects only identities present in the clean
decontamination manifest, replays the full-text SHA-256, and losslessly segments
the book at natural boundaries. It writes sixteen normalized-hash partitions
containing only component/shard/book/chunk locators, character spans, lengths,
code flags, and signed digests—never source text. Aggregate `818572` requires all
64 shard identities, verifies every partition byte/SHA-256, and proves the
signature document count equals the exact benchmark-disjoint book count. Both
jobs are CPU-only, requeue-disabled, and remain nontraining inputs to the final
cross-source decision.
The clean book manifest and final private rows preserve the independently agreed
genre and shared semantic domains, as well as the exact agreement and benchmark
screen record hashes. Deduplication therefore cannot erase the quality and
pedagogical metadata needed by the later spiral scheduler. The two-family
agreement also retains source-text-free work/edition candidates, quality floors,
complexity ranges, curriculum-band votes, shared concepts and prerequisites,
and only concept edges recovered by both judges; edge evidence is stored as
quote hashes rather than excerpt text. Final shard receipts count exact bytes by
PleIAs stratum and book genre/domain, making broad-coverage shortfalls measurable
before sampling.

The overall corpus target is now a **decimal 2TB maximum, not a quota to fill**.
If high-confidence gates yield 700GB, 1.2TB, or any other smaller amount, Sai
trains on that smaller verified corpus rather than padding with weak web text,
unreviewed books, or synthetic repetition. Quality, provenance, coverage, and
learnability take priority over byte count.

The durable data catalog is
[`Godlydonuts/Sai`](https://huggingface.co/datasets/Godlydonuts/Sai). It separates
upstream source references, model judgments, verified representations,
curriculum artifacts, and final training shards. The registry never treats a
downloaded dataset as training-ready and records reference-only sources without
copying bytes when their terms prohibit redistribution.

Hugging Face may display an inherited Parquet split named `train` or `test`.
Those names describe the upstream file layout, **not Sai admission state**. A
row under an upstream `train` split can still be corrupt, contextless, duplicated,
non-English, contaminated, or rights-blocked; an upstream `test` split is never
silently treated as Sai evaluation data. Only a future hash-receipted
`verified/` → `curriculum/` → `training/` release may be consumed by training.

#### Materialized Hugging Face source-lake checkpoint

The first physical source-lake boundary is now complete. At exact Hugging Face
dataset head
[`cc8576fbb3f949bdaf59049a150c1fa1d35f47c3`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/cc8576fbb3f949bdaf59049a150c1fa1d35f47c3),
Sai has **13,974 byte-identical LFS source shards containing
8,802,247,613,960 bytes (8.802247614 TB; 8.0055975686 TiB)**. This exceeds the
current 8.5 TB decimal registry-capacity target by 302,247,613,960 bytes and the
earlier 8 TiB binary target by 6,154,591,752 bytes. Every included
destination object was replayed against the size and SHA-256 of its pinned
upstream object; the replay found zero identity mismatches.

| Materialized source family | Files | Bytes | Snapshot status |
| --- | ---: | ---: | --- |
| PleIAs Common Corpus | 10,000 | 4,489,486,652,558 | Complete selected data snapshot; per-row rights required |
| FinePDFs | 1,250 | 3,082,436,502,565 | Deterministic partial snapshot at storage boundary |
| Common Pile, 31 families | 845 | 540,438,290,489 | Complete selected source snapshots |
| UltraData-Math L1 | 1,485 | 366,622,518,811 | Complete source snapshot |
| Nemotron specialized reasoning | 219 | 244,286,609,368 | Complete source snapshot |
| Nemotron specialized v1.2 | 90 | 53,621,158,028 | Complete source snapshot |
| FineMath-4plus | 64 | 18,365,184,633 | Complete source snapshot |
| Nemotron Legal v1 | 21 | 6,990,697,508 | Complete source snapshot |
| **Total** | **13,974** | **8,802,247,613,960** | **8.5 TB and 8 TiB physical targets met** |

Hugging Face accepted five deterministic FinePDFs copy batches and then
returned a public-storage-quota `403` before batch six. No partial sixth batch
was committed. Sai freezes the accepted head rather than silently changing the
selection. The file-level manifest contains 13,974 rows and has SHA-256
`56dc0d512db07aa26533decce8efd86bcb1b705355cd40069fdf9cb6311b5665`;
the aggregate receipt is
`0715eefc3c3bda8ee800fc4c80155df461055da3bbf2a473ad6c93cf93bea9d8`.
Both live under
[`artifacts/sai_hf_materialized_source_lake_20260825_r1`](artifacts/sai_hf_materialized_source_lake_20260825_r1),
and the executable remote verifier is
[`src/sai/data/hf_materialized_source_lake.py`](src/sai/data/hf_materialized_source_lake.py).
The receipt, updated dataset card, and a four-part text publication of the
manifest were remotely replayed at Hugging Face evidence head
[`122b98e0aa38130b0165ed3166cc7a569c3cddf0`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/122b98e0aa38130b0165ed3166cc7a569c3cddf0).
Ordered concatenation of the four remote parts reproduced all 13,974 rows and
the exact manifest SHA-256. The parts descriptor has canonical receipt
`3287b272bd78bd12fca1b9a580928a8b0556815ecc404f57b3cb6ee401632c7a`.

After the manifest-covered practical scans no longer required duplicate HF
custody, public-storage quota prevented even a 12 MB connection-curriculum
upload. Sai therefore reclaimed exactly **12,724** current `/data/` mirror
objects totaling **5,719,811,111,395 bytes** and compacted repository history.
Every removed path, byte size, and LFS SHA-256 matched the tracked source-lake
manifest; all remain reconstructable from their pinned upstream repository,
revision, path, and source SHA-256. The operation preserved all **110** small
source manifests/READMEs plus every registry, evidence, curriculum, and training
object. Its exact compressed deletion ledger and signed preflight/completion
receipts live in
[`artifacts/sai_hf_source_mirror_reclamation_20260825_r1`](artifacts/sai_hf_source_mirror_reclamation_20260825_r1)
and were copied to the authorized Stokes evidence root. The compact HF dataset
now holds 622 current files and 2,084,546,276 bytes at head
`2a05f42030c209c5f1c5221629bb44751c782c06`. Historical HF commit IDs in this
document remain provenance labels, while their byte evidence is preserved by
the tracked manifests and durable receipts rather than retained repository
history.

This is a major storage and custody milestone, not a training-readiness claim.
These are compressed source candidates, not eight trillion tokens and not a
final mixture. Global rights adjudication, English routing/translation,
benchmark decontamination, exact and semantic deduplication, Hermes quality and
pedagogy compilation, tokenizer accounting, curriculum assignment, and final
Parquet publication remain fail-closed. Stokes's 10 Gbps connection is reserved
for transformations and verification that cannot use zero-download Hugging
Face server-side copies; its network link is not the present bottleneck.

The physical lake is now joined to the exact reservoir rights inventory in a
38-source fail-closed admission matrix under canonical receipt
`e70b65ebec4d451be5d4a7094fe798e1154019a0db79cf64d99ec1ff6ee26ab6`.
It accounts for every one of the 13,974 materialized files and all
8,802,247,613,960 bytes. Of those bytes, 5,027,859,142,584 require per-row
license evidence, 3,100,801,687,198 have recognized declarations whose
obligations still must be applied, and 673,586,784,178 require source-terms
resolution. Every source separately exposes incomplete language/translation,
decontamination, exact and semantic deduplication, full-population Hermes,
representation, prerequisite, and spiral-curriculum gates. Physical custody can
therefore never be mistaken for silent admission.

Source-lake retention is evidence-based, not permanent. A file is deleted when
the whole exact object is proven unusable, or after every retained row has been
published in a byte-hash-verified filtered replacement. Mixed shards are not
deleted merely because an audit finds bad rows: those row identities are
excluded immediately, then the raw shard is reclaimed only after its good rows
have replacement custody. Each material deletion must record the remote path,
object SHA-256, bytes removed, decision evidence, replacement path and hash when
applicable, deletion commit, and whether recovery remains possible upstream.
Provenance manifests, benchmark-boundary versions, and other replay evidence
are retained even when they are small; deleting an extra pointer to the same LFS
object is forbidden when it saves no object bytes but breaks historical replay.
Transient acquisition caches are a separate boundary: a pinned,
upstream-recoverable cache may be reclaimed without asserting whole-source
unusability when it was never authoritative Sai custody, no active or admitted
derivative depends on it, and the source-safe audit and recovery coordinates
are already durable.

#### Full FineMath-4plus census

Sai has now scanned the entire materialized FineMath-4plus snapshot rather than
extrapolating from an audit sample. Sixty-four independent Stokes CPU jobs
hash-verified and scanned **6,699,493 rows**, **34,126,971,204 UTF-8 text
bytes**, and **9,573,187,002 upstream tokens**. All jobs exited successfully;
the dependency-bound global aggregate completed in 15 seconds.

The global content census found **zero byte-exact duplicate rows** and only
**seven normalized duplicate rows** after NFKC, case folding, and whitespace
normalization. Upstream marks every row `en`, but its own language confidence
places 349,840 rows below 0.50 and another 591,870 between 0.50 and 0.70.
Only 2,339,612 rows contain a nonzero math-extraction feature, demonstrating
that the source name and provider quality score cannot replace direct content
qualification.

Three nested, non-admission measurement profiles quantify the available
quality/length headroom:

| Profile | Rows | UTF-8 text bytes | Upstream tokens |
| --- | ---: | ---: | ---: |
| Broad: English confidence ≥0.70, score ≥3, 64–32k tokens | 5,734,795 | 29,622,108,077 | 7,941,391,804 |
| Core: English confidence ≥0.80, score ≥4, 128–32k tokens | 4,879,600 | 26,493,021,099 | 6,949,987,393 |
| Elite: English confidence ≥0.90, score ≥5, 128–32k tokens | 250,540 | 1,570,280,996 | 402,534,176 |

The complete source-safe publication receipt is
`bb578f5e969e8d15d96ae40ae3511d4dd6d2d9c42e834e5c641204719d53e4c2`.
All 64 shard receipts and the aggregate replayed byte-for-byte at Hugging Face
commit
[`db79c6bb4e7752aee2de8ce2414fcf5ef709e5c1`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/db79c6bb4e7752aee2de8ce2414fcf5ef709e5c1).
These profiles remain `training_ready=false`: official-boundary
decontamination, semantic/subdocument deduplication, content-density analysis,
Hermes judgment, rights obligations, and final curriculum assignment are still
required.

#### Conservative FineMath candidate qualification

The next population-wide pass intersected score 5, English confidence ≥0.90,
explicit `found_math=true`, 128–32,767 upstream tokens, at least 512 UTF-8
bytes, and a valid source host. From all 6,699,493 rows it retained **56,654
mechanical candidates**, **434,498,432 text bytes**, and **117,445,585 upstream
tokens**. This is a candidate filter, not a quality-admission shortcut.

Exact matching against the frozen 27,979,728-word-shingle and 475,804-code-
shingle official benchmark boundary then retained **52,277 rows** and rejected
**4,377**. The cause-complete receipt reports 1,422 rows with word overlap,
3,290 with eligible-code overlap, and 335 with both; the candidate population
had zero normalized-exact duplicates. A separate 32-CPU replay independently
reproduced receipt
`b0ba86aaa60dddfdfae6653882d489fde1ecf3ab0f043d9a3954bdd38e191277`
and output SHA-256
`c61a840375572bca1a9872d50d99c66ee0452f8f9be4aacd89c1cd7af5d84a7a`.

After a contextless-MCQ-answer-key failure mode was identified in the broader
source lake, a source-agnostic high-precision filter was added and replayed over
all 52,277 survivors. It found zero instances matching the strict bare-key
signature in this conservative FineMath subset, so it changed no document
identities. That does **not** absolve the wider lake: completed Hermès judgments
already flag 36 answer-farm rows, 149 SEO/content-farm rows, and 97 corrupted
rows among 2,243 audited documents. These are hard admission exclusions. The
post-filter 512-row semantic audit population is frozen at receipt
`de365117bee119d224196a3a712518a2814214e130580bf643ef261c56327e1b`.

That exact 512-row population is now under a three-perspective Hermès semantic
audit rather than being inferred from the FineMath source label. The first
four-request geometry produced repeated upstream HTTP 429 responses and only
one sealed judgment; it was stopped without deleting that receipt. A later
OpenRouter transport trial preserved the same model, rubric, population, and
identity geometry, but its first complete attempt returned empty content for
all 15 requested judgments. A 21-minute retry produced no additional receipt,
so that transport was terminated while the existing valid receipt remained
immutable. A watcher will resume the 64 stable identity shards on the proven
Hermès gateway only after PDR same-family verification releases that capacity;
each candidate still requires three independent `stealth/ox-alpha` judgments,
or 1,536 total. The worker skips only already sealed slot receipts and retries
transport/schema failures without changing candidate bytes. This audit can
qualify or reject the small 434.5 MB conservative math lane, but it cannot admit
the broader 34.1 GB FineMath text population and cannot override benchmark or
deduplication gates.

#### Source-agnostic mechanical quality gate

The answer-key incident is now covered by a reusable gate rather than a
FineMath-only exception. Before semantic admission, the gate detects bare MCQ
keys, scored answer sheets without problem statements, embedded control-byte
corruption, Unicode replacement-character corruption, repeated-character
gibberish, contextless link/markup/structured fragments, and heavily duplicated
boilerplate. Hard-reject, context-review, and cleanup-review routes take
precedence over a mechanical pass. A pass never implies semantic quality,
rights clearance, decontamination, global deduplication, or training readiness.

The code-expanded exact replay covers **10,371 distinct candidate rows** across
13 current populations, with zero candidate-identity overlap and zero exact
source-content overlap between those populations. It routed **10,360** to
mechanical pass, held **9** for
duplicated-boilerplate cleanup, routed **1** short contextless bibliographic form
to context review, and hard-rejected **1** contextless Cambridge physics
mark-scheme row. The catalog-form detector was added after a live Hermès audit
exposed a 258-byte field list that named a valuable book but contained no book
content; replay over all 10,371 identities found exactly that one new nonpass.
That row independently triggered both the scored-answer-sheet detector and 136
embedded backspace controls. An initially broader scoring-marker detector was
rejected during development because it falsely matched citations, array indexes,
PEPs, and papers; those cases are regression fixtures in the final policy.

The current source-safe publication is
[`artifacts/sai_source_mechanical_quality_gate_publication_20260826_r3.json`](artifacts/sai_source_mechanical_quality_gate_publication_20260826_r3.json),
with canonical receipt
`df34d6507032269351df3d841032e068de5ff986dcbcb7d5f92f212e98e82385`
and policy SHA-256
`436ea538156447a7188a15404764302c7b3290b3a06c12677d316f265ccc6c80`.
The decision streams retain identities and measurements but no source text and
are excluded from Git history. The preceding 12-population r2 evidence remains
byte-replayed at immutable Hugging Face commit
[`4ab25974f7b7f40d5ef0bfe2dd8eedfe267831fc`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/4ab25974f7b7f40d5ef0bfe2dd8eedfe267831fc);
the r3 publication adds the exact 2,048-row OpenCoder population and explicit
source-content duplicate accounting. All **28** r3 evidence files plus the
dataset card were downloaded back byte-identically from Hugging Face commit
[`444b1c482ff7e510d68f7e7115f1bf1d2087c936`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/444b1c482ff7e510d68f7e7115f1bf1d2087c936).
The authorized Stokes evidence root contains the same evidence plus the dataset
card as **29** byte-matched files under manifest SHA-256
`929c6b46f7e4de7ca17c5fc360337465d33ec62ba284fbd5ca052c7d61a73c89`.
Every nonpass row is barred from direct admission, and all 10,371 rows remain
`training_ready=false`.

#### Cross-domain connection compiler

Hermès has already proposed cross-domain metadata for **2,105 of 2,243**
completed source judgments: **6,751 assignments**, **730 distinct directed
domain-pair labels**, and **22,345 distinct concept labels**. These are bridge
proposals, not finished training examples.

Sai now converts that metadata into paired-source development work. A frozen
512-pair population uses 1,381 high-confidence English anchors across 290
directed bridge labels, caps each label at eight pairs and each source anchor at
two, and requires different candidate and content identities. The paired
synthesis contract requires exact evidence from both anchors, a conceptual
explanation, worked transfer problem, counterexample, analogy limits,
prerequisite map, and verification questions. Generated rows remain
`training_ready=false` until independent claim verification, benchmark
decontamination, deduplication, and transfer ablations close. The development
proposal receipt is
`0ef933e21252d060a4691cc9f5c63441bd40f4a796d5952db6651298c3c133e5`.
Generation now has a dependency-bound aggregate stage that requires exactly one
receipt per pair and all 64 shard summaries, replays every normalized judgment,
strips literal source quotes from derived candidates, and preserves only their
SHA-256 bindings. Even after generation closes, the aggregate remains
`training_ready=false` pending independent claim and transfer verification.

The first complete generation campaign closed on 2026-08-26 with **512/512**
pair receipts and **64/64** shard summaries. It produced 512 unique derived
candidates spanning all **290** frozen directed bridge labels from 1,381
source-disjoint anchors, using 3,488,388 prompt tokens and 1,169,045 completion
tokens. Independent replay confirmed that no literal source quotes remain in
the derived candidate file. Its candidate SHA-256 is
`d46f7e2c637b71085876cf180fff572095030e2eecf553ff4875aaee13e96bde` and
its canonical aggregate receipt is
`f2eaccffa188fe6ced475f7544006c863f0f9d3979031e35d984a12a6b0566e5`.
The exact aggregate and its 512-row verification population were copied and
byte-replayed under the authorized Stokes evidence root. This is a completed
generation result, not a quality result: the independently requested verifier
is active, and every generated row remains `training_ready=false`.

A second, dependency-staged Hermès pass now verifies all 512 synthesized bridges
against both restored exact anchors. It must cover every generated claim with a
byte-exact quote and separately judge the shared structure, substantive domain
link, worked transfer solution, counterexample, and analogy limits. Retention
requires every check to pass; otherwise the bridge enters an explicit revision
or rejection lane. This verifier is a separate request but uses the same model
family, so its aggregate truthfully records
`independent_model_family_verification_complete=false`, strips the private
anchor text, and still requires decontamination, global deduplication, and
transfer ablation before any bridge can become training-ready.

That same-family verification has now closed with **512/512** exact receipts and
**64/64** shard summaries. It routed **500** candidates to retain, **12** to
revision, and **0** to reject, consuming 4,671,875 model tokens. The canonical
aggregate receipt is
`adfa6897750ad1f883df1ffbb829fc45df0f1d228c789aba3ac013c6dc4a2a13`.
The aggregate's four exact files were checksum-replayed into the authorized
Stokes evidence root at
`grounded-bridge-verification-aggregate/20260826-r1`.
The source-safe generation and same-family verification aggregates were also
published together and force-downloaded byte-identically from Hugging Face
dataset commit
[`d553b1f935e3bc62dc42fbbe56e326eb4c973fce`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/d553b1f935e3bc62dc42fbbe56e326eb4c973fce).
This is meaningful positive quality evidence, but not admission: the aggregate
explicitly keeps `bridge_verification_complete=false` until the independent
Nemotron family, benchmark decontamination, global deduplication, and transfer
ablation close.

An independent-family lane now replays the same 512 immutable source-paired
candidates through `nvidia/nemotron-3-ultra-550b-a55b`. Its receipts bind the
canonical model identity, independent rubric, exact source quotes, every
generated claim, and one verdict per identity. Two transports are fail-closed:
the direct NVIDIA endpoint with the canonical model name, or OpenRouter's exact
`:free` alias only when the response reports provider `Nvidia` and the canonical
Nemotron model. The aggregate records counts by transport and rejects every
other endpoint/model/provider combination. Rate-safe workers operate over
disjoint locked shards. The direct OpenRouter path has its own process-wide
admission ceiling, eliminating the measured HTTP-429 request storm without
stealing capacity from the independent sixteen-request Nous/Hermès gateway
pool; the free
Nemotron transport additionally requests provider-enforced JSON-object output
without changing the bound rubric or validation schema. A deterministic
quote recovery step may replace a
model-rendered citation only with its unique normalization-equivalent literal
span in the assigned anchor; every repair records the raw/recovered hashes and
source offsets, while ambiguous or invented evidence still fails closed. Even
an independent retain is not admission. A
dependency-staged post-generation screen canonicalizes every retained bridge
thesis, shared structure, claim, representation, prerequisite map, analogy
limit, and verification question, then compares all word/code shingles against
the official benchmark-boundary index. Contaminated rows are excluded, clean
rows advance only the decontamination state, and global deduplication plus a
transfer ablation remain mandatory.

The resumable execution entry point is
`run_nemotron_grounded_bridge_verification_local.sh`. It assigns each of the 64
identity shards to locked resumable lanes, permits bounded requests per lane,
uses the canonical pre-existing independent-Nemotron receipt root, skips only
an already sealed shard summary, and refuses to aggregate unless all 512
receipts and all 64 summaries exist. Per-shard atomic locks prevent a resumed
lane from duplicating an already active request. The same run then applies the
pinned official-public benchmark boundary to independently retained rows. Neither
stage can set `training_ready=true`; global deduplication and the prospective
transfer ablation remain open afterward.

The terminal aggregate and benchmark screen are also serialized under one
process-shared file lock. A resumed launcher reopens a pre-existing terminal
output only when it contains a sealed receipt; an empty, partial, or symlinked
output fails closed. This prevents an obsolete or duplicate watchdog from
racing the joint Hermès/Nemotron retention aggregate. A live runtime audit
removed two pre-fix watchdogs before completion while leaving all scientific
requests and sealed judgments untouched; the corrected single finalizer remains
bound to the same-family aggregate and the official benchmark boundary.

NVIDIA's streamed endpoint does not currently supply token-usage fields for
these receipts. The independent aggregate therefore records exact provider
attempt counts and outcome counts, plus the number of receipts missing provider
token telemetry; missing telemetry is explicitly not represented as zero usage.
The first 161 valid independent receipts required 426 provider attempts: 262
responses failed strict model-output validation and three were transient
transport failures. Deterministic, error-specific correction hints now spell
out exact claim coverage, quote provenance, boolean fields, defect enums, and
verdict consistency without changing the rubric or accepting weaker outputs.
The independent aggregate also replays all 512 sealed same-family routes. A
bridge can enter its retained file only when both Hermès and Nemotron return
retain; a Hermès revision or rejection remains a hold even if Nemotron retains
the row. Cross-family aggregation therefore cannot overwrite either family's
more conservative disposition.

The independent-family campaign closed on 2026-08-25 with **512/512** exact
receipts and **64/64** shard summaries. Conservative cross-family routing
retained **460** bridges, sent **47** to revision, and rejected **5**. The
retained and route files replay byte-for-byte, and the canonical independent
aggregate receipt is
`ec7a946cb55a5884c1e360515bf57d464769d6e0218d112429a4501bf0ff9a1d`.
The pinned post-generation benchmark screen then covered all **460/460**
retained identities and found **0 contaminated rows**; its canonical receipt is
`56b650e6e3f11f18a4ee49f0264e86b25f9ba569d0f2fbdebd271daeef0247b8`.
Receipt self-hashes, every referenced file hash and byte count, and all 1,432
aggregate/decontamination JSONL rows were independently replayed. This closes
independent verification and benchmark decontamination for the retained bridge
overlay, while correctly leaving global deduplication and transfer ablation
open and `training_ready=false`.

Those 460 clean bridges have now been compiled into **3,220** exact curriculum
documents totaling **3,641,531 UTF-8 text bytes** across 20 semantic domain
families. The pair-disjoint split assigns **3,052** documents to the prospective
training stream and withholds **168** development documents from pretraining for
the transfer measurement. A 920-anchor foundation query binds every exact and
normalized candidate identity before final-corpus deduplication. The candidate
and query receipts are
`8db1dc1f277b08a2b79ae7cc3d067ba49e3eadeb881ccc065caecc5f26a4fe1c`
and
`cb9d081817df2eba713d27ae63ac1e57b9c49561573e3d2a70c9176265560a60`.
All records are now published under
`training/candidates/cross-domain-connections/20260826-r1` in HF commit
[`2a05f42030c209c5f1c5221629bb44751c782c06`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/2a05f42030c209c5f1c5221629bb44751c782c06).
The temporary quota-safe representation remains replayable as 18 base64-JSON
parts under signed multipart receipt
`7f9bfa8f84fc39f0c7712b85745b5bc04f261e76620ffbca0c8f0c06486bbf00`.
Hugging Face garbage collection has since removed all unreachable mirror
objects, reducing account LFS usage from 8.803 TB to five live objects totaling
1,012,813,803 bytes. The ordinary deterministic gzip is therefore also
published directly at `curriculum_candidates.jsonl.gz` in commit
[`1c8572cd35d888a4a90e24c68e5a1d66f2250f7d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/1c8572cd35d888a4a90e24c68e5a1d66f2250f7d).
It contains 2,265,770 bytes with SHA-256
`c03cbb6b57bb70b4bf8800881c6b7adc79f75aa52e5d1cf55667d8455a12f82e`.
A force-download from receipt commit
[`6aa60d393930677bcd5894b2ffde062fbf465aed`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6aa60d393930677bcd5894b2ffde062fbf465aed)
recovered the exact 12,202,517-byte JSONL with SHA-256
`194c95f4b7fa4f7ff371b61f856e68cdcde5b73a651cc8bfbb28cf55364723b5`;
publication receipt
`5f851099ffadeb92f261b2c088dd26cabf73dcd1b4652f110da35bcbeae8b8ae`
is byte-mirrored to Stokes.
The final loader must include only the train split and cannot declare this
overlay training-ready until foundation deduplication and the held-out transfer
ablation close.

This is a final-composition invariant, not a best-effort suggestion. A signed
Sai training release must name the reconciled cross-domain component and its
receipt alongside the practical prose and educational-code components. It must
exclude the 168 pair-disjoint development documents from pretraining, retain
their identities for the transfer gate, and fail closed if the 3,052
prospective train documents are silently omitted or mixed in before their
foundation-overlap and positive-transfer gates close. The multi-terabyte core
cannot be used to claim that connection data was included; the connection
component has separate byte, row, split, verification, and curriculum custody.
Practical-foundation reconciliation job `821936` is dependency-bound to exact
PleIAs admission job `820649`. It will hash-replay the 382,072-row Books
manifest and every final PleIAs locator, scan only the compact content-hash
columns, promote a provisional development pair to train if either source
anchor is already present in the training foundation, and withhold an exact
generated-text duplicate. Its launch receipt is
`e88d4e0983d9a3869e1134ddc5bbcf5c57488173d21c89ad8c31147595e1aad3`.
This closes exact-content and source-split reconciliation without pretending it
has performed normalized or semantic near-deduplication; positive transfer is
still measured separately.

The positive-transfer gate is a matched three-arm proxy experiment on the
pinned SmolLM2-360M revision. Its unchanged arm, equal-token source-only arm,
and verified-connection arm use identical held-out pair identities, decoding,
optimizer budget, and source-retention measurement. Each arm is an independent
single-H100 request; the connection arm must improve held-out connection NLL by
at least 0.5% against both controls while keeping source-anchor NLL regression
within 1%. A Stokes CPU launcher is dependency-staged after reconciliation and
submits the three jobs to Newton's H100 controller without holding an idle GPU;
their aggregate runs only after all three succeed. A positive proxy screen
authorizes matched multi-seed confirmation, not 4B training or a capability
claim. The final signed corpus must physically include the confirmed train
overlay and must keep its development identities excluded.

Staged launcher job `822003` now holds only a one-CPU Stokes dependency on
reconciliation job `821936`; it requests no GPU while waiting. Its immutable
runtime is commit `dfa62e149f6bafe68b05ba99c3294e23c7c4fa86`, and static Slurm
qualification succeeded independently for the launcher, all one-H100 arms, and
the Newton aggregate. The launch receipt is
`6eacd42e6570133b8f72bcd1d1cc7967bddae9adb16d807da5e32789df4232ec`.

Fresh-seed confirmation is code-ready but deliberately unscheduled until that
screen passes. Seeds `20260827`, `20260828`, and `20260829` each repeat the
unchanged, source-control, and connection arms as nine independent one-H100
jobs. Confirmation requires the treatment to clear the same capability and
retention thresholds on **every** fresh seed; a median win cannot conceal a
failed seed. Only that aggregate may authorize physical connection-component
admission, and it still cannot authorize 4B training. The complete repository
passes 1,357 tests after adding seed identity, cross-seed custody, negative-seed,
and tamper coverage. A successful confirmation automatically feeds a separate
admission job that rewrites only reconciled train rows with completed transfer
custody, emits a deterministic train-only gzip, and physically excludes every
development row. Its dependent publisher uploads that gzip and signed admission
receipt beneath `training/final/cross-domain-connections/20260826-r1` and replays
both remote LFS identities. A negative confirmation therefore produces no
admitted file and no Hugging Face publication.

#### Prerequisite-edge compiler

Sai is also converting Hermès's document-level `prerequisites_assumed` and
`concepts_taught` fields into graph-verification work. It does not equate
co-occurrence with a prerequisite. After both active compiler populations close,
the builder requires each proposed direction to recur in at least two distinct
candidate and content identities, applies the same conservative source-quality
floor, balances selection across domains, and freezes 192 repeated-evidence
edges. A separate Hermès request then compares every edge against two or three
exact source documents and distinguishes strict prerequisites, helpful
foundations, co-taught nonedges, and unsupported directions with byte-exact
quotes. Even positive same-family decisions remain graph candidates until
independent verification and acyclic graph construction close; every route is
`training_ready=false`.

The complete same-family verification closed on 2026-08-26 with 192/192 edge
receipts and 64/64 shard summaries. It classified only **28** proposed
directions as strict prerequisites and **4** as helpful foundations, while
**120** were merely co-taught nonedges and **40** were unsupported. This is an
important negative filter: repeated document-level co-occurrence overpredicted
directional prerequisites for 160/192 proposals, so those edges are excluded
instead of being allowed to distort the spiral curriculum. All output records
retain only source-evidence hashes. The aggregate canonical receipt and file
SHA-256 are
`81546db8ddf82bb85c45ed4dd083f266116fd55b7084c75f4cdc12ad82846e82`
and `dcca78e7b017536781880532280ee5746601f212f5c9f04d08cf9d423feb4c28`.
The exact source-text-free aggregate is byte-replayed under the authorized
Stokes evidence root. The 32 positive candidates still require independent
model-family verification and acyclic graph construction, so none is yet
training-ready.

All Hermès compiler-style workers now acquire the same persistent logical-shard
lock before replaying or creating receipts. This permits dependency-prestaged
single-process fan-out over disjoint bridge, book, prerequisite, representation,
and verification shards without duplicate model calls. The local capacity fan-out
begins only when its exact input receipt exists, preserves already-complete shard
summaries, and retries only unresolved identities when the provider rate-limits a
request.

Provider admission is now bounded across independent processes, not merely
inside each worker. A live replay of the 50 most recently completed receipts at
implementation time contained **62 HTTP-429 retries** in addition to 50 valid
responses, demonstrating that 32 simultaneous client attempts exceeded useful
provider capacity. A subsequent 10-row frontier shard still exhausted all five
retries on four rows at a 16-slot ceiling. The next accepted-capacity search
measured 4.59 completed rows/minute at eight slots and 4.29 at sixteen. A
prospective ten-slot probe then sustained 4.90 completed rows/minute over its
first ten-minute comparison window, while HTTP-429 outcomes per completion fell
from about 0.64 in the eight-slot cohort to 0.55. Subsequent loopback workers
therefore share **ten** OS-locked request slots across every compiler, bridge,
prerequisite, book, and verifier process. The limiter changes only request
timing: candidate identities, prompts, model, temperature, reasoning effort,
and receipt hashes remain governed by the same contracts. Non-loopback
endpoints are unaffected, and new receipts record the applied shared limit
explicitly.

A final bounded twelve-slot probe bracketed the optimum rather than assuming
that ten was maximal. It held at most eleven simultaneous requests, produced
zero valid probe receipts in 3m48s, and coincided with repeated five-attempt 429
exhaustions in the independently running frontier shard. Only the two probe
lanes were terminated; healthy production workers and every completed receipt
were preserved. Twelve was therefore rejected and the measured ten-slot limit
remains active.

Transient HTTP retries are now deterministically staggered by candidate
identity instead of waking every independent worker on the same 1/2/4/8-second
boundaries. The exponential backoff and 30-second ceiling remain intact; only
the transient-HTTP retry timing receives a stable 1.000--2.000 multiplier. This
breaks provider-side retry herds without changing prompts, judgments, candidate
assignments, request hashes, or the content of any accepted training record.
Every new receipt names the retry-timing policy, while completed receipts remain
immutable and replayable.

The PDR representation verifier exposed a separate direct-endpoint failure:
thirteen concurrent Nemotron lanes converted almost all work into HTTP 429s,
while unclosed HTTP-error response bodies accumulated 204 `CLOSE_WAIT` sockets.
Commit `e083611e3e4dff765997bd3b8410ee8da83918aa` explicitly closes streamed
HTTP-error responses and gives direct NVIDIA traffic one process-shared
admission slot. Seven pre-fix worker instances were terminated only after their
sealed receipts were counted; their persistent lane wrappers reloaded the fix
and resumed missing identities under unchanged prompts and shard locks. Socket
residue fell to single digits. The full repository passes **1,342 tests**. This
is a transport-pressure correction, not a change to candidates, rubrics,
models, verdicts, or already-created receipts.

The resumed 1,024-row byte-weighted teacher population has now closed across
FinePDFs, FineWeb-Edu, SmolLM, FineMath, Dolma, and OpenWebMath. Hermès returned
818 `retain`, 163 `review`, and 43 `reject` verdicts, but conservative routing
sent **511/1,024** identities to quarantine, 176 to cleanup review, 116 to
translation review, 89 to factual-grounding review, 23 to rights hold, and only
78 to representation verification. This is a coverage screen, not a source-
wide yield estimate. It blocks bulk admission for all six sources: FinePDFs
alone sent 324/596 sampled rows to quarantine, while OpenWebMath sent 20/35 to
rights hold. Aggregate receipt
`14f09f39cad9e8e7b0c4032deb3b1589d4ecbfc0f7ce4591beca43eb872fb784`
and decision receipt
`6c28bc37575e6b96e0857d942f91a903ca7f93f88e926dbacd875bd986313075`
are byte-matched under `weighted-reservoir-audit/20260826-r1` in the authorized
Stokes evidence root.

All 511 quarantine routes are also sealed in a text-free dataset-exclusion
manifest with 511 unique source-row, candidate-identity, and content hashes.
Every record has `dataset_materialization_allowed=false`; rejected source text
is absent. The manifest SHA-256 is
`4811e01af6474a41322620bebffb781d8934c99a70da9f86f7a8209695a36185`
and its canonical receipt is
`a4f279db7d7609453307bf8f52acc28db5fb3a6fd12ba2e7a8183c3ada313bab`.
It is durable at
`weighted-reservoir-audit-quarantine-exclusions/20260826-r1`. Mixed raw files
remain evidence-only until each salvageable row has clean replacement custody;
the quarantined identities cannot re-enter a later dataset materialization.
The README, aggregate, decision, manifest, and receipt were force-downloaded
and byte-replayed from Hugging Face dataset commit
[`e38aea8688b6e1e5ec6b9cad4f23444d220a19a0`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e38aea8688b6e1e5ec6b9cad4f23444d220a19a0).

The four current exact-replay quarantine manifests are now merged into one
fail-closed materialization registry: 12 UltraData-Math identities, 244
frontier-source identities, 511 weighted-reservoir identities, and an explicit
zero-row Institutional Books exclusion manifest produce **767/767 unique
candidate and content hashes** with no collisions. The empty book manifest is
a sealed audited input, not an omitted source. Every registry row denies
materialization and carries no source text. The registry SHA-256 is
`93d4eb890c61a7eb742b00fbf922c5ce5814a134b6fb0a6e164c70ba0fd4b180`
and its canonical receipt is
`f89d1da4e9b56e4112a19c3f03c5c3ab3297fdfb79540077ad7540980666d45e`.
Both files are byte-matched under
`quarantine-exclusion-registry/20260826-r2` in the Stokes evidence root. Future
materializers must join against this deny registry before emitting a training
candidate; new sealed audit manifests extend it deterministically.

The second resumed population has now closed with exact **1,007/1,007 row and
128/128 shard coverage** over benchmark-clean PubMed full texts. Hermès judged
998 rows conceptually retainable, 6 for review, and 3 rejects, but conservative
routing exposes a severe representation problem: 562 cleanup reviews, 292
quarantines, 145 representation-verification candidates, 5 pedagogical-quality
reviews, and 3 factual-grounding reviews. Risk signals include 794 rows with
OCR/extraction damage, 188 incoherent/corrupted rows, and 107 rows requiring
personal-or-secret-data review. Mean educational value is 3.714/4 and source
reliability 3.958/4, while formatting quality is only 2.695/4. The measured
decision is therefore targeted recovery and verification, never raw bulk
admission: valuable biomedical knowledge is preserved through cleanup, but
damaged representations stay out.

The 292 exact quarantine identities are sealed without text under manifest
SHA-256
`67448fa5e0f45b3bb5eda32ad13c0244461e31aba3539fed197cb71bdf66a84e`
and canonical receipt
`067398606cd244dae3e19e0cb51bf72361da858c346fae4458ffce781a9d1e7e`.
They extend the global materialization deny registry to **1,537 unique rows**;
registry r5 has SHA-256
`493cc806b657723ade2717f5f8f383165ef32c00751742b6232dae34356dcb14`
and canonical receipt
`e449fc43409e08db2adc9ccb0444184161351fe492e7f289841284e77a629236`.
The aggregate, decision, exclusions, and registry were force-downloaded and
byte-replayed from Hugging Face dataset commit
[`df273e0e5b55a896a7440eb4a1f7a700badb5630`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/df273e0e5b55a896a7440eb4a1f7a700badb5630)
and hash-match the authorized Stokes evidence mirror.

A global Hermès teacher census is now dependency-staged behind all 13 exact
source aggregates. It will refuse to run unless the aggregate candidate-file
hashes form a one-to-one match with the 10,371-row mechanical-quality
publication, every source population is complete, and the global verdict,
language, style, and conservative-route partitions each cover every row exactly
once. Reservoir and Institutional Books schemas are normalized without
collapsing their distinct curriculum taxonomies. The resulting publication
contains only counts, usage, source hashes, and aggregate receipts; it cannot
promote teacher opinions into verified admission. This turns the completed
Hermès work into an exact global gap map for scalable triage, representation
verification, translation, and curriculum allocation rather than another pile
of disconnected per-source reports.

#### Host-diverse code-web teacher expansion

The live teacher census exposed a material code shortage: among the first 1,692
high-confidence English anchors, only 16 came from code-repository sources and
only 13 had code style. NVIDIA's newer pretraining-code releases remain
manually gated for the current account, so Sai does not count their metadata as
code content. Instead, an exact public OpenCoder code-web shard is now pinned at
revision `9e8e48e…c06f3`. The complete 286,437,437-byte Parquet member replayed
its published LFS SHA-256 and all 197,882 rows.

The code audit considered 162,487 rows between 512 bytes and 512 KiB, found the
same number of unique content hashes, and froze an 8,192-row host-diverse
screen. The source-agnostic mechanical gate rejected two rows and the official
benchmark boundary rejected 39, leaving 8,151 clean screen candidates. A final
bottom-hash selection froze **2,048 unique documents across 1,922 web hosts**,
with no host contributing more than two rows. Candidate SHA-256 is
`3cf1a97021a22f8a2dbab932c0bbf58ac724bd49b03c679aa61d447126e46182`;
population receipt is
`53abfd09fb2bc71b17dba5b922c1eaa2c7752cb216654e1557b442701937e7c9`.
The dataset card's MIT declaration is bound but does not establish rights for
every underlying web document, so per-row rights provenance remains false.
Hermès compilation is dependency-staged and every row remains
`training_ready=false`. The source-safe receipt and updated dataset card were
downloaded back byte-identically from Hugging Face commit
[`861b793e68504f4a7df6b4e6ade4ce6322454300`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/861b793e68504f4a7df6b4e6ade4ce6322454300).
The same two files are mirrored under the authorized Stokes evidence root;
their SHA-256 values are `e6570108…105affb` and `410f8040…03ba87`.

The code-web population now has a prospective **276-row promotion screen**:
the exact identity buckets `64..71` and `96..103` under the frozen 128-shard
partition. This boundary was fixed after 47 receipts exposed a low initial
yield but before any screen result existed. The full 2,048-row audit is promoted
only if the screen routes at least 30% of rows to representation verification,
quarantines no more than 25%, assigns `computer_science` to at least 60%, and
achieves mean educational-value and technical-depth scores of at least 2.5/4
each. The wrappers pause without canceling an in-flight request as soon as all
16 screen-shard summaries exist. Failure reallocates scarce Hermès capacity to
a better code source; it does not relabel the failed web material or weaken the
thresholds.

`sai-evaluate-opencoder-promotion-screen` now makes that boundary executable.
It pins the population and receipt hashes above, replays all 16 shard summaries
and all 276 compiler receipts, computes every threshold with exact integer
comparisons, and emits a signed, source-text-free pass/fail receipt. The local
dependency runner invokes it as soon as the screen closes, so a failed screen
releases Hermès capacity without a manual scoring gap. The evaluator cannot
weaken a threshold, add a row, call a model, or turn a teacher judgment into
training admission.

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

#### Quality-core shift: approximately 2TB, not an 8TB trophy

The current optimization target is a verified approximately **2TB quality
core**, not preservation of the largest possible raw mirror. Two terabytes is
not a padding floor: if a byte does not contribute reliable knowledge, human
expression, executable procedure, grounded reasoning, or useful curriculum
coverage, volume alone cannot admit it. The earlier 8TB reservoir remains a
historical source-candidate checkpoint and recovery index, not the desired
training corpus.

The first bulk action under this policy removed the current FinePDFs mirror
from `Godlydonuts/Sai`: exactly **1,250 files and 3,082,436,502,565 bytes**.
The weighted audit had routed 324/596 sampled FinePDFs rows to quarantine and
only a small minority to representation verification, so keeping the entire
mirror as a volume anchor was contrary to the measured goal. Every deleted
path, LFS SHA-256, Xet hash, and Git blob identity was sealed before removal;
the exact upstream revision remains `HuggingFaceFW/finepdfs@220bac3acbf07789502c621d2d33952f51ac7f86`,
and repository history also remains recoverable.

The removal plan has canonical receipt
`3c4d686a86c0c11e8eba1e8b32bfc3b39cfb7d496f387a2a9bb68b424b2380cd`.
It was published and byte-replayed at dataset commit
[`d396d5f522518665c506e84411e4ef2d5e7e5682`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/d396d5f522518665c506e84411e4ef2d5e7e5682).
Deletion commit
[`b99ba469952341100fd9a1780944729431b7fbf8`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/b99ba469952341100fd9a1780944729431b7fbf8)
contains zero files under the planned prefix. The verified removal receipt is
`507b38a20b00d4fa4f303bbe33afb204c021171c2bf13bdadad07fd04ca3d552`
and was byte-replayed at dataset commit
[`6eeb4b868151f5b0ea4b9d815efe12083c2a7a9a`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6eeb4b868151f5b0ea4b9d815efe12083c2a7a9a).
The same plan and receipt are hash-matched in the authorized Stokes evidence
root.

After removal, the current Hugging Face `sources/` tree contains 12,834 files
and 5,719,814,783,273 bytes; its 12,724 payload files contain
5,719,811,111,395 bytes. The 1,024-row PleIAs audit has now closed with 417
quarantine routes and only 139 direct representation routes. Its complete
4.489TB metadata census is therefore measuring exact collection-language token
mass so reduction can follow quality, language, rights, and damage strata
rather than arbitrary file order. No bulk PleIAs collection is promoted merely
to reach 2TB.

The exact candidate envelope is now frozen at **2,000,000,000,000 maximum
bytes**. Outside PleIAs, the post-removal lake contains 36 source components,
2,724 payload files, and 1,230,324,458,837 raw candidate bytes. Therefore, if
every one of those nonbulk bytes survived all gates, PleIAs could contribute at
most 769,675,541,163 bytes; this is a provisional ceiling, not an admission.
The envelope's canonical receipt is
`1151bdcdb37e31f5793d9401cfed70f22ba8d23e544342e1b3b693c9ae749cf2`
and its file SHA-256 is
`f7b4f32b7d87953f0848726429e7882f83c17940f560792d27eee850a733df7d`.
It was byte-replayed at dataset commit
[`2c7390534cee3a545d63e1298dd4811dcfbacf0c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/2c7390534cee3a545d63e1298dd4811dcfbacf0c)
and copied byte-identically to the authorized Stokes evidence root. The
envelope explicitly permits a final core smaller than 2TB and grants no
training admission.

The signed conversion ledger now enforces the same boundary in executable
form. Revision r8 reports the historical reservoirs separately as 21.537 TiB
of overlapping, unverified candidate references, sets the final corpus maximum
to exactly **2,000,000,000,000 bytes**, and records that exhausting this
capacity is not required. It still reports zero training-ready bytes because
the active source-wide gates are unfinished; this prevents candidate volume
from being mislabeled as usable data. Its canonical receipt is
`73ceef1e075697c16acfdad40963f04ee1e744b19ef45fa91472fdbf86202b87`
and its file SHA-256 is
`5ef04d2ef2f3bf1d09e75fb965a6978af181baf249e54f72bb2460ee03bfeb49`.
It was force-downloaded and byte-replayed from dataset commit
[`9540fb7eb6ea281fdb19609bf926895fd6916d44`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/9540fb7eb6ea281fdb19609bf926895fd6916d44)
and copied hash-identically to the authorized Stokes evidence root.

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
ledger release r7 receipt
`b638e13a2e7cc74118071f14e29e6dbe0f0a0234f3d150644dc281a2dfe04c47`.
The ledger hash-verifies both reservoir manifests, all six immutable audit
populations containing 2,103 rows, corrected rights-inventory v2, and both exact
bounded text-payload probes. It also binds the first complete source census:
2,504,679 pinned arXiv abstracts with 2,458,156 mechanically eligible rows and
2,380,856,330 mechanically eligible text bytes. Duplicate audit, probe, or
full-census receipts are rejected. Its current funnel is deliberately blunt:
21.5369 TiB referenced candidates, 2,103 acquired audit rows, 17,638,716,209
mechanically useful bytes measured in nine bounded members, two completed
source pilots containing 3,290 bounded near-deduplicated rows, one complete
source census, and **0 training-ready bytes**.
The bounded measurement is not extrapolated to the reservoir. Of the candidate bytes,
7,899,196,133,417 require declared-license
obligation handling, 5,027,859,142,584 require per-row license evidence, and
10,753,021,022,760 require source-terms resolution. Cross-inventory overlap and
full-reservoir text-payload yield remains unresolved, so the candidate-byte sum cannot
be used as a training-data claim. The path-portable text-free r7 receipt has file
SHA-256
`9fb929317de7de7aecdee12b97b91150e99ffe3876ca92c49bd32655a68089bc`
and replayed byte-for-byte from Hugging Face dataset commit
[`288fe22adf52f8b5430cfa6834c039c45004cbbc`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/288fe22adf52f8b5430cfa6834c039c45004cbbc);
earlier ledger releases remain immutable historical evidence.

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

The 148-row Hermès audit completed on 2026-08-26 with exact 148/148 receipt and
128/128 shard-summary custody. It returned 129 `retain` and 19 `review`
verdicts, but verdicts are not admissions: conservative row routing assigned
68 to deterministic cleanup, 31 to source-bound transformation review, 27 to
factual-grounding review, 12 to quarantine, 8 to rights hold, and only 2
directly to representation verification. The source decision is therefore
`targeted_recovery_and_verification`, with bulk admission false. The aggregate
file SHA-256 and canonical receipt are
`9bbad154f8fc5f30c8cc3655b62655cd5d7b88eb1d296d523cd39cab6117e206`
and `336817928a03afdf5efbba0fe4d7838b9337ae17b09dfa0c8f75160c76b605b0`;
the decision receipt is
`e3944ddd66e88d1c1d87327a5fae25f357d0c945eb8d026c1c2d779945825af1`.
Both source-safe files were byte-replayed into the authorized Stokes evidence
root. The 12 quarantined and 8 rights-held rows are barred from dataset-facing
materialization; because this was a coverage screen rather than a source-wide
acceptance estimate, it does not justify deleting whole upstream parents.
The 12 exact quarantine identities are now sealed in a reusable text-free
exclusion manifest with SHA-256
`327c0a9437c521075e59fa24bbe26e8aa627b563652535b6c310965140a42c5f`
and canonical receipt
`4ddfb6632c4a9c6a91797a7d491d468e54e54b8656a5e3af3d8fc55bebadee17`.
Future materializers can therefore delete those rows by identity without
retaining their source text or silently discarding neighboring good rows.

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
8,822,685 excerpt bytes and complete text-free lineage. Compilation is now
complete with exact **567/567 identity and 128/128 shard coverage**. The
conservative work routes are 322 representation verification, 162 cleanup,
61 quarantine, 18 pedagogical-quality review, and 4 factual-grounding review;
no row entered rights hold. Thus 56.79% of the screen is a strong
representation-verification candidate and 10.76% is explicitly barred, while
the remaining rows require named work rather than blanket acceptance. This is
measured row-level routing, not a claim that compressed source bytes have the
same yield.

The 61 quarantine identities are sealed in a source-text-free exclusion
manifest with SHA-256
`3d85c28357ff604302928f42cfe3d50accc31cfdc08024643edc51dbc52248e7`
and canonical receipt
`66f2f73426c516c3609c09cc564c81ba07b9c6f7b4f121f2ee00debe61425668`.
They extend Sai's global materialization deny list from 767 to **828 exact
identities**; the r3 registry has SHA-256
`14e9d8daa480bee1491ad5268e869c7f471a6c6d6e3bff9189474a44d0fb6d60`
and canonical receipt
`94765522fcd09a57ae33af7ec8f03686905cef594ba67014f4a2a1a2ddff48d0`.
The aggregate, decision, source-specific exclusions, and merged registry were
downloaded back and byte-replayed from Hugging Face commits
[`fdd420afbff2a8b72781ef6e4665816d3f703c88`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/fdd420afbff2a8b72781ef6e4665816d3f703c88),
[`42157867e731a8f2bbc67763e16bce5125827257`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/42157867e731a8f2bbc67763e16bce5125827257),
[`ec293907f802e0fc232d170b27e50a11a90792b1`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/ec293907f802e0fc232d170b27e50a11a90792b1),
and
[`f1bb37328d2ea18e47b673e73206c52c1447b7b9`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f1bb37328d2ea18e47b673e73206c52c1447b7b9).
All four evidence sets are hash-matched in the authorized Stokes evidence
root. The census receipt is
`c1f18f641a31672cda7d2b10caf60769df766aa7edea62418d1089645319c92b`,
the compiler-population receipt is
`255e9aa09ec8d2f00c01db05b8eabb6bd06d9f93c77d59b1ed6e8ce2caf7a5ba`,
and the source-safe publication envelope is
`3eeb07c28d575542d87a670396452a155f0c97aaaeaa15f19d526f210568168e`.
Source text, individual decontamination decisions, and machine-local paths
remain unpublished. The 567 candidates remain non-training-ready until
representation verification closes. The three
source-safe receipts were downloaded back and replayed byte-for-byte from
Hugging Face dataset commit
[`756d941130a01fabb042178bf94a67b230a64e4c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/756d941130a01fabb042178bf94a67b230a64e4c).

#### CC0 arXiv temporal screen

Sai next expanded scientific reality anchors using the exact 2,504,679-row
`common-pile/arxiv_abstracts_filtered` snapshot at revision
`dc1ceab4755eb037ec61e49cf1350dab7ceee6e7`. The 1,128,382,223 compressed
source bytes expand to 3,473,188,609 reported in-memory bytes. A 32-row
source-disjoint confirmation had already retained 32/32 abstracts, routed
29/32 to representation verification, found one quarantine row, and observed
zero benchmark overlap. That result justified a broad screen, not bulk source
admission.

The active r2 screen partitions the complete ordered snapshot into **32 equal
temporal strata**, chooses one deterministic SHA-256 window per stratum, and
selects 32 source-disjoint rows from each window. All 32 official
dataset-server responses returned the exact pinned revision. The resulting
**1,024-row** population is disjoint from 36 earlier arXiv audit identities.
The official-public benchmark boundary rejected one row for one eligible code
shingle and zero word shingles, leaving **1,023 rows**. An exhaustive
**522,753-pair** exact/high-confidence near-duplicate replay found zero pairs.

Every sampled row carries the same exact CC0 declaration. The independent
declaration audit recognized `CC0-1.0` on 1,024/1,024 rows with zero rights
holds, attribution obligations, or share-alike obligations, while explicitly
not claiming source-provenance or legal clearance. The source-safe publication
receipt is
`0014f665fbbd09c691d03c8964fd7841bd8932e8ff4947c25e9b0b98eaeecdca`.
Source text and individual contamination decisions remain local. The 1,023
survivors were staged for Hermès after the existing PEP compiler closed;
neither the screen nor the 2.5-million-row parent is training-ready or
authorized for bulk ingestion. All five source-safe evidence files were
downloaded back and replayed byte-for-byte from Hugging Face dataset commit
[`5047dee73c4acbdc0f2f1abf044ff5049d4d59e9`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/5047dee73c4acbdc0f2f1abf044ff5049d4d59e9).

The complete Hermès screen has now closed **1,023/1,023 rows and 128/128
shards**: 1,014 `retain` and nine `review`. Conservative routing sends 851
rows to representation verification, 62 to quality review, 58 to cleanup, 40
to factual-grounding review, 11 to quarantine, and one to source-bound
transformation. Mean source reliability is 3.954/4, technical depth 3.751/4,
information density 3.608/4, and educational value 2.997/4. The screen found
953 rows with proposed cross-domain bridges; 762 rows belong to the
reasoning-depth curriculum phase. This is a strong targeted-verification
result, not source-wide admission: the sample is a temporal coverage screen,
and every routed row remains non-training-ready.

The aggregate and routing receipts are
`a6f63b61144ba7c1b763420887aebfd7d3be3853ffc91e4d79a56e357bcab564`
and `4d6e307535e69eba32f8839f9ea19374c44c83cc3bc7ea10f1f194a68bf81283`.
The exact 11-row text-free exclusion manifest has SHA-256
`3a23123e69dc868bfc3a4d26134316b4e8db84a86796da0abd59bec69b8fc027`.
It extends the global deny registry to **1,548 unique identities** under
canonical receipt
`eca4d137f91e2cdfb3d1ceec808190f61578f2a3467d7787d185176950a04279`.
All six artifacts were force-downloaded and replayed at dataset head
[`102f6a17f2a4217319f55c62c7a16966673ef811`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/102f6a17f2a4217319f55c62c7a16966673ef811)
and copied hash-identically to the authorized Stokes evidence root.

The complete text-free parent census then streamed both exact gzip parents,
verified all 2,504,679 rows and 1,128,382,223 compressed bytes, and removed each
parent before acquiring the next. It replayed all **1,060/1,060** protected
locator and full-text identities across the earlier audit populations. The
embedded provenance is valid and strictly monotonic for every row; parent 0
contains 1,654 upstream source-position gaps, explaining why physical gzip
lines and embedded provenance differ for 1,062,885 rows without constituting
lineage failures.

After excluding the protected audit rows and 45,463 mechanically short rows,
the census measured **2,458,156 unique eligible rows and 2,380,856,330 eligible
UTF-8 text bytes**. This is a complete mechanical ceiling, not a quality,
benchmark-disjoint, near-deduplicated, curriculum-ready, or training-ready
population. The census receipt is
`507561b16269da59bfe5f85ab9ae64e4f9b8b88d815078812d06f843e0cf2708`
and its source-safe publication envelope is
`00cc4bcd19ec550adef2f323e57b81746db7d52c07a6555fdeda16d86bfa52a3`.
The census, publication, and r7 ledger receipts were force-downloaded and
replayed exactly from Hugging Face dataset commit
[`288fe22adf52f8b5430cfa6834c039c45004cbbc`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/288fe22adf52f8b5430cfa6834c039c45004cbbc).

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

The post-audit expansion decision is now implemented in
`sai.data.common_pile_full_source_promotion`. It cannot run on a partial audit:
it replays all 3,290 compiler receipts, all 128 shard summaries, the combined
cross-source survivor population, and both bounded pilot receipts. Promotion
is per source and requires at least 1,024 bounded rows, at least 85% `retain`,
at most 5% `reject`, at most 15% quarantine, at most 2.5% rights hold, and
source-specific educational-value, reliability, and coherence floors. A pass
authorizes only full-source **candidate materialization**. It does not authorize
raw-source admission, training, or the 4B run. This makes the expansion path
automatic at audit closure while keeping bad rows and weak sources out of the
compiled stream.

`sai.data.common_pile_full_source_candidates` consumes only that completed
per-source promotion. It downloads one hash-pinned parent, excludes all prior
audit identities, proves every mechanically eligible row is covered, applies
the corrected official benchmark boundary, rejects high-confidence
contextless answer keys, runs external-memory normalized exact deduplication,
and writes exact attribution custody. The compressed parent is removed after
the scan. Every dropped row is absent from the final candidate file, while the
raw and intermediate files remain available for replay until replacement
custody and later cleanup close. Global near/semantic deduplication,
representation verification, rights clearance, training readiness, and the 4B
run remain false. A no-duplicate watcher is staged to run this path for
Pressbooks immediately if the final source-specific gate passes.

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

The PDR compiler and grounded-representation stages are now active. They join
the clean PDR texts to their exact content and rights lanes, retain only
identities routed to representation
verification, and freezes at most six compiler-requested derivative types per
source. The generation contract requires one exact source
citation for every representation, preserves CC BY-SA attribution and
share-alike obligations, and treats prerequisite edges and cross-domain
connections as unverified candidates. Generated text is emitted separately
from source text, with source citations represented by hashes in the candidate
corpus. It remains nontraining until post-generation benchmark screening,
global deduplication, source-claim verification, and independent representation
verification complete. Representation generation is authorized and running;
no generated representation is admitted to training by this stage, and no
model training is authorized by this pipeline.

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

The first complete representation population exposed one fail-closed boundary
misalignment before any generation request was accepted: a collection caption
contained only 145 UTF-8 bytes, below the worker's 200-byte grounding minimum,
so the worker correctly rejected the full 759-row input. The corrected
`20260826-r2` population now freezes the same 200..262,144-byte source envelope
at population construction, records exclusions separately, and retains 758
fully replayable rows. Exactly one short caption was excluded; no oversized row
was present. The corrected population has receipt
`36c8f6f7dbf5c15800481bb48f85b80a230846e7ad81f87601b40a6cee56acda`
and candidate-file SHA-256
`c083178562d27103085600d140fd979e6aeb0f78d7ed84345b759d92ceec6df6`.
Two lock-protected Hermès generation lanes are active against only this corrected
population, with four requests per lane under the shared ten-request gateway
ceiling. A dependency watcher now waits for all 128 generation shard summaries,
then deterministically aggregates the complete population, screens every
generated representation against the pinned official benchmark boundary,
freezes exact source/generated verification pairs, runs eight disjoint
same-family verification lanes in parallel with eight independent-review lanes
through the Nemotron Ultra model family, and seals conservative cross-model
retain/revise/reject custody. Retention requires both families to retain; any
disagreement routes to revision and either rejection removes generated text.
The watcher cannot skip an incomplete shard, and every stage replays content
and receipt hashes before creating downstream output. Cross-model retention
marks representation fidelity complete but leaves source-claim verification,
global deduplication, curriculum admission, and training readiness false.

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

Because PleIAs is the lake's largest unresolved component, a new 1,024-parent
screen has now completed across all ten `common_corpus_*` partitions. It
selected 102–103 parents per partition by deterministic SHA-256 rank, excluded
all 40 parents from the earlier screen, and fully verified 460,098,704,855
parent bytes without claiming statistical representativeness. Eight identity-
disjoint 128-parent shards each opened exactly one parent at a time, verified
the full pinned byte count and SHA-256 on disk, iterated the selected row group
in 16-record batches, persisted one deterministic usable row, and removed the
parent before continuing. All 1,024 parent identities are unique; the aggregate
candidate and lineage SHA-256 values are
`4f4bbb3c87b0f5e65f7490cf469a04dda03b11e5bc863d0404c6f4fd4f90fd32`
and `566404ea466227d61db98b0cc3664c26c99564b6e70efb4e866caa32c106fe4d`;
the canonical acquisition receipt is
`06cce088ccdc2c58f89d0e467961f4b2a648b5448775f8ed333445af672ed1e3`.

The corrected official-public benchmark boundary then found 33 contaminated
rows and 991 clean rows, with 846 exact word-shingle and 28 eligible code-
shingle overlaps. Contamination spans every partition and is highest in the
sampled partition 8 lane at 7/102 rows. The source-safe screen receipt is
`bc9f207c328c3d8ea8387d0c4692f2fdae216706eb00761e906fc8e3b0f17988`.
The source-safe lineage, acquisition receipt, and screen receipt were
force-downloaded and byte-replayed from Hugging Face dataset commit
[`4741fcd1da4462733d463475c276bd87d4ab7d5d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/4741fcd1da4462733d463475c276bd87d4ab7d5d).
Hermès source-quality judgment has now closed with exact **1,024/1,024
identity and 128/128 shard coverage**. Conservative routing sent 417 rows
(40.72%) to quarantine, 212 to cleanup, 111 to translation review, 87 to
factual-grounding review, 39 to pedagogical-quality review, 18 to rights hold,
1 to transformation review, and only 139 (13.57%) directly to representation
verification. The measured decision is targeted recovery and verification,
not bulk admission. Especially strong negative collection signals include
Chinese-Court-Decisions at 102/103 quarantine, US-PD-Newspapers at 42/58,
Wikidata at 28/42, and VoxPopuli at 35/68. Github Open Source (104/303 direct
representation routes) and StackExchange (20/47) remain candidates for
source-disjoint confirmation rather than being discarded with the bulk.

The 417 exact quarantine identities are now sealed without source text under
manifest SHA-256
`e3c1b1414b7bce33606601b7534d2fbf926ff0520cc6789c29013c99942429a2`
and canonical receipt
`7ef4a411ad623e796f98aa2292d187a828e3ae23b25b201c5917de51e43bbe2c`.
They extend the global materialization deny list to **1,245 unique rows**;
registry r4 has SHA-256
`e2e5749226832ceac2d9bfcd6073cf4825e66cf8b2b439159ea5a13c74e4497f`
and canonical receipt
`bf098027573d3f0979fee8e8da25d212a0fed979295b6a7f815d11a00974afae`.
The aggregate, strata, decision, exclusions, and registry were downloaded back
and byte-replayed from Hugging Face commits
[`899051f6053fd2c687a54432d20721c93ad69314`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/899051f6053fd2c687a54432d20721c93ad69314),
[`86eda20b0ff1782f6e3a5f5a85433a5773065eb0`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/86eda20b0ff1782f6e3a5f5a85433a5773065eb0),
[`b93af40c7c87e2d1b01bb19afdd8f007a94e808f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/b93af40c7c87e2d1b01bb19afdd8f007a94e808f),
[`f7a732bc4505108e03375b24c6d953e698c7e78f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f7a732bc4505108e03375b24c6d953e698c7e78f),
and
[`3c4855b5189441231ad6031bfccfe58c876ff75d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/3c4855b5189441231ad6031bfccfe58c876ff75d).
The same evidence is hash-matched in the authorized Stokes root. Source-wide
yield, clean full-source materialization, and training admission remain false;
the live complete metadata census will attach exact token mass to the measured
collection-language routes before any bulk retention decision.

The live 128-shard census exposed a long-tail infrastructure risk: 57 shards
linearly projected beyond their fixed 16-hour walltime while remaining healthy.
The recovery path now partitions only a failed logical shard into eight exact,
identity-disjoint parent segments. A merge step verifies every segment receipt,
parent hash, file-row hash, axes, totals, nonoverlap, and complete path coverage,
then reconstructs byte-for-byte the same canonical whole-shard `files.jsonl`
and receipt that an uninterrupted run produces. Recovery is CPU-only,
non-requeueing, source-text-free, and is launched only for verified failures;
healthy shards are neither canceled nor duplicated.
Dispatcher job `818603` is staged `afterany` on census array `818243`; aggregate
job `818244` now waits on that dispatcher. The dispatcher exits without child
work when all 128 canonical receipts exist, and otherwise binds `818244` to the
exact per-failure merge jobs it creates. Institutional Books uses its existing
per-parent checkpoints instead: after-any retry array `818602` resumes only
unfinished parents from `818439`, and aggregate `818440` now waits on `818602`.

The Hermès compiler launch was also hardened against provider availability. A
live `stealth/ox-alpha` OpenRouter preflight on 2026-08-24 returned a temporary
upstream shared-pool HTTP 429, not an authentication or billing failure. Both
primary compiler lanes remain pinned to the same model, endpoint, rubric,
candidate identities, and receipt schemas—no fallback model can silently mix
judgments. The PleIAs worker now matches the Books worker's exact-summary skip
and bounded-attempt outer retry behavior. After-any retry arrays `818615`
(Books, 32 lanes) and `818616` (PleIAs, 64 lanes) are staged behind the original
arrays `818512` and `818538`; completed signed summaries return immediately, so
the retry graph creates no duplicate judgments. Aggregates `818513` and `818539`
now wait on the retry arrays. This converts temporary free-provider throttling
into resumable delay rather than an orphaned data graph.

That census now feeds a second fail-closed boundary rather than a bulk copy.
Stokes job `818525` is dependency-staged to join the complete metadata geometry
to the source-disjoint Hermès calibration and produce collection-language work
routes. Those routes are explicitly **not** row admission. After that policy
closes, 128 source-parent-disjoint CPU shards will reopen one pinned Parquet
parent at a time and apply a new row-level gate. Only rows in the strongest
English direct-verification groups, with a complete explicitly reusable rights
label, a stable identifier, coherent size metadata, and a clean mechanical
quality decision can enter the bounded candidate population. Unknown or mixed
rights labels fail closed; non-English material remains available for the
separate high-value translation path; contextless answer keys, corruption,
markup fragments, and duplicated boilerplate are excluded.

Candidate diversity is selected by a deterministic source-row/content hash,
not input order. The first pass samples 20,000 per million mechanically clean
rows and enforces a hard maximum of **1,610,612,736 uncompressed text bytes per
shard**, or **206,158,430,208 bytes across all 128 shards**. This is a storage-
safe discovery population, not a quota to fill. Every output row retains its
exact upstream repository, revision, parent path and SHA-256, row index,
identifier, license, and content SHA-256. Even successful rows remain
`training_ready=false` until semantic verification, official-boundary
decontamination, and global exact/near deduplication finish. The cap therefore
protects both quality and Stokes quota while leaving room for the independently
qualified books and other source families.

The exact row-level array is Stokes job `818530_[0-127%64]`, dependency-bound
to successful policy job `818525`; aggregate job `818531` is in turn bound to
successful completion of every array identity. Both jobs have requeue disabled.
The dependency graph was verified with `scontrol`: no row-level shard can begin
before the census-derived policy exists, and aggregation requires the whole
array. No GPU is requested by this graph.

The next semantic boundary is also implemented but remains dependency-staged.
It streams every hash-verified bounded candidate, balances by collection × open
type × four token-length bands, keeps at most 32 lowest seeded identities per
stratum, and then round-robins strata into at most 8,192 private beginning /
middle / end excerpts. This prevents large collections and easy short records
from monopolizing Hermès attention. The sample retains exact full-text hashes
and locators but never places full source text in Git. Sixty-four identity-
disjoint Hermès shards will apply the frozen polymath compiler rubric; their
aggregate remains nontraining and cannot promote a row. This layer measures
writing quality, knowledge density, reliability, pedagogy, reasoning, cultural
and human-expression value, cross-domain potential, risks, and curriculum
placement before any candidate-family promotion decision.

The exact staged jobs are semantic population `818537`, Hermès compiler array
`818538_[0-63%16]`, and aggregate `818539`. They form a strict chain after the
bounded row aggregate `818531`; `scontrol` verified every dependency. All are
independent CPU requests with requeue disabled and none authorizes training.

To measure same-family optimism instead of trusting it, the primary aggregate
feeds a bounded independent-review population: up to 256 rows from each of the
nonretain, severe-risk retain, cleanup-risk retain, and clean-retain strata
(at most 1,024 rows). An independent Nemotron Ultra family then repeats the
identical frozen rubric, and the source-safe comparison records exact verdict,
route, and active-risk agreement. Disagreement goes to adjudication; it never
defaults to retention. This second family is focused where it changes the
decision, rather than spending calls re-reviewing all 8,192 rows.

That chain is dependency-staged as independent population job `818541`,
Nemotron array `818542_[0-31%8]`, and comparison job `818543`, all after the
primary Hermès aggregate `818539`. `scontrol` verified the exact dependencies;
requeue is disabled and no GPU is requested.

The final semantic-stratum decision is fail-closed. A stratum needs at least
16 primary rows, at least 87.5% primary representation-verification routes,
zero primary quarantine/rights routes, and at least 3.75/4 mean information
density, educational value, source reliability, and coherence. It also needs
at least four complete independent rows, at least 87.5% independent
representation-verification routes, zero independent quarantine/rights routes,
and at least 87.5% exact cross-family route agreement. Passing this gate advances
a stratum only to full-content decontamination and global deduplication—not to
training.

The source-safe semantic-stratum decision is staged as job `818544`, strictly
after cross-family comparison `818543`; its dependency was verified with
`scontrol` and requeue is disabled.

Every advanced stratum is then replayed over the **full retained document**, not
the semantic excerpt, against the pinned official-public benchmark boundary.
Any exact 13-word shingle or eligible code-shingle overlap is excluded. The
same pass removes exact duplicates within each shard and emits a source-safe
content-hash index for the subsequent global exact/near-deduplication pass.
Outputs retain the original provenance and license columns and remain
`training_ready=false`; global deduplication is explicitly unfinished at this
stage.

Full-content decontamination is staged as array `818547_[0-127%64]` after
semantic decision `818544`, with aggregate `818548` after every array identity.
The exact dependencies were verified with `scontrol`; both jobs are CPU-only,
have requeue disabled, and preserve one-output-per-shard custody.

Global exact deduplication is disk-backed rather than memory-bound. A single
SQLite decision streams every source-safe content-hash index and assigns each
full-content SHA-256 to the lowest stable source-row identity. A second 128-way
rewrite retains exactly those identities and seals aggregate accounting. This
removes duplicates across source parents and shards without placing source text
in the decision database. Near-duplicate and cross-source deduplication remain
separate later gates, so even this output is still nontraining.

The exact-dedup chain is staged as SQLite decision `818550`, rewrite array
`818551_[0-127%64]`, and aggregate `818552`, strictly after full-content
decontamination aggregate `818548`. `scontrol` verified the dependencies;
requeue is disabled and the graph is CPU-only.

At exact-dedup closure, a fixed-allowlist custody job copies only source-safe
aggregate receipts, decisions, and the text-free hash database to the authorized
durable evidence root. It explicitly excludes semantic excerpts, model evidence
quotes, compiler receipts, candidate Parquets, and source text. Every copied
file is byte-counted and SHA-256 replayed before a durable mirror receipt is
sealed.

The durable evidence mirror is staged as job `818554` after exact-dedup
aggregate `818552`; `scontrol` verified the dependency and requeue is disabled.

The bounded 2% row pass is a quality-calibration surface, not a volume claim.
Once the conservative semantic decision exists, a separate full-source
production-descriptor census reopens every hash-pinned PleIAs parent and replays
the same direct-English, explicit-rights, mechanical-quality, and advanced-
stratum predicates. It persists **no source text**. Each eligible row contributes
only its stable locator, byte/token counts, full-content SHA-256, NFKC/casefold/
whitespace-normalized SHA-256, a 32-entry bottom-k sketch of unique five-word
shingles, and the conservative floor and mean of the four measured core-quality
scores for its exact semantic stratum. Those descriptors allow global exact,
normalized-exact, near-duplicate, diversity, quality-priority, and byte-budget
decisions to happen before bulk text is materialized.
Benchmark decontamination is deliberately still false at this census boundary
and must replay on the finally selected full documents.

The full descriptor census is dependency-staged as independent CPU array
`818558_[0-127%64]` after semantic decision `818544`; aggregate `818559` requires
successful completion of every array identity. `scontrol` verifies one CPU and
8 GiB per shard, requeue disabled, the exact dependency, and a 24-hour limit.
This stage turns the decimal 2 TB ceiling into a deterministic selection problem:
it neither pads toward 2 TB nor copies bulk text before the global quality and
deduplication decisions are known.

Production descriptor closure feeds a disk-backed normalized-exact decision.
SQLite retains the representative from the stratum with the highest conservative
core-quality floor, then the highest core-quality mean, and only then uses the
lowest stable source-row identity as a tie-breaker for each NFKC/casefold/
whitespace-normalized full-document SHA-256. It separately counts byte-exact
duplicates. The decision database contains locators, sizes, strata, quality
ranks, and hashes, but no source text. It is indexed for the later near-duplicate
and deterministic byte-budget decisions; both those gates and full-document
benchmark screening remain explicitly incomplete. Job `818560` is staged
strictly after aggregate `818559`, requests one CPU with 8 GiB for eight hours,
and has requeue disabled.

A subsequent high-precision document-near-duplicate pass uses four independent
two-fingerprint bands from each 32-value sketch, compares only bounded candidate
buckets, and requires at least 75% sketch overlap plus an 80% document-length
ratio. Connected duplicates collapse under the same quality-floor, quality-mean,
stable-identity priority. The
source-safe output retains only dropped identity → representative mappings; it
explicitly leaves high-fanout buckets and cross-source comparison for the final
global pass rather than claiming recall it does not have. Job `818561` is staged
after normalized-exact job `818560`, CPU-only with 16 GiB, a 24-hour limit, and
requeue disabled.

The post-near candidate set then receives a deterministic, disk-backed byte
selection. If it already fits, every surviving row is kept. Otherwise a first
pass limits any one semantic stratum to 20% of the decimal 2 TB ceiling, then a
quality-ranked refill uses remaining capacity without ever crossing the ceiling
or padding; stable identity is only the final deterministic tie-breaker. The
text-free output contains exact parent/row locators, quality ranks, and content
hashes for later full-document replay. Job `818563` is staged after
near pass `818561`, CPU-only with 8 GiB for eight hours and requeue disabled.
Benchmark decontamination, cross-source deduplication, verified materialization,
and final training admission remain false at this boundary.

Selected identities now feed a storage-bounded full-text materializer. Each
parent-disjoint shard re-downloads only pinned parents containing selected rows,
reconstructs every row/content identity, and screens the complete document
against the pinned official benchmark boundary. Retained rows are uploaded to
`Godlydonuts/Sai` under `candidates/nontraining/pleias/20260826-r1/`; the worker
replays the remote LFS byte count and SHA-256 at its returned commit before
deleting the temporary local Parquet. An eight-worker throttle limits concurrent
Stokes payloads and downloads instead of risking the storage quota. Array
`818564_[0-127%8]` is staged after byte decision `818563`; aggregate `818565`
requires every shard and re-verifies all 128 LFS identities from one repository
snapshot. Both jobs are CPU-only, have requeue disabled, and remain nontraining
until the final cross-source/subdocument dedup and corpus ledger close.

All three large PleIAs writers—the initial materializer, the internal
subdocument rewrite, and the final cross-source rewrite—now construct their
Parquet payloads only inside the Slurm job's node-local `${TMPDIR}`. A payload is
removed by the temporary-workspace boundary after its exact Hugging Face LFS
size and SHA-256 are replayed; only the small signed receipt is then created on
Lustre. Upload failure leaves no durable partial output, so the shard remains
resumable. This means even a 2 TB candidate envelope does not require a second
bulk copy beneath the 1 TB Stokes user quota.

The destination account's public LFS allowance is nevertheless already closed:
the 8.802 TB source lake is present, the authenticated account is not on a paid
plan, and Hugging Face rejected even the small compressed lake manifest while
ordinary metadata commits remained available. Running the original materialize
→ upload → download → rewrite graph would therefore create multiple terabyte-
scale copies and fail at publication. Array `818564` is held before execution;
all upstream census, semantic, descriptor, exact/near-deduplication, and byte-
selection jobs remain active or dependency-staged through `818563`.

Sai now has a storage-virtual alternative that reuses the exact raw PleIAs LFS
objects already present at their pinned source-lake identities. Each of 128
parent-disjoint workers reopens only selected rows, replays content identity and
semantic metadata, applies the complete official benchmark boundary, and emits
sixteen normalized subdocument-signature partitions plus a source-safe retained-
locator Parquet. Locators preserve repository, revision, parent/row identity,
rights, language, quality, difficulty, curriculum, and content hashes but never
source text. The aggregate requires exact selection-row and selected-byte
coverage. Existing external-memory deduplication accepts this virtual coverage
without pretending the documents were materialized. This removes the first
terabyte-scale duplicate upload while preserving deterministic reconstruction;
final cross-source decisions, reconstruction hashes, tokenization, curriculum
packing, and durable training custody still must close before admission.
The quota-safe graph is dependency-staged as virtual-signature array
`818630_[0-127%8]` after byte selection `818563`, exact aggregate `818631`, and
sixteen independent external-sort decision jobs `818632_[0-15%16]`. Every job
uses one CPU, requests no GPU, has requeue disabled, and cannot run before its
exact source receipt exists.

The next quota-safe stage is now implemented, regression-tested, and
dependency-staged. Array `818635_[0-127%8]` waits for all sixteen decisions in
`818632`, reopens each pinned source partition, replays the benchmark-clean
locator/content identity, applies the exact internal deletion records with the
same coherence-restoration rule as the materialized path, and re-signs the
transformed text. It persists only post-transform hashes, character/byte counts,
metadata-bearing reconstruction locators, and normalized signatures; source
text remains transient in node-local scratch. Aggregate job `818636` validates
all 128 receipts and exact retained-document coverage before making those
signatures eligible for the existing cross-source comparison. Both jobs are
CPU-only, requeue-disabled, and remain explicitly nontraining. This eliminates
the second large upload/download cycle without weakening identity, deletion, or
coverage checks.

The quota-safe graph now continues through the complete final join. Sixteen
independent cross-source jobs `818638_[0-15%16]` wait on both the clean-book
signature aggregate `818572` and virtual post-internal PleIAs aggregate
`818636`; aggregate `818639` verifies every book/PleIAs deletion partition.
From that one decision, PleIAs reconstruction array `818641_[0-127%8]` replays
both deletion layers and emits only final reconstruction locators, followed by
aggregate `818642`. Private-book array `818643_[0-63%16]` emits the physically
partitioned final book payload and aggregate `818644`. Ledger `818645` has an
AND dependency on both component aggregates, binds exact surviving UTF-8 bytes
under the 2 TB ceiling, and records PleIAs custody honestly as pinned raw
objects plus verified reconstruction locators rather than pretending a second
payload copy exists.

The earlier static 1.5 TB PleIAs reservation has now been removed because the
measured private-book component was not on track to consume its full 500 GB
reserve. Upstream PleIAs selection may retain as much as the full 2 TB candidate
ceiling so useful rows are not irreversibly discarded before exact final sizes
exist. After both final rewrite aggregates close, a source-text-free byte
balancer computes `2,000,000,000,000 - exact_final_book_bytes - 1,000,000,000`
as the admissible PleIAs ceiling. The final 1 GB remains explicit headroom for
verified connection data and indivisible-document packing rather than hidden
padding. It proportionally assigns that ceiling across all 128 PleIAs locator
shards, selects within each shard by conservative quality floor, quality mean,
and stable identity, and uses a second pass to refill otherwise unused space.
Only held-over locator identities are persisted; no source text is copied.
The exact balance receipt is consumed by the foundation ledger, transient
tokenizer replay, spiral-curriculum index, custody manifest, and global bridge
overlap scan, so no downstream path can silently reintroduce held-over rows.
Allocation, 128 independent selections, and aggregate validation are separate
CPU-only, requeue-disabled dependency stages. This closes both failure modes:
an underfilled corpus caused by a guessed component reservation and an
over-ceiling ledger caused by selecting PleIAs before final book bytes were
known. All stages remain nontraining; final tokenization and curriculum packing
remain open.

The live balance graph is pinned to immutable runtime commit
`047b0bbc8737a2923b74dbce733e8f8da52e0fda`: allocation job `818779` waits on
both final component aggregates, selection array `818780_[0-127%32]` waits on
that allocation, and aggregate `818781` waits on every selection shard.
Pending ledger `818645`, PleIAs tokenizer sample `818720`, and PleIAs curriculum
index `818732` now depend on the balance aggregate. The source-safe launch
receipt is also mirrored under the authorized durable evidence root with file
SHA-256 `7ce7d9216f5b2fe1cf1e744aa8a1c05580101f22afa8a057db28451a7c12de1e`
and evidence-manifest receipt
`fefdf7ca1da36ab24b7e285a181c64d8f7070678fbc4d82471ff733cf5f59596`.
No source text is present in either launch artifact.

Custody job `818649`, strictly after ledger `818645`, then hash-manifests the
13,974-object pinned source lake, final private-book aggregate, virtual PleIAs
aggregate, complete cross-source decision, exact foundation ledger, and the
clean runtime Git commit. It atomically creates byte-identical receipts in both
the working corpus root and authorized durable evidence root. The receipt
explicitly keeps Hugging Face metadata publication, final tokenization,
curriculum packing, final-corpus completion, and training readiness false.

The quota-safe path now also has an executable transient tokenizer interface.
`sai.data.pleias_virtual_transient_stream` validates the signed final aggregate
and its complete ordered shard-receipt set, reopens each pinned parent, replays
the selected-row identity and both deletion layers, rebuilds the intermediate
locator, and requires the resulting final locator to match every sealed field.
Only then does it emit a standard benchmark-disjoint pretraining document plus
the source's semantic phase, difficulty, prerequisites, concepts, domains, and
split as JSONL to stdout. Its only durable output is a source-text-free receipt
binding the ordered JSONL and envelope digests, exact byte/document accounting,
and shard identity. A failed pipe, incomplete replay, duplicate virtual row,
source mutation, decision mutation, or aggregate/shard custody mismatch creates
no successful receipt. This is the bridge needed to feed tokenizer sampling and
final packing without materializing another terabyte-scale PleIAs payload; it
does not itself make the corpus training-ready or authorize 4B training.

A bounded tokenizer-measurement consumer now sits directly on that pipe.
`sai.data.transient_tokenizer_sample` hashes the complete incoming stream,
requires the producer's atomically sealed receipt to match it after EOF, excludes
the source-disjoint development partition, and dynamically rebalances a
quality-first bottom-hash reservoir across curriculum-phase × semantic-domain ×
code/prose strata. Each source shard is capped at 64,000,000 JSONL bytes, so the
complete 128-shard PleIAs tokenizer sample cannot exceed 8.192 GB. The sampler
retains text only in this bounded tokenizer-measurement artifact and preserves
source-text-free input/output custody receipts. Slurm array script
`sample_pleias_transient_tokenizer_stokes.sbatch` requests two CPUs and no GPU,
uses pipe-fail semantics, requeue disabled, sixteen-way concurrency, and an
immutable detached runtime. It is intended to run only after final virtual
aggregate `818642`; it is preparation for a representative tokenizer tournament,
not corpus admission or model training.

The corresponding aggregate replay is implemented in
`sai.data.transient_tokenizer_sample_aggregate`. It reopens all 128 bounded
samples, verifies every shard/source receipt and byte ceiling, parses every
standard pretraining row, and uses a disk-backed identity index to reject both
cross-shard document-identity duplicates and distinct identities carrying exact
duplicate text. Its aggregate contains only counts and hashes, including domain
coverage and the ordered sample/document identities. CPU-only job
`aggregate_transient_tokenizer_samples_stokes.sbatch` is the exact closure gate
for the later 32K/48K/64K tokenizer build and qualification.

Institutional Books now has an equally exact private tokenizer lane rather than
being omitted or assigned a generic license label. The transient book streamer
requires the final 64-shard aggregate, its complete ordered receipt set, the
strict English quality-selection receipt, and the physical train partition. It
binds each surviving book to its exact HathiTrust rights code (`pd`, `pdus`, or
`cc-zero`), quality-agreement record, benchmark-decontamination record,
cross-source transform, curriculum metadata, and final text hash. Only the
source-disjoint training split is emitted. The generic bounded sampler then
retains at most 32,000,000 JSONL bytes per book shard, or **2.048 GB** over all
64 shards, stratified by curriculum votes, semantic domain, and prose mode.
Scripts `sample_institutional_books_tokenizer_stokes.sbatch` and
`aggregate_institutional_books_tokenizer_samples_stokes.sbatch` are CPU-only,
requeue-disabled, immutable-runtime stages intended after final book aggregate
`818644`. The book text remains private and is never uploaded to Hugging Face.

The combined tokenizer tournament is prepared as three independent CPU jobs,
one each for exact lossless 32K, 48K, and 64K byte-level BPE candidates. Every
candidate consumes the identical ordered population: 64 rights-bound private
book samples followed by 128 benchmark-clean PleIAs samples. Independent builds
allow the scheduler to admit available CPU capacity without serializing three
multi-hour vocabulary constructions. A single downstream qualification reopens
all 192 sample files, evaluates every candidate on identical bytes and the full
protected English/code/math/science/technical string suite, and requires zero
round-trip, unknown-token, or empty-encoding failures. Final custody then binds
both sample aggregates, all source file hashes, all three build manifests and
tokenizer trees, the matched qualification report, and the mechanical 48K
selection receipt. Mechanical selection is explicitly not represented as a
capability winner: production selection and capability comparison remain open.

Materialization also preserves the exact semantic stratum plus its conservative
quality floor and mean from the selection database. Both later rewrite schemas
carry those fields unchanged, so curriculum and mixture construction do not
have to infer quality from a collection name after text transformations. The
stratum decision now also aggregates Hermès difficulty, prerequisite burden,
curriculum phase, domain votes, and recurring concepts/prerequisites; the
materializer binds that decision receipt and carries those source-text-free
signals into every final row. Final receipts report exact bytes by phase,
difficulty, domain, and quality floor.

To avoid pulling a second 2 TB copy onto Stokes for subdocument deduplication,
the remotely verified candidate shards are reopened one at a time and segmented
losslessly at natural prose/code boundaries. Only component/shard/document/chunk
locators, character spans, normalized SHA-256 identities, lengths, and code flags
are retained locally; no source text enters the signature index. This makes a
global frequency/length-aware boilerplate decision possible with a comparatively
small external index, followed by targeted remote-shard rewrites. Signature array
`818566_[0-127%32]` is staged after materialization aggregate `818565`; aggregate
`818567` requires all identities. Both are CPU-only, requeue-disabled, and still
mark global subdocument and cross-source deduplication incomplete.

The signature files are hash-partitioned into sixteen normalized-SHA-256
buckets so the global decision does not become one serial bottleneck. Array
`818568_[0-15%16]`, dependency-bound to signature aggregate `818567`, independently
external-sorts one complete hash bucket across all 128 source shards. It applies
the frozen frequency/length-aware retention budget, preserves all matching
occurrences in a boundary document for coherence, and emits 128 source-shard
deletion maps containing only identities, chunk spans, hashes, frequencies, and
budgets. Scratch runs use job-specific Lustre directories and are removed at
closure. This completes the parallel PleIAs deletion decision but deliberately
does not claim the remote rewrite or cross-source decision has closed.

Once all sixteen hash buckets close, rewrite array `818569_[0-127%8]` builds one
disk-backed deletion index per remote source shard, replays every decided chunk's
document identity, span, and normalized SHA-256, and removes duplicate spans only
when at least 100 characters can be deleted without emptying the document. Short
or destructive edits are restored for coherence. Final candidate shards are
uploaded under `final/nontraining/pleias/20260826-r1/`, remote LFS size/SHA-256
verified, and temporary local text removed. Because transformations invalidate
source token counts, the output preserves those counts only as lineage, recomputes
word counts, and requires final retokenization. Aggregate `818570` verifies all
128 rewritten files from one repository snapshot. Cross-source deduplication and
training admission remain false after this PleIAs-only rewrite.

The rewrite is re-signed before any cross-source claim: array
`818574_[0-127%32]`, staged after aggregate `818570`, reopens each verified final
remote shard and rebuilds the same sixteen text-free signature partitions from
the post-deletion content SHA-256. This prevents chunks already removed by the
PleIAs-only pass from inflating later global frequencies. Aggregate `818575`
replays all 128 rewritten receipt identities, final text-byte totals, signature
partitions, and document counts. It is the exact final PleIAs input to the
cross-source book/PleIAs comparison; both jobs are CPU-only, requeue-disabled,
and do not claim training admission.

The two independently sealed component aggregates converge at array
`818577_[0-15%16]`, with an AND dependency on book aggregate `818572` and final
PleIAs aggregate `818575`. Each CPU-only job external-sorts one complete
normalized-hash bucket across 64 book shards and 128 PleIAs shards, applies the
same frequency/length retention rule, and partitions deletion decisions back to
all 192 exact component/shard locators through a bounded file-handle pool.
Benchmark-disjoint Institutional Books have representative priority over copied
PleIAs passages; identity order is only the later tie-breaker. Aggregate
`818578` verifies all sixteen bucket receipts, every deletion file byte/SHA-256,
all component/shard partitions, and exact deletion accounting. No receipt stores
source text, and rewrite/training admission remain false until both component
rewrites and the final corpus ledger replay these decisions.

The exact global decisions now feed two separately controlled final writers.
Private Institutional Books array `818583_[0-63%16]` reopens only the
benchmark-disjoint book identities, replays source/content/chunk hashes, and
emits exact post-deletion private Parquet shards; aggregate `818584` requires
all clean books and all globally assigned book deletions. It explicitly records
`huggingface_redistribution_authorized=false`, so no book text is uploaded.
PleIAs array `818585_[0-127%8]` applies the same verified replay to the final
internally deduplicated remote shards, uploads only under
`final/nontraining/pleias-cross-source/20260826-r1/`, verifies every LFS byte and
SHA-256, and removes temporary local text; aggregate `818586` verifies all 128
identities and global deletion accounting. All four jobs are independent
single-CPU, requeue-disabled requests, strictly dependency-bound to `818578`.
Both outputs still require exact final retokenization and corpus-ledger
admission; neither authorizes training.

Single-CPU ledger job `818588` has an AND dependency on both component
aggregates. It sums exact post-rewrite UTF-8 text bytes, rejects totals above
2,000,000,000,000, records unused headroom rather than filling it, and binds
private versus redistributable custody separately. The ledger deliberately
keeps final tokenization, curriculum scheduling, synthetic-bridge admission,
final-corpus completion, and training authorization false.

PleIAs production selection now uses the full 2,000,000,000,000-byte candidate
ceiling, followed by the exact post-rewrite byte balancer described above. This
does not permit PleIAs to crowd out Institutional Books: the final balancer
subtracts exact surviving book bytes before admitting PleIAs. It simply avoids
discarding high-quality candidates based on a guessed reservation. When the
PleIAs quality core exceeds its final allowance,
the first pass gives every surviving semantic stratum an equal-byte opportunity
(also bounded by the 20% single-stratum cap); only then does a deterministic
quality-ranked pass refill unused capacity. The ceiling is still not a target.

The component rewrites construct the source-disjoint split directly instead of
leaving it in a lossy sidecar or mixed final file. A fixed SHA-256 policy assigns
5% of group buckets to development: every row from one pinned PleIAs source
parent stays together, and every Institutional Books row in the same globally
connected work-ID candidate family stays together, including transitive overlaps
such as `A-B` plus `B-C`. Writer schema v2 routes rows at creation time into
physically distinct `train` and `development` Parquets. PleIAs uses separate
remote prefixes; private Books uses separate files in each shard directory.
No second corpus copy is created, and an empty shard-side split is represented
explicitly rather than by an empty or ambiguous file. Both aggregate receipts
require the physical partition flag and exact train+development document/byte
accounting. The foundation ledger independently requires both components to
carry the same canonical split-policy SHA-256, requires both global lanes to be
nonempty, and records their exact post-rewrite byte totals. This split is for
held-out data/model selection and remains separate from all official public
benchmark boundaries.

The same aggregate boundary now fails closed on metadata coverage. Every final
PleIAs row must contribute exactly one semantic stratum, quality floor,
difficulty, and curriculum phase plus at least one domain; every final book must
carry independently agreed curriculum metadata, exactly one genre, at least one
domain, and at least one curriculum-band vote. The 2 TB ledger accepts neither
component unless both semantic-quality and curriculum coverage are complete.
The ledger also preserves the exact per-component and combined domain, genre,
semantic-stratum, quality-floor, difficulty, curriculum-phase, and
curriculum-band counters. This makes breadth and difficulty imbalance visible
before tokenization or scheduling instead of hiding them behind one byte total.

For Institutional Books, that accounting now extends beyond broad domain labels.
Every final row must replay a structurally complete two-model consensus record:
quality floors, linguistic/conceptual/reasoning complexity ranges, curriculum
votes, styles, translation disposition, work/edition identities, and all
source-text-free shared lists must be well formed and hash-consistent. The final
component and foundation receipts expose agreed culture/geography, historical
period, subdomain, style, recommended representation, translation type, quality
floor, and complexity-range counters. Culture and period may legitimately be
empty for a culture-neutral work such as a general mathematics reference, but
available non-Western evidence is no longer discarded from the corpus ledger.
This makes “English-only is not Western-only” an auditable coverage property.

A separate source-text-free real-row diagnostic now checks the same mechanical
policy against private materialized book text rather than relying only on unit
fixtures. It selects deterministic rows from every book shard complete at the
moment of execution, verifies shard/parent/file/content hashes, runs the full
mechanical gate on the complete text, and persists only measurements, flags,
decisions, hashed barcodes, and exact source pointers. It is explicitly not an
acceptance-rate estimate and cannot change admission. The bounded CPU job uses
one thread and no GPU; any unexpected real-text flag becomes inspectable
evidence for a precision correction before the full book gate runs.

The gate's implementation was also tightened for full-corpus scale without
changing the frozen tests' decisions: alpha-word and answer-key context counts
stream through iterators instead of allocating complete word lists, and
navigation/error marker scans are bounded to short pages (at most 16,384 UTF-8
bytes). Long technical or historical books therefore avoid thirteen irrelevant
full-text substring scans per row while preserving the stricter noise policy.

The current-policy replay then closed on **52/52 passing rows across 26 complete
book shards**, with no hard-reject, context-review, or cleanup flags. Its policy
SHA-256 exactly matches the optimized gate:
`3112551ef1b7578a69f0ae316c32f5adea38f80ff6ce055d183f80c66d322def`.
The canonical receipt is
`684228cf150ae87bfe04059612050f61cfff629cfc869be9cc79c11d85e1e24e`;
the receipt file SHA-256 is
`01415319f5069776af897040463c2ab25656273cc196df6e3700f5463c602327`.
The source-safe artifact was force-downloaded and byte-replayed from Hugging Face
dataset commit
[`fe78514246a40f8113c5e55c3f14ba80ee991b49`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/fe78514246a40f8113c5e55c3f14ba80ee991b49)
and remains byte-identical in the authorized Stokes evidence root. The earlier
r1 receipt remains immutable historical evidence but is not cited as proof of
the current policy because its policy hash predates the streaming optimization.
This bounded result supports precision on real long-form inputs; it is not
evidence that all books pass or that any book is training-ready.

Two source-disjoint collection confirmations now sharpen that bulk pause with
**80 additional rows across ten named collections**. Every collection has eight
primary Hermès judgments and eight independent full-coverage Gemini 3.5 Flash
Lite judgments; partial Gemini 3.1, Groq GPT-OSS, and Nemotron reviews remain
auxiliary evidence only. The conservative, text-free decision advances Github
Open Source, StackExchange, and USPTO to targeted verification; holds
Chinese-Court-Decisions, US-PD-Newspapers, and Wikidata as high-blocking;
routes Wikipedia and VoxPopuli to translation/grounding adjudication; and keeps
Creative Commons Common Crawl plus dotgov in targeted recovery. No collection
is automatically admitted or excluded. The fail-closed r2 decision additionally
requires exactly eight rows per collection and rejects duplicate comparison
receipts. Its canonical receipt is
`126e956caf96af31a5c7689126f90592b1652cb5f5b61806a2f524aec7d3b0a5`;
all seven source-safe receipts/comparisons were force-downloaded and byte-
replayed from Hugging Face dataset commit
[`dd85ba5628df827e56f4cc4cd2b92469b884d64f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/dd85ba5628df827e56f4cc4cd2b92469b884d64f)
and the stricter decision was separately replayed from commit
[`4fa469cd95556e39f951dc79f174592132407b96`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/4fa469cd95556e39f951dc79f174592132407b96).
All files hash-match the authorized Stokes evidence mirror at
`pleias-collection-confirmation/20260826-r1`.

The larger 512-row frontier-source compiler has now closed with exact identity
and receipt coverage. Hermès returned 348 `retain`, 125 `review`, and 39
`reject` verdicts, but conservative routing sent only **25/512 (4.88%)** to
representation verification and **244/512 (47.66%)** to quarantine. FineWeb2-HQ
and both Ultra-FineWeb snapshots are therefore bulk-paused; Nemotron
specialized reasoning and UltraData-Math L1 remain targeted-recovery and
verification sources. The aggregate replayed 512 valid outcomes, 210 transient
HTTP errors, and 177 repaired or retried invalid model outputs. Aggregate
receipt `9c2d0e49d062c1886f9809b8852c4c48c38f41b40bea210631d1cc0f7236c6de`
and decision receipt
`7656214825b6f66984c007f00a6f089c5d2c43791af6b775c466db56d952a48d`
were remotely byte-replayed in Hugging Face commit
[`ced4fa4db0a90b8804aa0b42ba98e01597920433`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/ced4fa4db0a90b8804aa0b42ba98e01597920433).
The same three source-safe files are byte-matched under the authorized Stokes
evidence root at `frontier-source-audit/20260825-r1`.
All **244** quarantined identities are now additionally sealed as a text-free
dataset-exclusion manifest. Its SHA-256 is
`9ccb9c64c125d907d8bdfa46d01dcc2dbf5c6e1cffc6adeee3fa6995400615ab`
and its canonical receipt is
`50803c3adb2b2dc758344542106735b9a0b2e9c403ef629e3d9ed872191ff256`.
Those rows can no longer re-enter Sai through a later bulk materializer; the
manifest deletes identities from admission without treating a mixed source
parent as uniformly bad.
These measured routes are neither whole-source yield estimates nor admission;
they prevent dataset branding from silently substituting for content quality.

The frozen OpenCoder code-web promotion screen has also closed with exact
276/276 identity coverage across 16 preselected logical shards. It passed only
computer-science coverage: **130/276 (47.10%)** rows route to quarantine, only
**20/276 (7.25%)** route to representation verification, educational value is
**2.195/4**, and technical depth is **1.659/4**. The exact replay therefore
records `stop_full_audit_and_reallocate_hermes_capacity`, not a full-audit
promotion. All post-screen OpenCoder workers were stopped. Its
286,437,437-byte local Hugging Face acquisition-cache blob and snapshot symlink
were deleted, reclaiming the full physical blob while retaining the population,
compiler receipts, shard summaries, and source-safe decision. The raw object
was never uploaded to `Godlydonuts/Sai` and remains recoverable from pinned
upstream revision `9e8e48e666c226294d6f9e6c2e13f2c84c1c06f3`; this bounded screen does
not claim every upstream row is unusable. Screen receipt
`29a7ceed9841f99213d4087a40e0107277a07793b490760ee800242bcad7be70`
and cache-reclamation receipt
`a45fa68a77018c1b58900b58aa703ad295c249183528ec8495816fe95c6ac172`
were remotely byte-replayed in Hugging Face commit
[`f6151be578e7e353af45152426a55681a27eae80`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f6151be578e7e353af45152426a55681a27eae80)
and copied byte-identically to the authorized Stokes evidence root at
`opencoder-promotion-screen/20260826-r1`.

The same cache policy removed a second, unrelated object without waiting for a
semantic source audit: a partial `willdepueoai/parameter-golf` snapshot holding
ten FineWeb training binaries and one validation binary already encoded by an
unrelated 1,024-token benchmark tokenizer. The **2,124,321,260-byte** cache was
not original text, appeared nowhere in Sai code or live processes, contained
only 11 of the upstream manifest's 196 binaries, and had zero paths in
`Godlydonuts/Sai`. Every local file hash was verified before the exact cache
root was quarantined and deleted. It is not locally recoverable, but is exactly
re-downloadable from upstream revision
`a85b0e6035c3c94bc23685a07c81a8f3bf89db80`. Reclamation receipt
`612423e30092478571c9a43eae23d0271d8278eaa816e92f90a3d605ae1a91fe`
was remotely byte-replayed in Hugging Face commit
[`40d688182f3e6a65b3b09a96eeb33ee9e48a441e`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/40d688182f3e6a65b3b09a96eeb33ee9e48a441e)
and mirrored byte-identically to Stokes under
`parameter-golf-cache-reclamation/20260826-r1`.

An unused `ProsusAI/finbert` cache was also removed after proving zero Sai-code,
live-process, and open-handle references. It contained two redundant formats of
a narrow financial-sentiment classifier, not a Sai data, tokenizer, semantic
deduplication, or training dependency. All 13 files and six symlinks were
hash-manifested before **876,191,371 bytes** were deleted. The cache is not
locally recoverable; main revision
`4556d13015211d73dccd3fdd39d39232506f3e43` and safetensors revision
`7db323f79b751944bcfa66298ec06977e4518306` remain pinned upstream. Receipt
`bc82ae86511c507edf8128f8e5ced567e58b2e2d300128258da2b02bab7b2117`
was remotely replayed in Hugging Face commit
[`985b5bae55c694825d3ba4cfaffa01774d04287c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/985b5bae55c694825d3ba4cfaffa01774d04287c)
and mirrored to Stokes under `finbert-cache-reclamation/20260826-r1`.
The similarly sized `all-mpnet-base-v2` cache is deliberately retained because
it remains directly useful for semantic deduplication.

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

The first complete-parent execution of this layer is now closed over the
54,509 Pressbooks and Public Domain Review candidates. It indexed 4,650,337
chunks, found 494,414 duplicate groups and 943,565 duplicate occurrences,
removed 90,691 chunks / 11,216,449 characters, modified 9,573 documents, and
fully removed two duplicate-only documents. The adaptive output contains
54,507 documents under canonical receipt
`9d3ee20c4d5d0732589c3baea55414752e0184bb4d81a06a3b71f6894bff1e8e`;
the text-free 19,949-record transformation manifest has SHA-256
`fed1c0767589b5b596adf8008c0d8a273c58e502421424cacd00ad9d5f75ff5f`.
All temporary indexes were removed. This is a deterministic transformed
candidate, not evidence that the adaptive policy improves a model.

An exact post-transform deletion join then removed all 11 Pressbooks source
rows that the sealed 3,290-row Hermès compiler pass had hard-rejected. Those
judgments include ten weak-grounding flags, eight duplicated-boilerplate flags,
and five personal-or-secret-data flags. The join also removed attribution for
the two upstream fully deduplicated documents, leaving exactly 54,496 candidate
and attribution rows with identical source-row sets. Candidate, attribution,
and text-free exclusion-manifest SHA-256 values are
`02bd40ea6a7b9a5710861e4284a09981101512f341e91e549c75effc0a76faa8`,
`9cb7af6399bc527865dee98fc623691d827222155fecc011946b54cfa7f8011a`,
and `d07030784790a8799d039c275428b1ca25947e62eafeb2f7c1de12cffa99a474`;
the canonical exclusion receipt is
`89756c4dbd45b772889bbd813138583fcc9a73850305a2202baefcdbef18df43`.
All four outputs were byte-verified in durable Stokes custody before the local
unfiltered candidate was permanently unlinked. The four small source-safe
receipts/ledgers were force-downloaded and byte-replayed from Hugging Face
dataset commit
[`10c6b42c61cf9eac463014416e659109d7e639f4`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/10c6b42c61cf9eac463014416e659109d7e639f4).
The clean 534,981,699-byte candidate and 43,758,121-byte attribution files could
not be added to that commit because Hugging Face returned an explicit public
storage-quota 403 before creating any large-file commit. Their exact local and
Stokes custody remains valid; this is a publication-capacity blocker, not a
missing-data condition.

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

The complete metadata census shows why this metadata-first policy matters:
only 382,961 deduplicated, rights-bounded English works occupy the strict
OCR≥95 tier, totaling 78.690B tokens. Another 91,272 non-English OCR≥95 works
total 23.628B tokens and remain translation candidates whose treatment depends
on genre. These exact tiers, rather than the 242B-token raw total, now define
the selective materialization frontier.

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

The first three completed book receipts exposed a throughput issue: valid
outputs required two to five provider calls because generic retry text did not
spell out the book schema's most common failures. Future shard processes now
receive deterministic book-specific correction hints for exact excerpt
quotes, concept-edge evidence, English translation disposition, enum fields,
risk keys, domains, and representation labels. The strict schema is unchanged;
the correction only reduces wasted invalid retries. Existing healthy book
workers are not interrupted and automatically pick up the new code when their
next immutable shard process starts.

The complete bounded book pilot has now closed: **185/185** candidate receipts
cover 182 nonempty immutable hash shards, with 182 `retain`, three `review`, and
zero `reject` model verdicts. The compiler identified 5,121 unique concept
labels and 1,129 unique explicit prerequisite-edge claims. The curriculum
distribution is four basic, 61 intermediate, 108 advanced, and 12 expert rows;
the source-language distribution is 173 English, eight Czech, three French,
and one German. The model requested 2,123,954 prompt tokens and emitted 207,638
completion tokens. These are compiler measurements, not admissions: OCR,
historical-context, factual-grounding, deduplication, translation, and
representation-verification work remains separately routed, and the aggregate
retains `training_ready=false`.

The aggregate receipt is
`31a20de0a616b61ac1c5f5fbc22c36fdecb0575237b346f3a4b1252909315d78`
and its file SHA-256 is
`7fbd1ebb7ab85f6dc48abaa15a1098f1e0b4b793c399709b974dbff021596336`.
No row reached the conservative quarantine route, so the source's exclusion
manifest is intentionally empty and hash-sealed, with receipt
`57aead38995182a329188aedf5318720afa806e63a1dea30ef4a763059f3eccb`.
The aggregate, empty exclusion evidence, and four-source registry are
byte-matched in the authorized Stokes evidence root and were force-downloaded
and byte-replayed from Hugging Face dataset commit
[`0e26ff13821ae4cca9c64c380aecefddaa265c98`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/0e26ff13821ae4cca9c64c380aecefddaa265c98).

#### Independent frontier-model review capacity

Sai now has a separate, fail-closed review worker for exact provider/model
pairs. Its receipts bind the candidate, rubric, request, response model,
endpoint, token accounting, and repair attempts; they cannot be silently
merged into the primary Hermès stream. Live qualification confirmed Google
Gemini 3.1/3.5 Flash Lite and Gemma 4 26B-A4B/31B, Groq GPT-OSS 120B and Qwen
3.6-27B, and Cohere Command A Plus/Reasoning. NVIDIA's independent grounded
bridge verifier is separately pinned to
`nvidia/nemotron-3-ultra-550b-a55b` and cannot mark a bridge or training row
ready by itself.

The first matched one-row check produced schema-valid judgments from Hermès,
Gemini 3.1 Flash Lite, and Groq GPT-OSS 120B. All three independently selected
`retain`, `grounding`, and `biology_medicine`; Hermès and Gemini also agreed on
the `duplicated_boilerplate` risk, while GPT-OSS did not. This verifies
cross-family execution and exposes a real disagreement; one row is not an
accuracy estimate. Provider qualification also failed closed where appropriate:
Cerebras returned a billing gate, Groq Qwen repeatedly returned 429, Google
Gemma and both Cohere models failed the strict JSON contract in their first
pilot, and none of those lanes was scaled. Gemini 3.5 completed eight of nine
rows before one strict enum failure. Deterministic enum-specific repair hints
were added without weakening the schema, and the full repository suite passes
1,078 tests.

A larger calibration screen now freezes **45 rows** across PleIAs (16), PEP
(14), and PubMed (15), stratified into clean retain, cleanup-risk retain,
severe-risk retain, and non-retain cells. Gemini 3.1 and 3.5 covered all 45
identities and agreed with each other on 45/45 verdicts and 37/45 conservative
routes. Each agreed with the primary Hermès verdict on 36/45 rows, but only
17/45 and 14/45 primary routes. Most importantly, Hermès marked nine rows
non-retain while both Gemini models retained all nine. Across Hermès and both
Gemini lanes, only 12/45 routes were unanimous; 11 of those 12 came from the
clean-retain controls.

Nemotron Ultra then targeted the 12 clean controls and nine disputed
non-retains. It returned 18/21 valid signed judgments; three remained explicit
non-JSON endpoint failures after all retries. Nemotron retained all 11 covered
clean controls and all seven covered non-retains, but routed ten of those 18
rows to cleanup, grounding, translation, or rights work rather than direct
representation verification. All four available judges agreed on eight clean
routes and zero non-retain routes. This is actionable calibration evidence:
single-model risk labels are useful for triage, but cannot justify irreversible
deletion or automatic admission. Future row deletion requires deterministic
evidence or adjudicated agreement; collection-level pruning remains eligible
when the complete source audit demonstrates persistently poor yield.

The Gemini consensus receipt is
`99a5e35c7d8f42ef1a5961670c99df22cd2ccf3e67bcb53cbb15b4615bbb52cd`.
The Nemotron target-coverage receipt is
`3b3e5c38407f5d83586f5f0c6c95725987439736842d68dc80777f42403180a5`,
and the combined Gemini/Nemotron comparison receipt is
`8085928cdfa68daf5c439d57912ce2a11d429d1421147e33412591ce5103a5c2`.
All source-safe evidence was byte-replayed at dataset commit
[`bee9b4d2d619e8bc8edcfd5a97c79fcc0c4ba5f3`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/bee9b4d2d619e8bc8edcfd5a97c79fcc0c4ba5f3)
and copied byte-identically to the authorized Stokes evidence root. Sampled
source text was not published.

The previously reported contextless physics answer sheet is also bound to an
exact source location: FineMath-3plus row 50 of upstream
`train-00076-of-00128.parquet`, content SHA-256
`5f3c24824cb8c10cfeb19ff6242966541ed2a274fd0bdd6f88418b296d718482`.
Its hard-reject decision remains active. That mixed 503,424,134-byte upstream
shard is not present in `Godlydonuts/Sai`, so there is no bad published file to
delete; deleting the whole mixed shard would also discard unrelated good rows.

The same row-level gate now catches three additional high-confidence web-noise
families before semantic admission: repeated `lorem ipsum` placeholder text is
hard-rejected; short navigation/cookie/account shell pages with at least four
distinct shell markers are held for context review; and short access-denied,
anti-bot, JavaScript-required, or error placeholders with multiple independent
markers are also held. The rules bind exact measurements and a canonical policy
hash. Deliberately adversarial preservation tests keep real web-security prose,
design-history discussion that mentions `lorem ipsum`, code, mathematics,
tables, worked questions, and contextual technical writing eligible. This is a
precision filter, not a license to delete every page containing one UI phrase.

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

The complete bounded compiler gate has now closed across all 3,290 rows and
128 immutable shards. Hermès returned 3,163 `retain`, 116 `review`, and 11
`reject` verdicts, while deterministic routing held 240 rows in quarantine and
21 on rights review. Pressbooks retained 1,829/1,948 rows with 11 rejects;
Public Domain Review retained 1,334/1,342 with zero rejects. Both sources pass
the frozen threshold for **full-source candidate materialization only**. This
is not verification or training admission: global near/semantic deduplication,
rights resolution, representation verification, and final curriculum custody
remain open. The aggregate and source decision have canonical receipts
`ca9884af6ec7e5ef8f2b39a7fcbbe8892423be0e53b75006fcbf987f7ae76484`
and `99f96fcf15a4bdd69fdf451c1220d544c7d008c7682abdfc303450b4797075b2`.
All three source-safe release files were force-downloaded and replayed
byte-for-byte from Hugging Face dataset commit
[`7a447380b42cf631581a1604b249accecbb153bc`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/7a447380b42cf631581a1604b249accecbb153bc).

The resulting complete-parent Pressbooks pass scanned all 54,455 rows. It
excluded 36 prior-audit rows, 35 short rows, 130 oversized rows, and 1,088
official-boundary overlaps, leaving 53,166 unique candidate and source-row
identities. The final candidate SHA-256 is
`85256ecea000b6aa0a2b1e638a61af87362e6d7effc87e0722a0ac6e994da2d7`;
its exact attribution manifest SHA-256 is
`59fb344c48fa68eabf740faba647387afe305399c64d0d9ce7e3e0aea67d6119`;
and the canonical run receipt is
`eb7e822323ec3cd928cd5b7775809207ed610634ce0c123ce77546b23065ab1e`.
After final candidate custody was hash-verified on Stokes, three redundant
recoverable intermediates were deleted locally, reclaiming 1,645,579,101
bytes. No mixed Hugging Face source
shard was deleted. The three source-safe receipt files were force-downloaded
and replayed byte-for-byte from dataset commit
[`207d24434b073d552003d11154ab43bfe2c1bdb0`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/207d24434b073d552003d11154ab43bfe2c1bdb0).

The companion Public Domain Review pass scanned all 1,406 parent rows,
excluded 36 prior-audit and 17 short rows, and removed ten official-boundary
overlaps. It leaves 1,343 unique candidate/source-row identities. Its final
candidate and attribution SHA-256 values are
`1ac38f4a6dde7e05011da644fa7e8db7acd5a0db3432eb51d259874bd5fe80a2` and
`81401c05a387dbcbeef157c892912fab055d589b7d049d1d79ee72b687b7811e`;
the canonical run receipt is
`1214c429cdd9251998c0c947660344e63d0384d513ff2c8c1c07d896b8e03cac`.
After durable candidate custody, 22,406,095 bytes of redundant local
intermediates were reclaimed. The source-safe
PDR receipts replay byte-for-byte from dataset commit
[`2360c039136873ef3b4a653bf642425a78fe440a`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/2360c039136873ef3b4a653bf642425a78fe440a).

Cross-source normalized exact deduplication then covered all 54,509 surviving
Pressbooks and PDR candidates. It found zero duplicate groups and zero drops.
The canonical run and source-safe publication receipts are
`1be527cf0a814b7586a350757c9f63c90ee43844cae67ddb124929c650a397b4` and
`7e69e12d131e4cc78d6956d0f7793a418c1940c202ea49feb028943d40576de2`.
Because its 546,324,994-byte combined output was only a deterministic reordered
copy of two already durable inputs, it was reclaimed after evidence custody.
The later unfiltered subdocument candidate was also moved to the separate
Stokes evidence root before its 535,008,085-byte local copy was permanently
unlinked; the clean 54,496-row replacement remains local and durable. Across
all evidence-backed cleanup to date, 6,036,268,343 local bytes have now been
removed. The source-safe exact-dedup receipts replay from
dataset commit
[`526e801fae1fb01a4f9ced8f260c6a2ef51c7823`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/526e801fae1fb01a4f9ced8f260c6a2ef51c7823).

The next operational work is to expand sustainable, stratified compiler lanes;
build cross-source exact and semantic duplicate families; populate the concept
prerequisite graph; create verified English translations and grounded
representations; and measure accepted bytes and tokens by domain, culture,
style, complexity axis, and epistemic function. Only the accepted, replayable
output of those steps can become a curriculum shard.

#### Live virtual-corpus production and spiral index

The large-source path is now storage-bounded rather than copy-bounded. The raw
Hugging Face reservoir contains 8,802,247,613,960 immutable source bytes across
13,974 objects, but Stokes has roughly one terabyte of working quota. Sai does
not create a second 1.5–2 TB PleIAs copy. It records exact final locators after
quality selection, benchmark decontamination, internal subdocument deletion,
cross-source subdocument deletion, rights custody, and source-disjoint split.
The final tokenizer/packer can reconstruct each selected row transiently from
its pinned upstream object and must reproduce the sealed final content hash.
This makes the corpus replayable without calling raw reservoir size accepted
training data.

The live Institutional Books materializer has completed **all 64 of 64** private
shards. Final identity 48 retained 5,955 of 5,964 selected rows, excluded nine,
and sealed 1,159,146,835 enriched tokens under shard receipt SHA-256
`354d1fdc656bcda98548a8cd10769f6766464a6be048417344ed3c94e762d3fa`.
Create-only aggregate job `818440` completed its full replay in 11:42 with zero
exit status. It accounts for 382,961 selected rows, 382,166 materialized rows,
795 exclusions, 69,845,861,132 enriched tokens, and 86,116,375,079 private bytes
across 4,664 files. Its semantic receipt SHA-256 is
`ab15d696872cdbd593aef53733b38473b3b72fe4db78b41fcbff8a55be0d9c77`;
the aggregate file SHA-256 is
`0745b4667a94f73a6a1bc20ab184388fcc91d7b39225b75e9129e46c8b44148f`,
mirrored read-only beneath the authorized durable evidence root. That receipt
released all 64 independent mechanical-gate identities in array `818505`.
The PleIAs metadata census has completed at least **79 of 128** canonical
shards. At the four-hour acceleration cutoff, eight still-running
identities—1, 42, 69, 72, 74, 75, 84, and 95—had
measured projected remaining runtimes between 1.35× and 3.68× their remaining
walltime. Those originals were terminated before any recovery work began; their
partial directories were moved intact to the durable evidence root. Independent
eight-segment arrays `818902`, `818911`, `818920`, `818929`, `818938`, `818947`,
`818956`, and `818965` were then admitted immediately, with merge jobs `818903`,
`818912`, `818921`, `818930`, `818939`, `818948`, `818957`, and `818966` rejoined
to aggregate `818244`. The surviving likely-finish original workers continue
unchanged, and the normal dispatcher recognizes the early dispatch markers so
it cannot duplicate them. The source-safe acceleration receipt has SHA-256
`11018582d87511616b52cd19c5bea5be84f45ed9e2d78f106f742efcfac401f8` and is
copied with all eight dispatch markers beneath the authorized durable evidence
root. These are materialization/census measurements, not final admission
counts.

Deadline projection then identified 13 additional originals with more than
three measured hours remaining. They were transitioned through the same
cancel-before-recovery boundary, bringing accelerated custody to 21 identities
and 168 nonoverlapping segment tasks while leaving faster originals intact. The
second dispatch receipt has SHA-256
`24c82b088aab91d359911877f87720812a7070b5f18cc028d5d9d4edc72b99e7`.
At the final four-hour deadline pass, five more exact identities—48, 63, 101,
103, and 124—still projected to require 2.25–3.06 hours as serial originals.
Each original was verified terminal before its partial directory was preserved
and its eight nonoverlapping recovery segments were admitted. Arrays `819223`,
`819232`, `819241`, `819250`, and `819259` feed merge jobs `819224`, `819233`,
`819242`, `819251`, and `819267`; aggregate `818244` now requires all five exact
merges. This brings accelerated custody to 26 identities and 208 independent
segments. The immutable five-row dispatch receipt has SHA-256
`76d8cdaf5577bfa3983c93faa9e3dc7f565b66b1c131e727a0eb4cf1c6de8378` and is
read-only beneath the authorized durable evidence root.
During admission, `ec65` cancelled 29 segment identities in one or two seconds
with zero output. The host is now excluded from every census recovery request;
only those exact failed segment indices were resubmitted as repair arrays
`819092`–`819097` and `819101`–`819107`. Each existing merge was rewired to an
AND dependency on termination of its surviving original array and successful
completion of its repair array. The 13-row repair receipt has SHA-256
`de4a13f40b4958210b9545cb7a8b6883b5f0e2ecd9cb2654b69f8d87f1a11829`.
Both receipts and all 21 dispatch markers are read-only in the durable evidence
root; no healthy segment was duplicated or cancelled.

The dependency graph already stages the complete PleIAs virtual pipeline:
subdocument signatures, global signature aggregation, exact deletion decisions,
internal rewrite replay, cross-source decisions, final locator reconstruction,
exact final byte balancing, global byte/document accounting, a two-component
corpus ledger, and durable
custody evidence. Independent bounded tokenizer samples then feed three
separately built 32K, 48K, and 64K candidates. Every stage is CPU-only,
non-requeueing, create-only, and pinned to an immutable runtime; no 4B training
job is part of this graph.

The four-hour closure pass verified that the older physical PleIAs
materialize/rewrite branch had zero consumers in the final ledger: byte
allocation job `818779` depends only on virtual PleIAs aggregate `818642` and
virtual Books aggregate `818644`. Its 15 still-pending jobs (`818564`–`818570`,
`818574`–`818575`, `818577`–`818578`, and `818583`–`818586`) were therefore
cancelled before any task started, preventing redundant multi-terabyte writes.
The ten source-disjoint, non-model virtual reconstruction, tokenizer-sampling,
and curriculum arrays that feed the final ledger were raised to 32-way
admission. This changes no source identity, selected row, model request,
decision, or output byte. The exact scheduler receipt is
`artifacts/sai_four_hour_corpus_acceleration_20260824_r1.json`.
It was force-downloaded byte-identically from Hugging Face dataset commit
[`5d4038c59a95cb8042d2e8b499173c8229448cea`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/5d4038c59a95cb8042d2e8b499173c8229448cea).

`sai.data.virtual_spiral_curriculum_index` adds the curriculum layer after the
two final component aggregates close. It emits source-text-free Parquet rows
that bind component/shard identity, final content hash, exact UTF-8 byte count,
train/development split, rights, semantic quality, three difficulty signals,
concepts, prerequisites, source custody, and a deterministic curriculum
priority. Difficulty is the maximum of conceptual difficulty and prerequisite
burden, divided into foundation, intermediate, advanced, and expert bands. The
prospective moving-center schedule is:

| Stage | Run fraction | Foundation | Intermediate | Advanced | Expert |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foundation | 25% | 65% | 25% | 8% | 2% |
| Expansion | 35% | 40% | 40% | 15% | 5% |
| Depth | 25% | 20% | 40% | 30% | 10% |
| Synthesis | 10% | 10% | 25% | 40% | 25% |
| Annealing | 5% | 10% | 20% | 35% | 35% |

The aggregate reopens all 192 component shards, recomputes every derived row,
checks exact document and content uniqueness in a disk-backed database, binds
the final source receipt for every shard, and reconciles per-component and
per-split document and byte totals. It deliberately leaves token counts and
exact stage allocation open until the selected tokenizer retokenizes the final
stream. Curriculum indexing is therefore a necessary custody layer, not a
claim that the 2 TB corpus is already training-ready.

`sai.data.virtual_curriculum_coverage` then streams every indexed row again and
measures the quality-density claim directly. Its prospective contract requires
1.9–2.0 trillion post-rewrite UTF-8 bytes, all 20 polymath domains, all four
spiral bands, a development partition in both components, concepts on at least
90% of documents, prerequisites on at least 25%, multi-domain metadata on at
least 5%, and at least 80% of bytes at semantic quality floor 6/10 or higher.
It records unique concepts, prerequisites, domain pairs, quality-floor mass,
and every component/split/band/domain byte total. These thresholds are a corpus
qualification hypothesis that downstream proxy evaluations may tighten; they
do not replace real capability measurement.

The analyzer explicitly marks domain-label co-occurrence as **not verified
connection data**. A final corpus still requires independently verified,
source-grounded bridges plus exact token allocation from the selected tokenizer.
Its receipt is mirrored to durable evidence and remains non-training even when
the structural coverage gate passes.

The independently generated connection-data path now has a separate curriculum
candidate compiler. After both the same-family verifier and Nemotron Ultra
retain a bridge and the complete generated text clears the official benchmark
boundary, the compiler creates four exact lesson forms: a bridge overview,
each verified representation, explicit analogy limits, and answer-bearing
verification questions. Every representation from one anchor pair remains in
one deterministic provisional train/development group. The compiler now binds
both original anchor identities, content hashes, and source metadata so a later
foundation scan can move the entire pair onto the anchor's actual global split
or reject a pair whose anchors cross splits. Exact and whitespace-normalized
duplicates within the bridge component are rejected. Pair-disjoint placement
is not misreported as global source-disjointness.

The corpus-wide bridge boundary is now executable rather than aspirational.
`grounded_bridge_foundation_query.py` converts every generated lesson and both
anchor coordinates into a source-text-free SQLite query database containing
only exact 13-word and eligible eight-token code signatures, candidate
identities, content hashes, and normalized source keys. The source-key logic
reconciles both `repository@revision` and separated repository/revision
provenance. Identical inputs produce byte-identical databases, and any database
or receipt mutation fails before a foundation row is read.

`grounded_bridge_foundation_scan.py` then replays all **128 final PleIAs shards
and both train/development partitions of all 64 final Institutional Books
shards**. It checks bridge signatures against the final rewritten text while
also comparing each anchor with original, intermediate, and final content
hashes, so deduplication rewrites cannot hide shared ancestry. Each shard emits
only sorted matching SHA-256 digests, source-group/split anchor matches, exact
coverage hashes, and accounting; it never persists foundation or bridge text.
`grounded_bridge_foundation_scan_aggregate.py` requires all 192 receipts,
assigns every surviving pair to its matching foundation split, rejects an
entire pair when anchor matches cross train/development, holds an overlapping
representation, and rejects a pair when every representation overlaps.

`grounded_bridge_foundation_reconcile.py` physically writes only the surviving
generated representations, binds every output row to its document and pair
decision, and replaces the provisional pair split with the reconciled global
split. Its output remains explicitly ablation-pending and non-training. The
completion watcher builds the query after independent bridge verification,
copies candidates and hash evidence to Stokes, creates a read-only immutable
runtime, and dependency-stages 128 PleIAs scans, 64 Books scans, their aggregate,
and reconciliation behind the already-frozen final foundation jobs. The full
implementation currently passes **1,340 tests**, including deterministic replay,
tamper rejection, exact overlap ownership, source-key aliases, conflicting
anchor splits, and compiler-to-query integration.

The connection-data verification population has now closed all 512 grounded
bridge candidates under both model families. Conservative intersection retains
460, routes 47 to revision, and rejects 5; all 460 retained bridges pass the
official-boundary contamination screen. Compilation produces 3,052 prospective
train documents and withholds 168 pair-disjoint development documents. The
complete candidate stream is hash-replayed in Hugging Face commit
`2a05f42030c209c5f1c5221629bb44751c782c06`. This is a completed candidate
result, not an admission result: final retention still requires global
foundation overlap reconciliation and positive transfer measurement.

These bridge lessons remain candidates, not admitted data. Their receipts keep
global deduplication and split reconciliation against the final foundation
corpus, transfer ablation, and training readiness false. Candidate publication
is complete, but publication alone is not admission. This prevents attractive synthetic prose from entering Sai merely
because two model families liked it; the connection data must still demonstrate
positive transfer and survive the complete corpus-wide duplicate boundary.

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
hash-bound**; the 8T-token spiral is **prospective**; one production book, 224
Common Pile confirmation rows, and both bounded source pilots are **judged**;
the two pilots contain 3,290 benchmark-screened, near-deduplicated rows and
have passed the source-specific full-candidate-materialization gate without
becoming training-ready; the UltraData Math L2/L3 screen has 148
benchmark-disjoint Hermès judgments with only 2 direct representation routes
and therefore requires targeted recovery rather than bulk admission; the
complete PEP candidate census has 567/567 Hermès judgments, including 322
representation-verification routes and 61 exact quarantines now enforced by
the deny list; the complete 1,024-row PleIAs screen has only 139 direct
representation routes and 417 exact quarantines, bringing the global deny list
to 1,245 identities while its full metadata census determines which measured
strata are worth retaining;
a 1,024-row CC0 arXiv temporal screen has 1,023 benchmark-disjoint survivors,
zero near-duplicate pairs, and complete declaration coverage awaiting Hermès;
the complete arXiv parent census measures 2,458,156 mechanically eligible
unique rows and 2,380,856,330 text bytes while leaving every downstream
quality, contamination, deduplication, and training gate open;
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
