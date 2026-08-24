# Sai Hugging Face Data Registry

Status: active public registry at
[`Godlydonuts/Sai`](https://huggingface.co/datasets/Godlydonuts/Sai).

The registry is the durable transfer surface for Sai data. It is not a flat
dataset and it does not imply that every published artifact is training-ready.

## State machine

Every record advances through explicit, non-interchangeable states:

1. `source_reference`: upstream repository, revision, license/terms, and exact
   hashes;
2. `candidate`: bounded source material awaiting compiler judgment;
3. `compiler_judgment`: source-bound quality, representation, translation, and
   prerequisite decisions;
4. `verified`: derived material whose grounding, duplication, contamination,
   and transformation checks passed;
5. `curriculum`: records assigned to epistemic function and prerequisite-aware
   spiral phases;
6. `training`: final immutable shards with tokenizer, ordering, exposure, and
   replay receipts.

Only state 6 may be consumed by a training job. A model judgment is not a
verification result, and an upstream quality label is not a Sai admission.

## Repository layout

- `registry/` stores the complete local artifact hash index, upstream source
  registry, and sanitized build manifests.
- `compiler_judgments/` stores strict Hermes outputs that remain
  `training_ready=false` until verification.
- `curriculum/` stores Sai-authored concept graphs and scheduling inputs.
- `verified/` is reserved for verified representations.
- `training/` is reserved for final packed populations.

## Conversion-yield ledger

Dataset commit
[`f5ec9e07e987f008c52a29b31922c2e361c8472a`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f5ec9e07e987f008c52a29b31922c2e361c8472a)
adds current conversion ledger release r6 under receipt
`1970d1ecdb70ccc6f554ee9927df49eb2b42d15ed3a516bc955862748a545fb6`.
It retains every r5 reservoir, audit, rights, and text-probe binding and adds
two bounded source pilots: 3,353 raw rows, 3,301 benchmark-disjoint rows, and
3,290 within-pilot near-deduplicated rows totaling 27,573,127 bytes. The
separate exhaustive cross-source pass covered the full 3,290-row bounded
population and found zero additional duplicate groups. Rights and
representation verification remain incomplete, so training-ready bytes remain
zero. The remote ledger file SHA-256 replayed as
`36b2dcc3542b9a282eb4836fcf1d64883d3c75624c33ab25464998bca007cb86`.

Dataset commit
[`7b9c6a2d57f60cb9fa4e98f26d89925a29975413`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/7b9c6a2d57f60cb9fa4e98f26d89925a29975413)
adds the superseded conversion ledger release r5 under receipt
`f8369fca50e0142ee8fd505f1bf6aa9167f7a41bddb4c641a673249c9c881083`.
It retains all r4 population, rights, and probe bindings and additionally
verifies the exact FinePDF probe. Across both probes, nine measured members
total 13,359,494,149 physical bytes, 22,576,343,154 text bytes, and
17,638,716,209 bytes in the mechanical useful-size window. Full-reservoir text
yield remains explicitly unmeasured and no sample extrapolation is allowed.
Training-ready bytes remain zero. The remote file SHA-256 replayed as
`b79f5b4d4b35bf98991887bf43b265e3b44dde18f73022e31d69e753fa98ecfa`.
Earlier releases below remain immutable historical evidence.

Dataset commit
[`8d83bab552d5b6cbd4ca82fcc068f6cedc21567c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/8d83bab552d5b6cbd4ca82fcc068f6cedc21567c)
adds conversion ledger release r4 under receipt
`d6e9cd17cf3515bde743bf036c15e712342760713dbfcdfc66a4283c562e67a6`.
Its eight measured members contain 12,252,341,634 mechanically useful bytes;
its remote file SHA-256 is
`0a4461ea06f8d4be5a506472badce66ee901033fd03d185343d4a310ff408c16`.
It is superseded by the FinePDF-bound r5 accounting above.

Dataset commit
[`4b468991397fb123f4bf73674803ac931c1dd2ff`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/4b468991397fb123f4bf73674803ac931c1dd2ff)
adds the complete-population ledger release r3 under receipt
`b8d5e6278aa9a076c4d143807f6e09f64b75ab992564651f9fa94dbcf5cb2337`.
It binds six distinct immutable populations totaling 2,103 acquired audit rows
and rejects duplicate population receipts. Reservoir bytes and rights routing
are unchanged from r2: 23,680,076,298,761 referenced candidate bytes and zero
training-ready bytes. The remote file SHA-256 replayed exactly as
`6db524e74dc64bafe2b93261fe6f9b3ef99aef30de450b71ed2637801776c163`.
It is superseded by the probe-bound r4 and r5 accounting above.

Dataset commit
[`8c3baa4452d8bf06c2277e72f4dd79b5628e8d26`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/8c3baa4452d8bf06c2277e72f4dd79b5628e8d26)
adds the rights-routed ledger v2 under receipt
`73c6ecaca197bae639cd2a66642d07af8a4fd4a26ce8ba8f3a6211a599ad244b`.
It binds the same 23,680,076,298,761 candidate bytes to corrected rights
inventory v2 and routes 7,899,196,133,417 bytes to declared-license obligation
handling, 5,027,859,142,584 bytes to per-row evidence, and
10,753,021,022,760 bytes to source-terms resolution. The ledger still records
zero completed pilots and zero training-ready bytes; it grants neither legal
clearance nor training admission. The remote file SHA-256 replayed exactly as
`74562e87fe36480e98a3c47c785156da71435ed424a75b558dab03b97c7d193a`.
It is superseded by the complete-population r3 accounting above.

Dataset commit
[`22eb617b741bc21c38d154f36fe040e8652e7b2a`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/22eb617b741bc21c38d154f36fe040e8652e7b2a)
adds the first source-safe conversion-yield ledger under receipt
`b6a984a0552f2fc58352d5d66a79c4cdd8d771aff26f82a0f12e0fb0cf4bcea8`.
It hash-verifies two distinct reservoir manifests totaling
23,680,076,298,761 referenced candidate bytes and two Common Pile population
receipts totaling 348 audit rows. It records zero completed bounded pilots and
zero training-ready bytes. Cross-inventory overlap and exact text-payload yield
remain unresolved, so candidate physical bytes are not presented as unique
text, tokens, or training data. The receipt contains no source text and its
remote SHA-256 replayed exactly as
`659e2eee6d95e065f1c6f5d2a31256847adf60f6e8b5d96ec19cfd79b173792c`.

Dataset commit
[`53e71e0c0f6e794d933421d5e459a0ec70e3f933`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/53e71e0c0f6e794d933421d5e459a0ec70e3f933)
adds the text-free full-reservoir rights inventory under receipt
`6ea8d853f23de1afded8bf66033fa4e873898c8608677b2fa191341d5f7bdf0c`.
It binds 46 source lanes, 45 exact repository revisions, 42,600 files, and
23,680,076,298,761 physical candidate bytes. Eleven lanes route to recognized
declaration obligations, 31 to per-row license evidence, and four to
source-terms resolution. One exact pinned tree, Common Pile StackV2 HTML, has
no README; the inventory records the absence instead of borrowing a mutable or
different-revision card. The remote file SHA-256 replayed as
`a0f316bb11b75e7c3d49594ea10bbec90e91229f52ca44c2e230886cd2da6d1e`.
The inventory is evidence routing, not legal clearance or training admission.

That first routing is superseded. It allowed a recognized wrapper-card license
to outrank a manifest declaration containing upstream/generator terms. Dataset
commit
[`b7b60404ab737b9fd1e44740f6f781dc8d56da38`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/b7b60404ab737b9fd1e44740f6f781dc8d56da38)
adds corrected schema v2 under receipt
`8e72391081af17323aa1e1b8d0480ddbe70dcb232006e6cf37ed7228d34d3d80`.
It routes five exact manifest declarations to obligations, 31 Common Pile
lanes to per-row evidence, and ten composite/ambiguous lanes to source-terms
resolution. The remote receipt SHA-256 replayed as
`2362e903a4bf067863971a50cf2a1445de670bbd46c84cf2093767d45b02aca1`.

The first registry commit is
`89152fff3e47d85e35e75cad6b419b4f304a4e85`. It contains 22 paths: the dataset
card, a 462-file local artifact index, five source-registry entries, a sanitized
Institutional Books pilot manifest, 14 FineWeb-Edu compiler judgments, and four
foundational-syllabus artifacts. The local artifact-index SHA-256 is
`a86a6e294ed8dd42f9a0743ce508febce65fa5f8e545c3deafa08bb40481d9ec`.

## Publication boundary

Each indexed artifact has an explicit publication disposition. Default is
staged review, not upload. Source-specific terms always override the dataset
card.

- Gated Institutional Books text and excerpt-bearing candidates are
  `reference_only_gated_no_redistribution`; only sanitized hashes and counts are
  public.
- Sai-generated curriculum graphs are publishable.
- FineWeb compiler judgments are published with upstream lineage but remain
  non-training artifacts.
- FineMath, Dolma, and authored-source artifacts stay staged until their exact
  source-specific license and redistribution bundles are complete.

## Frontier audit mirror

Dataset commit
[`35efe5b49e62a44dbd430f2c238116acbc571e82`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/35efe5b49e62a44dbd430f2c238116acbc571e82)
publishes text-free external-provenance metadata for every one of the 3,290
bounded pilot survivors. Exact parent replay recovered 2,844 unique Pressbooks
chapter/book URLs and 1,342 Public Domain Review article URLs plus source row,
native ID, declared license, and bibliographic fields. The parents were
size/hash verified and removed. Manifest SHA-256 values replayed as
`253e433031600aae7a5b1155122d2f93afd276f8e61d8c61663a5b6b87dfef40`
and
`d4228cc643ae4797b294cc1b697ee8f97c61082d7f00cbf7469f7af57e8eb4d1`.
The source cards warn about license laundering and inaccurate metadata, so
this is internal metadata lineage—not external page verification, rights
provenance, legal clearance, or training admission.

Dataset commit
[`be6becc19c2e2e1bccfa12640b9f5ca4368da43c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/be6becc19c2e2e1bccfa12640b9f5ca4368da43c)
publishes the text-free live rights-page probe for the bounded pilots. The
probe covered all 3,290 retained records through 1,160 frozen targets: 901
unique Pressbooks work pages, 258 Public Domain Review essay pages, and one
official Public Domain Review reuse-policy page for 1,084
collection/conjecture records. Public Domain Review returned 259 HTTP 200
responses with matching CC BY-SA evidence, covering all 1,342 records.
Pressbooks returned 193 HTTP 200 responses, of which 182 contained the expected
declaration and covered 377 records; the other Pressbooks outcomes were 632
HTTP 403, four HTTP 401, two HTTP 404, and 70 exhausted transport retries.
Overall, 1,719 records had matching page evidence. No HTML was persisted and
no response was truncated.

The page probe is a bounded temporal observation, not a rights decision. It
does not establish which page region a declaration governs, cure upstream
license laundering, or convert a blocked response into permission. Its receipt
therefore retains false values for rights-provenance verification, legal
clearance, training readiness, and 4B authorization. Receipt SHA-256 is
`2483d76d4c596541044cd45eda8c73ad0b6539c5e5767f6f3529865f8ee5b5de`;
the `results.jsonl` SHA-256 is
`5f85c155881a820153136bfb2c4c774cd9f16a468f6fb1a0be207893db1966e8`.
Both remote files replayed byte-for-byte.

Dataset commit
[`e1a2f00a121cbfec417cabe657111e5cb6a2de30`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e1a2f00a121cbfec417cabe657111e5cb6a2de30)
publishes the resulting 3,290-row text-free, fail-closed adjudication queue.
Observed evidence routes 377 Pressbooks records to book/section scope review,
258 PDR records to essay license/exception review, and 1,084 PDR records to
policy-scope and embedded-material review. Unresolved Pressbooks outcomes route
52 records with HTTP 200 but no expected canonical pattern to declaration
review, 1,300 records behind HTTP 401/403 to source-owned-statement review, two
records on HTTP 404 pages to moved/withdrawn-source review, and 217 records with
exhausted transport outcomes to manual source-existence review. The queue did
not bypass access controls and makes no automated legal decision. Receipt
SHA-256 is
`afc84f09628ca9153c626d1f527f715b23bb967a644a19ca3f719db5825fe7c3`;
queue SHA-256 is
`8273f615b97e13a7cd7815078acc4b60666a1874e2b32355eed507ab1e638335`.
Both remote files replayed byte-for-byte.

Dataset commit
[`c2dbb5dfe68a85b06e85c5d1962162d12a62c68f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/c2dbb5dfe68a85b06e85c5d1962162d12a62c68f)
publishes both immutable generations of the text-free Public Domain Review
scope audit. The audit replays all 1,342 frozen candidate identities against
their live source pages using the selector geometry pinned from Common Pile
commit `9457f04a14cb2355ab00023420369d46ffd4a395`, requires the applicable
official-policy or page-specific CC BY-SA evidence, excludes nested
`blockquote` and `q` elements, and retains only hashes, counts, routing status,
and transport evidence. It never stores page HTML, source text, or scoped text.

Generation `20260825-r1` is retained as superseded audit history because it
revealed that a live license footer also matched the upstream content-block
class. Active generation `20260825-r2` deterministically removes that exact
license UI before replay. It has 1,342/1,342 unique signed results: 1,253 exact
live-page replays, 85 source-page drift reviews, and four unavailable
responses. It excludes 961 quotation elements containing 446,625 codepoints.
The active receipt SHA-256 is
`779eeef0a192dcd73744f68aa47af46305c5a45391601a244a87be3cdbf0f40a`;
its result-file SHA-256 is
`3d9b8793a2696dfc072ab226262c577ceb76212a22794fcd312d8e254a7d6271`;
its ordered per-row result digest is
`50fff9f3d15d52def277717ff2901847baf53952196c99d0cb3fb723a9f4d8c1`.
Forced remote download reproduced both r1 and r2 receipts, files, row counts,
and per-row signatures exactly. Neither generation is a legal decision,
rights clearance, training admission, or 4B training authorization.

Dataset commit
[`6885a18a0a98eb10c3d5d0e73ad276dd49a99a0d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6885a18a0a98eb10c3d5d0e73ad276dd49a99a0d)
publishes the first actual text-bearing output of that scope audit. A third live
replay materialized all 1,253 eligible pages—995 Collections, 243 Essays, and
15 Conjectures—into 5,919,449 UTF-8 bytes. It excluded 883 tagged quotations
containing 412,039 codepoints; all 85 drifted and four unavailable pages remain
absent. Every row carries its exact source URL, upstream identity, provenance,
rights, scope-result, attribution, and share-alike bindings.

Because removing a quotation can create new token adjacencies, the transformed
text was screened again against the pinned official benchmark boundary. All
1,253 rows survived with zero exact word-shingle and zero eligible-code-shingle
hits. Materialization receipt
`52484c5f8b22d79b231e71d2d03962fd10ea18b29c6740c02b86afd25ebd7741`
and post-scope screen receipt
`9a051d33874a8515938d072914dbe3888e7cde52ed7eff7754c12d7efd528097`
replayed exactly. A forced remote download also reproduced all seven files,
every candidate signature, every text-free contamination decision, and the
candidate SHA-256
`04626195b471db0a543a1de7a7b3feb672696ac0069da9a1025c64e11b2ddbf6`.
The population remains candidate data: content quality is not yet independently
verified and no row is training-ready.

Dataset commit
[`bb34c47c1cf77f3bb9b3603ccdfa8c61ac6d2caf`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/bb34c47c1cf77f3bb9b3603ccdfa8c61ac6d2caf)
publishes the source-safe bounded-pilot compiler population receipt and its
text-free 3,290-row lineage. The compiler input contains 1,948 Pressbooks and
1,342 Public Domain Review survivors with exact pilot, decontamination,
attribution, source-revision, file, and row bindings. Candidate text is not
published. Receipt
`b87e7c864fec79de60dc90576777b347dfd66bdbe40af40c42db9f91f422a372`
and the remote receipt/lineage file SHA-256 values replayed as
`b9d82b58176dc7ac9ff5952f5ecbb1e30e441405d8411fa56c3ce2101a7e8c17`
and
`b68ff632fa6c5fda2677a289569679387d29157771a1f382851e0ef983fe76a0`.
Compiler judgments remain in progress and establish no verification or
training admission.

Dataset commit
[`44fbdd30cedc89ac908057929468d3162651d645`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/44fbdd30cedc89ac908057929468d3162651d645)
completes the source-safe Common Pile confirmation mirror with the aggregate,
per-source decision, and frozen promotion-v2 receipt. The 224-row confirmation
contains 207 retain, 11 review, and six reject model verdicts; deterministic
conservative routing instead assigns 141 rows to representation verification,
49 to cleanup review, 15 to factual-grounding review, 17 to quarantine, and two
to transformation review. All 32 shard summaries are complete, 187 rows carry
cross-domain bridge candidates, and total measured usage is 958,783 model
tokens. Only Pressbooks and Public Domain Review are authorized for bounded
streaming pilots. No full-source ingestion, bulk admission, or model training
is authorized. Remote file SHA-256 values replayed exactly as
`52633a9edef045da4ed3a1e04ede23ea79c42fee86846350a72cbaa6846d8983`,
`4b9d4d71ed00dd2a5827cf46b8cd8c81c0f24d46e022ef2937bb56adc09122e0`,
and
`32b72f5e6fd1d25b478b2c7c59844f71c610ad7d6368005ba7463ca7fd43035e`.

Dataset commit
[`de17529bd3ba9ea67355c26985b70350e6b8377f`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/de17529bd3ba9ea67355c26985b70350e6b8377f)
adds seven source-safe files and passed a byte-for-byte post-upload replay. The
512-row frontier screen has population receipt
`b772920d9c86d5eddeae69f338e53a7d1b520f161f034b0616c3ee2140088631`
and zero-pair duplicate receipt
`26ebacefe45c320a9f319ecf42f63fe5c510236879581e1cbc48293bc65fc8b1`.
The 91-row PleIAs/Nemotron expansion has population receipt
`2c91d84c8ed64a008f46c72062a5f387ce6bebac1a08ad08fa41d9079a05b5eb`
and zero-pair duplicate receipt
`35df65eee58e3a4b67cb2f409666a201350cdba8c27438ff992b53a7b3397f8b`.

The combined exact-content report covers all 1,879 candidates in the five
screen populations and reports zero cross-population pairs under receipt
`e31954f5bd2b220004c6b19c0dd35949052f74a464f0ad009af476e2f6dff0be`.
This does not estimate full-reservoir or semantic duplication. Only lineage,
population receipts, duplicate receipts, and the combined statistics were
published; raw candidate text and evidence-bearing judgments remain local.

Dataset commit
[`bd7d7cd92bfe61d8b9b0dfda8790d11d0fa3cdef`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/bd7d7cd92bfe61d8b9b0dfda8790d11d0fa3cdef)
adds the completed 128-row original-reservoir aggregate under receipt
`c0706f92535aded29c679fff5c35798a6380c01b58dc9bdf95ffd155f9a76359`.
Commit
[`e388950b30231779215b688b1defbfcbf785f3df`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e388950b30231779215b688b1defbfcbf785f3df)
adds its deterministic source-work ledger under receipt
`7cd1a6b040eaa00a40eb37f2578045780815931d6f712a43d5bd33848a4e250e`.
The ledger is descriptive triage, not a source-yield estimate or training
admission.

Dataset commit
[`90a87727f9b5e88b0268153001f19d47c091101d`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/90a87727f9b5e88b0268153001f19d47c091101d)
adds the completed 124-row Common Pile aggregate and its conservative work
ledger under receipts
`d79749882b8e306e87997a2e0f13bd558e0bef268356b696e6d140eab656bd22`
and `e036d9d96bfc260fc3d64f6851db00213231545ac1aca2fc1bbe00ed4427ae58`.
Both uploaded files replay byte-for-byte against local SHA-256 values. Raw
candidates and evidence-bearing judgments remain local, and every source stays
`training_ready=false`.

Dataset commit
[`6618216352dbecfae8e3c92eef53d4e14e1e24f1`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/6618216352dbecfae8e3c92eef53d4e14e1e24f1)
preserves the first Common Pile confirmation plan under receipt
`a48d9860193460e037c095f5483eb18b4b5199ec6b7be05eba8c6ebcfe562676`.
It selects seven clean-signal lanes and 224 source-disjoint confirmation rows;
it contains no candidate text and grants no bulk or training admission. The
downloaded remote bytes replayed exactly against the local plan. This version
is superseded because its universal different-parent requirement is infeasible
for selected collections with only one pinned parent.

Dataset commit
[`77cb201f68dab8f447f3d3a6e81b63a9ee4407f5`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/77cb201f68dab8f447f3d3a6e81b63a9ee4407f5)
adds the executable v2 plan under receipt
`350e96f2c1bbffa473eb7801fcd43548b03141754622c1ba0cd55a1e7bb9e625`.
It keeps the same seven lanes and 224-row target, requires exact identity and
content disjointness, selects a different pinned parent whenever available,
and otherwise uses exact discovery-line and content-hash exclusions. The plan
contains no candidate text and grants no bulk or training admission; its remote
bytes replayed exactly against the local file.

Dataset commit
[`e12b599463c8dfe0ff88338aa00e6b472e8bc1af`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e12b599463c8dfe0ff88338aa00e6b472e8bc1af)
publishes the source-safe evidence for the 224-row Common Pile confirmation:
its population receipt, exact lineage, discovery/confirmation duplicate report,
and corrected benchmark-contamination screen. All four remote files replayed
byte-for-byte. Raw candidate text and future evidence-bearing compiler
judgments were not uploaded. Population receipt
`40e72050e1c5a44d0e7618413d6e731de23232be9982f8f4be5d13eada44b6a5`
binds 224 rows, 32 per lane, and 2,637,343,362 fully verified compressed parent
bytes. Duplicate receipt
`6fd6b8491a627021d2cfd2db75c6eb8b495bcd48044254452583037eff2f8785`
records zero flagged exact pairs. Benchmark-screen receipt
`02fa2ead3bd14689fb6f46bf7eaca4f1518342aea8e3c08393d44aac1eb9acba`
records 223 clean rows and one contaminated GitHub Archive row. These files
remain audit evidence, not training data.

Dataset commit
[`e6b1210f26a7fb7e06e45c193131aa71d2c574df`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e6b1210f26a7fb7e06e45c193131aa71d2c574df)
adds the text-free Common Pile rights-declaration audit under receipt
`357414811d687921225830732feae6f45508707f126c01cf7b01624eaed0df40`.
It binds seven exact repository revisions and README hashes, recognizes 222/224
row declarations, and records two unversioned GFDL LibreTexts rows as rights
holds. The remote file replayed byte-for-byte. It establishes neither source
provenance nor legal clearance and grants no training admission.

Dataset commit
[`2a085eacf1479293e3c369d7eaa8e476d7f84054`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/2a085eacf1479293e3c369d7eaa8e476d7f84054)
publishes the completed 91-row modern-source expansion aggregate and
source-work ledger under receipts
`afd82b43ac66f3a485d167b97f79fccc75bc67026c94e182d846a6923f9dea23`
and `487c50a9fd4ba39ae22e73e4478673762e60fbedef254be387766fa41d740978`.
Both remote files replayed byte-for-byte. Raw candidates and compiler judgments
were not uploaded, and the ledger grants no bulk or training admission.

## Exact text-payload probe

Dataset commit
[`e15ca127c695d2d42df04e15738e56525f0bb3ce`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/e15ca127c695d2d42df04e15738e56525f0bb3ce)
publishes the separately planned FinePDF probe. Its plan receipt is
`325382746db5836ccffa12ea437fcfdfaf12ee0f29e469ac47cf0e43c0559017`
and its measurement receipt is
`5947564751b941b18d8a025abd3451c2e81cfa6e6357c0cad28213561d372919`.
The exact 4,836,418,450-byte member yielded 9,944,850,928 text bytes across
414,000 rows: 397,166 rows and 5,386,374,575 bytes fit the mechanical useful
window, 4,821 rows were short, and 12,013 rows were oversized. Oversized
documents require structure-aware segmentation rather than blanket rejection.
The remote plan and result SHA-256 values replayed as
`acc165fb92dc6961c99c7c761d3b511277607dc746cd5191a8a137e5bade29b7`
and
`07115706935ee29104a000bc885ddcae24a9fd0a1a7b20bdd15ef05b450da48a`.
The temporary source member was removed, and no source text was published.

Dataset commit
[`fecd9d596c18dd63ab6ea7a89dda7b2544eca4a1`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/fecd9d596c18dd63ab6ea7a89dda7b2544eca4a1)
publishes the source-safe plan and result for the first exact text-payload probe.
The SHA-256-ranked plan receipt is
`4f5312f7d9ae86b3fbe8998c7e780c7238eae9394fc767fdbedad2affbacc66c`;
the measurement receipt is
`1d550e0abc513c5b4e61f0ce5890155bfff01bcdbd2a6896f9a078c26952f848`.
Eight exact members totaling 8,523,075,699 physical bytes yielded
12,631,492,226 text UTF-8 bytes and 12,252,341,634 bytes in the mechanical
200 B–128 KiB useful-size window. FinePDF's preselected 4,836,418,450-byte
member exceeded the frozen 4 GiB cap and was not replaced. Every measured
member matched its pinned full-file size and SHA-256 and its temporary bytes
were removed. Neither file contains source text. The remote plan and result
SHA-256 values replayed as
`f4afe3dcb6a0e1fb4808ef750c14e90c5c79f884257c715344d21923ba9cd471`
and
`62e82000db517e66e927747c25d521d3be113fa2bf5c34ed1f1f7bb60117968b`.
This bounded probe is not a source-wide statistical yield estimate, quality
decision, rights clearance, or training admission.

## Benchmark boundary mirror

Dataset commit
[`ad178281de02625f043359a89070e905944452b9`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/ad178281de02625f043359a89070e905944452b9)
stores the non-reversible official public-benchmark boundary under
`registry/benchmark-boundaries/official-public-20260824-r1/`. The 895,351,296
byte word index has SHA-256
`f470834f84ff24dbd1bee66c115a460e913b5384470cc8a91e09c56459111acd`;
the 61,025,632 byte code index has SHA-256
`8e1238e2fe382b532638dd82692647ebf431eb99905f2c64c9b487654f3ecb86`.
Remote LFS object sizes and SHA-256 identifiers, plus the downloaded receipt,
replayed against the local files. The index contains hashes only, never raw
benchmark prompts or answers. This r1 boundary is immutable audit history, not
the active gate: its code policy admitted punctuation-only windows and has been
superseded by the substantive-window v2 policy.

Dataset commit
[`5dc89bfeceadf56663a8f00c479f5d41d5229671`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/5dc89bfeceadf56663a8f00c479f5d41d5229671)
adds the five source-safe population screens. Each screen binds its population
receipt, index receipt, policy, aggregate source/stratum counts, and ordered
decision digest. Individual candidate decisions and source text are not
published. The exact receipts are:

- original 128: `a851a4210bdd48b525b46cada58eabca77c932afee00d86109f55b048dbb1613`;
- weighted 1,024: `e28c7afb9799671d5fc7d6275fd82299da23271380b964a85eb09b71ee903db4`;
- frontier 512: `f2e3c82b1386f6c668757a53fc6dd5e554ea79b6ecf992667ac155d3d74c933c`;
- Common Pile 124: `71fcf44f96340feb73c399c82b4be00fd8c555bf5287c78b1464d9ad06bbb54c`;
- PleIAs/Nemotron expansion 91: `cca8dfbd408d6f4bf8914e8e64b78c75db05cf927734131160f92099d741524b`.

These five r1 screen receipts are also superseded. Their 286/1,879 overall and
77/96 Nemotron conclusions are retracted.

Dataset commit
[`43ae57ee4981c78ae23c111436b1fc9b6aa27023`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/43ae57ee4981c78ae23c111436b1fc9b6aa27023)
stores the corrected create-only boundary under
`registry/benchmark-boundaries/official-public-20260824-r2/` and adds five
replacement `benchmark_contamination_screen_v2.json` files without overwriting
r1. The word index exactly reproduces the r1 SHA-256. The 15,225,728-byte code
index has 475,804 unique substantive windows and SHA-256
`d438ea1176ed8357b7139475d469ce42dbe4c147f62cbab301b48e26e68dea39`.
Boundary receipt
`9fee65cb9f99813407ea4d5e4c35b4bc0bb7659c1720342f0f50bd1a8c237667`
and every remote file replayed byte-for-byte.

The replacement screen receipts and contaminated-row counts are:

- original 128: `be7829eb302e4824477c08199b8ace2d41deacf0af9284a063c37284f2937785`, 6;
- weighted 1,024: `280103a098d2367d6f9bd5e7c2cb46e5ad8437fcf15d3a1d02d215d2dcc4fef0`, 26;
- frontier 512: `54f3630f3a1b1a7a77799550dda8c9f9031ef459a37ceaf232657174e091d4a8`, 28;
- Common Pile 124: `f8608096167fa77c208ed63a5e7270af408b9c4abce7e9eaaa4eb85bc5da2ca7`, 7;
- frontier expansion 91: `9b73917db65ea70b1e4b7c9d044a917ce30d5b3fd862e01d8f966a268177ea1a`, 2.

The corrected combined result is 69/1,879; Nemotron specialized reasoning is
25/96. Individual decisions, raw candidate text, and raw benchmark text remain
unpublished.

Credentials never enter the registry. The local `.env` is ignored by Git,
mode-restricted, and used only by authenticated clients.

## UltraData Math L2/L3 tier audit

Dataset commit
[`28227ed9ba5a22887c2a0bb3bee20502e0982253`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/28227ed9ba5a22887c2a0bb3bee20502e0982253)
adds three source-safe files under
`evidence/ultradata-math-tier-audit/20260825-r1/`. They bind a deterministic
160-row population from exact `openbmb/UltraData-Math` revision
`fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87`, 20 official dataset-server
responses, and the active benchmark-screen result. Twelve rows were rejected
for exact benchmark-shingle overlap, leaving 148 rows for dependency-staged
Hermes compilation.

The publication receipt has canonical SHA-256
`e9211d5a5936c624e3253c3ac2bd2bd8c3cac4c5f9dd2889596171e2bb2a25dc`
and file SHA-256
`407aec6659ecfb68a1824c833f774dc08f6a58a7fafbf2a2a71f816a3e6c1504`.
The population and batch-receipt files replayed remotely as
`bed980b96d35a608b08c7c3cfdb94ca1bf872f1ea81e2cdcb60c6d3ed17435a5`
and
`b8663c7da4f64fd1533e03f2274e3187efb18b76f84d5be672067ea440e46d96`.
No sampled source text, benchmark text, individual contamination decisions,
credentials, or absolute local paths were published. This evidence grants no
quality, mixture, bulk-ingestion, curriculum, or training admission.

## Common Pile PEP complete-parent census

The exact `common-pile/python_enhancement_proposals_filtered` parent at
revision `582170907dd303c207770fceacd38e6abf133edc` was acquired once and
verified at 3,723,467 compressed bytes with SHA-256
`4bb61eded5168ac7f0059a92ed242577c67e4fced8c0d019c84bfaca5596c791`.
The complete scan accounted for 655 rows: 36 audit-population identities and
one short row were excluded; 50 of the remaining 618 rows overlapped the
official-public benchmark boundary; and one of the 568 clean rows was removed
by bounded near-duplicate filtering. The final population contains 567 unique,
benchmark-disjoint candidates and 567 exact public-domain attribution records.

Canonical receipt SHA-256 values are:

- complete filtered census:
  `c1f18f641a31672cda7d2b10caf60769df766aa7edea62418d1089645319c92b`;
- complete survivor compiler population:
  `255e9aa09ec8d2f00c01db05b8eabb6bd06d9f93c77d59b1ed6e8ce2caf7a5ba`;
- source-safe publication envelope:
  `3eeb07c28d575542d87a670396452a155f0c97aaaeaa15f19d526f210568168e`.

The respective file SHA-256 values are
`acb93f59f4f66387f9a9c4439fc3b5ef6858b50454ea9b21e6d810b6b9c08878`,
`d75940bbe96a3a5f179199b9ac420829eee832e66a33a96c806dcbd39a84b02b`,
and
`4f05f5b7dbd97ed49fcc825fbf34d84933d959fc466f39f66ded296d1fc4040b`.
Only those three source-safe receipts are eligible for the public registry.
Raw PEP text, Hermès excerpts, text-free lineage containing local source
locators, individual decontamination decisions, and nested receipts containing
absolute local paths remain unpublished. Hermès compilation is dependency-
staged; quality, representation verification, curriculum placement, and
training admission remain false. All three files were force-downloaded and
replayed byte-for-byte from dataset commit
[`756d941130a01fabb042178bf94a67b230a64e4c`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/756d941130a01fabb042178bf94a67b230a64e4c).

## CC0 arXiv abstracts temporal screen

Release `20260825-r2` binds 1,024 source-disjoint rows from exact
`common-pile/arxiv_abstracts_filtered` revision
`dc1ceab4755eb037ec61e49cf1350dab7ceee6e7`. The population uses 32 equal
temporal strata, one deterministic 64-row dataset-server window per stratum,
and SHA-256 bottom-32 selection inside each window. Every response returned the
pinned revision; the 32 response receipts have file SHA-256
`4e6d7f489e7efefb6d568b279a87ad012d9be1f90282fdc6e372185d906503b8`.

The active benchmark boundary rejected one row with one eligible code-shingle
hit and no word-shingle hit. All 522,753 unordered pairs among the 1,023 clean
rows were then compared exactly; no duplicate pair or group was found. The
declaration audit recognized `CC0-1.0` for all 1,024 source rows with zero
rights holds, attribution obligations, or share-alike obligations. It remains
a declaration audit, not source-provenance verification or legal clearance.

Canonical receipt SHA-256 values are:

- source population:
  `ee78511de966815be8c39bf2b65ecb2881d422ee3ce39c22f90479e85af2c046`;
- clean population:
  `bb29883925207811857083b1fb4d5b166afd2d8d48b10b35b27f4d98405d8671`;
- duplicate report:
  `db28854afb609d6014a3b86f4dee47124b1005c0500f96624672f8b29041c65f`;
- rights declaration audit:
  `e4fb89a22676e0438e29be905872e024e2725ee1341db809072dbc6abae6fda6`;
- source-safe publication:
  `0014f665fbbd09c691d03c8964fd7841bd8932e8ff4947c25e9b0b98eaeecdca`.

The publication, source-population, batch-receipt, duplicate-report, and rights
files have SHA-256 values
`adbaaef3848eecef8a617408ee5761f913b5f58efa749ad94603c2a6fd48e9bb`,
`67e96dfe668c4f02b02c90044699e67e05df7c1b7442d2ae81306091df5f1deb`,
`4e6d7f489e7efefb6d568b279a87ad012d9be1f90282fdc6e372185d906503b8`,
`9b51a09558ece7191bf6841fd0630f913abde7482d7816bf5f7c7d61b3d757db`,
and
`0c1a7a5d894cd47a120da78cdcce4cb90d1268b19b1c2a4b918eaf12fc22c3ec`.
Those five source-safe files are eligible for publication; sampled text,
individual contamination decisions, and machine-local paths are not. Hermès
quality compilation, full-source ingestion, and training admission remain
false. All five files were force-downloaded and replayed exactly from dataset
commit
[`5047dee73c4acbdc0f2f1abf044ff5049d4d59e9`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/5047dee73c4acbdc0f2f1abf044ff5049d4d59e9).

## CC0 arXiv abstracts complete text-free census

Release `20260825-r1` streamed both exact pinned parents one at a time and
replayed their SHA-256 identities before and after scanning. It covered
2,504,679 rows, 1,128,382,223 compressed bytes, and 2,388,470,891 UTF-8 text
bytes without retaining source text. The scanner distinguished physical gzip
line coordinates from embedded upstream-provenance coordinates and replayed
all 1,060 protected audit locators against all 1,060 expected full-text hashes.
All source provenance values parsed and increased strictly; 1,654 missing
upstream positions account for 1,062,885 rows whose embedded coordinate differs
from their physical line.

The mechanical funnel is exact:

- 2,504,679 scanned rows;
- 1,060 prior-audit rows excluded;
- 45,463 rows below the 200-byte floor;
- zero oversized rows and zero exact-content duplicate rows;
- 2,458,156 mechanically eligible unique rows;
- 2,380,856,330 mechanically eligible UTF-8 text bytes.

Canonical receipt SHA-256 values are:

- full census:
  `507561b16269da59bfe5f85ab9ae64e4f9b8b88d815078812d06f843e0cf2708`;
- source-safe publication:
  `00cc4bcd19ec550adef2f323e57b81746db7d52c07a6555fdeda16d86bfa52a3`;
- portable conversion-yield ledger r7:
  `b638e13a2e7cc74118071f14e29e6dbe0f0a0234f3d150644dc281a2dfe04c47`.

The corresponding file SHA-256 values are
`9dabab688f69ad4c6a9b6639e6c41ce3495f30c8fd7bcd5028a7bdbc75e5f086`,
`cf7706c396c9fd5f7112c419d52e78285755cb1ca209c22506c2ad76b2c7f16d`,
and
`9fb929317de7de7aecdee12b97b91150e99ffe3876ca92c49bd32655a68089bc`.
Only these three text-free receipts are eligible for publication. The raw
parents, individual rows, and audit text remain unpublished. Benchmark
screening, source-wide near-duplicate filtering, Hermès quality compilation,
curriculum placement, bulk ingestion, and training admission remain false.
All three eligible files were force-downloaded and replayed byte-for-byte from
dataset commit
[`288fe22adf52f8b5430cfa6834c039c45004cbbc`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/288fe22adf52f8b5430cfa6834c039c45004cbbc).

## OpenCoder code-web promotion screen and cache reclamation

Release `20260826-r1` contains three source-safe files under
`evidence/opencoder-promotion-screen/20260826-r1/`: a compact README, the frozen
promotion screen, and a recoverable-cache reclamation receipt. The screen
replays exactly 276 preselected identities across 16 logical shards. It passes
one of five gates (computer-science coverage) and records the frozen decision
`stop_full_audit_and_reallocate_hermes_capacity`.

The screen and reclamation canonical receipts are
`29a7ceed9841f99213d4087a40e0107277a07793b490760ee800242bcad7be70` and
`a45fa68a77018c1b58900b58aa703ad295c249183528ec8495816fe95c6ac172`.
Their file SHA-256 values are
`12fd9e5daeeddcc05194311068d8a700489f6542466c1c643c6ddcc4cbb9d543` and
`d07a93c33ddc64eac7e2a50901b4dde043ab1bacaa83ea0d95fdf24e6231824e`.
The README SHA-256 is
`ad6e1c6e0c344a97faaa111841b27e2f10a73b88fcd9ecb573b472c1d1a78a49`.

After exact replay, all remaining OpenCoder workers were stopped and the
286,437,437-byte local acquisition cache was deleted. The object was never
uploaded to `Godlydonuts/Sai`, remains recoverable from the exact upstream
revision, and had no active file handle or admitted derivative. The audit
population and evidence remain intact. All three published files were
force-downloaded and replayed byte-for-byte from dataset commit
[`f6151be578e7e353af45152426a55681a27eae80`](https://huggingface.co/datasets/Godlydonuts/Sai/commit/f6151be578e7e353af45152426a55681a27eae80).
