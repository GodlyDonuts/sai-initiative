# Sai FineMath 4+ Shard Audit — 2026-08-22

## Immutable input

This audit inspected one bounded, deterministic shard before any Sai download
or optimizer job was authorized on FineMath:

- repository: `HuggingFaceTB/finemath`;
- repository revision: `e92b25a616738fe95dc186b64dfb19f9c8525594`;
- file: `finemath-4plus/train-00000-of-00064.parquet`;
- bytes: `286,267,316`;
- file SHA-256:
  `5d0b26114b4cf309c82a4ba8f6f45857480b27812e23a46b61b45e8bbed61fe5`;
- rows: `104,680`.

The file hash exactly matched the repository's large-file object identity. The
audit was read-only and local; it consumed no Newton quota and launched no
training or cluster job.

## Population evidence

- `94,692` rows carry upstream integer score `4`; `9,988` carry score `5`.
- `36,385` rows (`34.7583%`) set the included extraction metadata field
  `found_math=true`; the remaining `68,295` do not.
- All `104,680` text SHA-256 values are unique within this shard. This is useful
  exact-duplicate evidence but says nothing about cross-shard or semantic near
  duplication.
- Character-count quantiles p10/p50/p90/p99 are
  `872 / 2,908 / 9,813 / 37,042`.
- A literal lower-bound scan found `170` essay-service, `652` casino/betting,
  `1,126` SEO-marketing, `1,672` answer-key/homework-site, and `3,580`
  under-80-word documents. These keyword counts overlap, are not a classifier,
  and must not be reported as total bad-row prevalence.
- Leading domains include `physicsforums.com` (`4,417`),
  `math.stackexchange.com` (`4,404`), `mathhelpforum.com` (`4,244`),
  `jiskha.com` (`3,035`), `coursehero.com` (`2,081`), `socratic.org`
  (`2,078`), `gmatclub.com` (`1,993`), and `gradesaver.com` (`1,617`).
  There are also `1,329` rows with an empty parsed hostname.

## Qualitative evidence

The first physical row is upstream score `5` but contains incoherent and
incorrect elementary-algebra prose plus essay-service links. The next row is a
coherent Singapore Math Grade 3 scope-and-sequence document that explicitly
states prior knowledge. This adjacent contrast is direct evidence that the
upstream score alone cannot be Sai's quality or prerequisite gate.

A deterministic twenty-row sample was selected by the lowest
`SHA256(b"sai-finemath-review-v1" || SHA256(text))` values. It repeatedly
contains:

- scraped homework questions and conversational fragments without a reliable
  final explanation;
- unit-conversion pages instantiated with arbitrary values;
- answer-key, quiz, course-resource, and commercial worksheet boilerplate;
- topic-index and search-result pages rather than complete teaching documents;
- malformed math extraction and punctuation;
- valid explanatory or worked mathematical content mixed with the above.

The sample is not a prevalence estimate. It is sufficient to prove that the
shard contains failure modes which upstream `4plus` admission does not remove.

## Decision

FineMath 4+ is a **raw mathematics candidate pool**, not a qualified Sai source.
No unfiltered FineMath row may enter the curriculum or a training stream.

Before a matched source-addition screen, Sai must:

1. manifest and audit every selected shard rather than extrapolate from this
   one shard;
2. preserve URL, crawl, extraction, score, and math-evidence metadata;
3. apply explicit answer-farm, commercial-homework, essay-service, gambling,
   SEO, navigation/index, malformed-extraction, and empty-host policies;
4. require self-contained mathematical exposition or verified worked structure,
   not merely math keywords or a classifier score;
5. globally exact- and near-deduplicate against all Sai sources;
6. decontaminate against every development and terminal benchmark, including
   problem statements and solution variants;
7. build a deterministic accepted/rejected review packet and independently
   inspect it before freezing thresholds;
8. compare the filtered source addition against an equal-token web-only control
   with architecture, tokenizer, order, seed, optimizer, and compute fixed.

This is a rejection of blind admission, not of the entire candidate. A filtered
subset may still be valuable, but it must earn that conclusion empirically.

