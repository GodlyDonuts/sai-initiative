# Sai FineMath-4plus full-corpus census

## Measured result

The exact `HuggingFaceTB/finemath` revision
`e92b25a616738fe95dc186b64dfb19f9c8525594`, subset `finemath-4plus`, contains:

- 64 Parquet shards and 18,365,184,633 compressed bytes;
- 6,699,493 rows;
- 34,126,971,204 UTF-8 text bytes;
- 9,573,187,002 upstream-tokenizer tokens; and
- 33,692,440,601 declared characters, with zero declared-versus-observed
  character-count mismatches.

Every source file was replayed against the pinned manifest SHA-256 before its
rows were scanned. Slurm array `817885` ran one no-requeue, one-CPU job per
source shard. All 64 tasks exited `0:0`. Dependency job `817886` merged the
sorted digest streams and produced the aggregate.

## Duplicate evidence

Every row emitted a byte-exact SHA-256 and an NFKC/casefold/whitespace-normalized
SHA-256. The sorted 32-byte digest streams were globally merged:

| Identity | Rows | Unique | Keep-first duplicate rows | Duplicate groups | Maximum multiplicity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Byte exact | 6,699,493 | 6,699,493 | 0 | 0 | 1 |
| Normalized | 6,699,493 | 6,699,486 | 7 | 7 | 2 |

This proves exact and normalization-equivalent multiplicity only. It does not
prove semantic or subdocument uniqueness.

## Language and length evidence

All rows carry upstream language label `en`, but confidence varies:

| Language confidence | Rows |
| --- | ---: |
| `<0.50` | 349,840 |
| `0.50–0.70` | 591,870 |
| `0.70–0.80` | 778,508 |
| `0.80–0.90` | 2,351,777 |
| `0.90–0.95` | 2,076,018 |
| `≥0.95` | 551,480 |

The upstream token-length distribution is:

| Tokens | Rows |
| --- | ---: |
| `<64` | 7,152 |
| `64–127` | 92,267 |
| `128–511` | 1,647,292 |
| `512–2,047` | 3,924,011 |
| `2,048–8,191` | 918,400 |
| `8,192–32,767` | 103,772 |
| `≥32,768` | 6,599 |

## Quality interpretation

FineMath-4plus is not a homogeneous block of pristine mathematical exposition.
Its provider score is either 4 (6,054,653 rows) or 5 (644,840 rows), but only
2,337,752 rows set `found_math=true`, and 2,339,612 rows expose any nonzero
math-extraction feature. The corpus includes weak-language-confidence and
commercial-web material even after the upstream math-quality filter.

The broad, core, and elite profiles in the README are therefore measurement
counterfactuals, not acceptance rules. Selecting a threshold requires real
downstream proxy evidence and must remain separable from benchmark
decontamination, source obligations, semantic deduplication, Hermès judgments,
and curriculum coverage.

## Evidence

The aggregate canonical receipt is
`4d40be3e16fe47476c47195a66498a689cf21784737366c4bba0c74658baa25c`.
The source-safe publication receipt is
`bb578f5e969e8d15d96ae40ae3511d4dd6d2d9c42e834e5c641204719d53e4c2`.
The publication file SHA-256 is
`af4d4e017a5efab86aa12c9251b347b6a8aa0e9587641d4616a004ff3d866484`.
Hugging Face evidence head
`db79c6bb4e7752aee2de8ce2414fcf5ef709e5c1` contains the aggregate,
publication, and all 64 shard receipts with zero replay mismatches.

No source text is included in the evidence publication, and no row is yet
training-ready.
