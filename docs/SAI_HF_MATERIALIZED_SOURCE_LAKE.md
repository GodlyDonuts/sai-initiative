# Sai Hugging Face materialized source lake

## Result

Sai crossed the exact 8 TiB physical-source target at immutable Hugging Face
dataset head `cc8576fbb3f949bdaf59049a150c1fa1d35f47c3`:

- 13,974 LFS source objects;
- 8,802,247,613,960 bytes;
- 8.00559756859002 TiB;
- 6,154,591,752 bytes above the exact 8 TiB boundary; and
- zero destination-versus-upstream size or SHA-256 mismatches.

The file-level manifest is
`artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl`. It has
13,974 rows, 10,353,483 bytes, and SHA-256
`56dc0d512db07aa26533decce8efd86bcb1b705355cd40069fdf9cb6311b5665`.
The aggregate receipt has canonical receipt SHA-256
`0715eefc3c3bda8ee800fc4c80155df461055da3bbf2a473ad6c93cf93bea9d8`.

## Exact boundary

The materialized lake contains complete selected snapshots of FineMath-4plus,
three Nemotron specialist datasets, UltraData-Math L1, 31 Common Pile source
families, and the 10,000 data shards selected from PleIAs Common Corpus. It also
contains the first 1,250 path-ordered FinePDFs shards. Hugging Face accepted
those five 250-file FinePDFs copy batches, then rejected the next commit because
the account had exceeded its public-storage allowance. The failed request did
not create a partial sixth batch.

The frozen head therefore records the exact last successful transaction. Sai
does not describe the 1,250-file FinePDFs subset as the complete 3,573-file
upstream snapshot. The source lake already exceeds the physical-byte target, so
the missing 2,323 FinePDFs files do not invalidate the 8 TiB custody result.

## Why Stokes bandwidth does not change this boundary

Stokes has a 10 Gbps network path and remains valuable for source transforms,
Parquet scans, deduplication, and output publication. The large copies completed
here used Hugging Face's server-side cross-repository LFS copy operation. That
avoids downloading and re-uploading source objects and is therefore preferable
even to a fast Stokes link. The stopping condition was Hugging Face account
storage, not transfer bandwidth. Stokes also does not currently have enough
free quota to stage the remaining multi-terabyte FinePDFs snapshot locally.

## Replay

With an authorized `HF_TOKEN` in the environment, the exact remote replay is:

```bash
sai-freeze-hf-materialized-source-lake \
  --destination-revision cc8576fbb3f949bdaf59049a150c1fa1d35f47c3 \
  --output-root artifacts/sai_hf_materialized_source_lake_REPLAY
```

The command resolves every component's source-publication manifest, checks its
canonical receipt, resolves the pinned upstream revision, and compares every
source and destination LFS object by byte count and SHA-256 before accounting
it. FinePDFs is compared directly to its pinned upstream revision because the
quota boundary prevented publishing a component manifest after batch five.

## Admission boundary

The materialized lake is deliberately marked `training_ready=false`. Physical
custody does not establish:

- source-wide or per-row rights clearance;
- complete English translation routing;
- global exact or semantic deduplication;
- benchmark-disjoint status;
- full-population Hermes quality and prerequisite judgment;
- accepted representation selection;
- tokenizer-level token counts;
- spiral-curriculum assignment; or
- authorization to start the 4B run.

The source lake solves availability and immutable identity. The data compiler
must still solve admission, transformation, and pedagogy.
