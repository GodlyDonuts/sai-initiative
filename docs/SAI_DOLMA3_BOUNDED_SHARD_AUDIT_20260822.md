# Sai Dolma 3 Bounded Shard Audit — 2026-08-22

Status: four exact compressed-shard diagnostics completed. No row is admitted
for tokenizer training, pretraining, mid-training, or any other optimizer use.

## Boundary

The parent no-download inventory binds
`allenai/dolma3_mix-150B-1025` at revision
`afa92bfb22366821c5e6cd427cdd036b34b713ef`. It contains 6,081 data shards and
110,586,325,507 compressed bytes. This audit selected the smallest compressed
shard from each of four relevant component partitions so inspection remained
bounded and used little temporary storage.

Selecting the smallest shard can overrepresent tail fragments and is not a
random or population-representative sample. The observed proportions must not
be extrapolated to the full 150B mix. They are sufficient to disprove an
assumption that every row in the wrapper is unique, nonempty, or licensed.

`sai-audit-hf-shard` verifies the compressed size and SHA-256 against the exact
inventory, streams Zstandard JSONL, binds ordered document/text identities,
measures physical multiplicity, empty rows, source identities, metadata keys,
integer quality scores, and per-row `license_type`, and emits no text content.

## Exact results

| Component shard | Rows | Unique IDs/texts | Duplicate-ID rows | Empty text | License metadata | Receipt SHA-256 |
|---|---:|---:|---:|---:|---|---|
| Common Crawl science/math/technology `0019/1597` | 19,750 | 2,420 | 17,330 (87.7468%) | 0 | missing on all rows | `d20290492e6b21bd948f3e78a08dd1e9d8a00a631203dd22ad71dc952bfb3fdd` |
| olmOCR science PDF `575` | 28 | 26 | 2 (7.1429%) | 0 | missing on all rows | `00e024ff507a59e707e722ffbd0f6b616cd90e1f9df6378ab79b4afd4d787b77` |
| FineMath 3+ `37` | 19,342 | 4,340 | 15,002 (77.5618%) | 5 | missing on all rows | `b349edc189e81839a9c0cc7f9f8551f71d42aeba4a83e9391269da19ec7cec9b` |
| Stack-Edu Python `0` | 1,201 | 269 | 932 (77.6020%) | 0 | 960 `no_license`; 241 `permissive` | `aa274c62b0bd0e6077d26c6e81d20e7f75908d05752e15034cb7f7271fb3fb5a` |

The exact compressed member SHA-256 values are respectively
`af60f8c4454a551909af4c49a9d6856ac5e3cb347d458cd4711b89fcf8e1dbc6`,
`0eb5faf0ffd3fcf25e719cdd5ffe2f337be1478421a632b83b9f096bc0509bc1`,
`0a50b0fdcd57d3036cac65b18b6dac918ab24e1b7e239fd875d14bb4c0144001`,
and `d0661f3e81d913329c9b5927359a4e3f5303da7e70f2e57077e679205352ca7f`.

Document multiplicities are highly structured rather than accidental-looking:

- Common Crawl unique IDs appear predominantly eight or nine times.
- FineMath unique IDs appear four or five times.
- Stack-Edu unique IDs appear four or five times.
- The two duplicated olmOCR IDs appear twice.

This is consistent with physical row repetition being used to represent mixture
weights, but the audit does not claim the upstream intent. It establishes only
the exact observed bytes.

## Sai consequences

1. A wrapper dataset row count is not a unique-data count. Sai first constructs
   canonical unique document/text identities and reports multiplicity.
2. Deduplication and sampling weight are separate operations. If repeated
   exposure helps, Sai represents it as an explicit curriculum/mixture weight
   over one canonical document rather than silently multiplying source
   diversity.
3. The 150B sample must not train Sai's tokenizer directly. Physical repeats
   would bias BPE merge frequencies, and the wrapper does not provide broad
   per-row license evidence.
4. Dataset-level ODC-By metadata does not convert a `no_license` code file into
   an admitted source. Every Stack-Edu member still needs the existing
   per-file allowlist, attribution/removal metadata, opt-out replay, secret
   scan, deduplication, and benchmark-decontamination gates.
5. Empty FineMath rows are rejected before normalization. Its upstream score
   remains a candidate signal rather than a Sai quality decision.
6. olmOCR extraction quality, document type, public-document/PII metadata,
   source URL, and license require review before science PDFs can become a
   source class.
7. A representative stratified audit must sample prospectively by component,
   source identity, quality band, and document length before any source-addition
   experiment is materialized.

The modern corpus remains useful as a source pool. It is not a ready-made Sai
mixture.

## Prospective stratified follow-up

The next content audit is now selected before content inspection by
`SAI_DOLMA3_STRATIFIED_AUDIT_SPEC.json`. Its canonical specification SHA-256 is
`bbc2fb37c2caf90f599231802d0762038330f05641dc6d9ba37bb52703a69299`.
`sai-plan-hf-stratified-audit` replayed the exact parent inventory and selected
110 unique shards by SHA-256 rank, never by compressed size or observed
content. The plan receipt is
`37818ae0d46a1989798945ea5c70679dcaffb898df13a69d58e51bcf7c084319`;
the local plan-file SHA-256 is
`9665d2792611e27baa96b909d9313ffa213fb35fd42948cb65335e6ee4ca9a63`.

The 110 members cover 21 low-band, 24 middle-band, and 24 high-band Common
Crawl topics; 23 science-PDF topics; 15 Stack-Edu languages; and one member
each from FineMath, the arXiv partition, and the English encyclopedia
partition. Their total compressed size is 2,583,644,891 bytes. This is a
selection plan only: no member was downloaded, no content was admitted, and
no training was authorized. Content inspection will be streamed or performed
only after current run artifacts release enough storage headroom.
