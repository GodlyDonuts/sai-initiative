# Sai FineMath Human Review Workspace

The FineMath language-score ladder contains 3,114 rows that passed every
non-language V1 quality rule and a blinded 192-row review packet: 64 rows from
each hidden language-score stratum. These rows are candidates, not training
data.

`sai-build-finemath-review-workspace` turns the exact replay-validated packet
into one self-contained offline HTML file. The browser receives only row
identities and text. It does not receive source URLs, language scores, score
strata, the hidden review key, or any threshold candidate. The content-security
policy prohibits external requests.

For every row, a reviewer records:

- accept, reject, or uncertain;
- mathematical correctness;
- explanatory structure;
- whether the text is self-contained;
- English clarity in parts per million;
- declared defects;
- at least one unique literal evidence quote.

An `accept` label is mechanically available only when the row is correct,
explanatory, self-contained, has clarity at least 800,000 ppm, and has no
declared defect. A rejection requires a declared defect. Progress can be
exported and restored without a server or browser persistence.

The exported JSONL is still blind evidence. It does not select a language-score
threshold and does not authorize training. `sai-adjudicate-finemath-review`
requires two complete reviews with distinct reviewer pseudonyms. Only after
both validate does it open the hidden score key.

The decision rule is frozen before labels exist. A score stratum passes only
when the two-reviewer consensus acceptance rate is at least 90% and its Wilson
95% lower confidence bound is at least 80%. The adjudicator selects the lowest
of no floor, 0.90, or 0.95 for which every included stratum passes. If the
highest stratum fails, FineMath is rejected rather than relaxed after the fact.

Any selected candidates still require global near-deduplication, benchmark
decontamination, licensing/provenance replay, and semantic prerequisite
placement before any Sai stream can include them.
