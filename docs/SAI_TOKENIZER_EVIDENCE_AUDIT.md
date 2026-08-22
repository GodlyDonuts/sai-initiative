# Sai Tokenizer Evidence Audit

Status: the 32K, 48K, and 64K candidates are mechanically qualified. No
tokenizer capacity has won a capability comparison, and no production
tokenizer is selected.

## What the completed tournament established

The immutable Newton report
`tokenizer_tournament_be505b6_r1.json` has file SHA-256
`38444b7748532fa083e0ef18acb99c95ea79ed47877c4943c0a9cf15a0a20c54`
and canonical report SHA-256
`a393dceca2f21fb23b0741e8ab402ccca36a647b7ddb84160d53723d73f61f2b`.
All three candidates round-trip without unknown or empty encodings and preserve
the special-token contract.

The measured corpus contained `459,376` documents and `1,789,679,563` UTF-8
bytes, all labeled `english`. Its exact aggregate results were:

| Candidate | Tokens / 1K UTF-8 bytes | UTF-8 bytes / token | Protected tokens / 1K bytes |
|---|---:|---:|---:|
| 32K | 219.184099 | 4.562375 | 604.782882 |
| 48K | 213.812936 | 4.676985 | 568.281938 |
| 64K | 210.980267 | 4.739780 | 550.660793 |

These results prove the expected compression/parameter tradeoff. They do not
prove that 48K—or any other capacity—produces the strongest model. The
protected suite contains code, math, science, technical strings, identifiers,
numbers, and arbitrary Unicode, but it is a losslessness probe rather than a
representative broad-domain population.

## Correct interpretation of the 48K receipt

The 48K tree identity
`cf4879ee5b3914b4af187abcc93be5678e41ff942e0b0a14f6eeb1a089f6f76d`
is a qualified fixed-geometry default. The selection function hardcodes 48K
after all three candidates qualify. Its receipt is not an empirical winner
receipt and must not be cited as one.

At width 2,560, tied embeddings contain 81.92M parameters for 32K, 122.88M for
48K, and 163.84M for 64K. Moving 32K→48K spends 40.96M parameters for a 2.45%
reduction in corpus tokens per byte; moving 48K→64K spends another 40.96M for a
further 1.32% reduction. Capability evidence is required to decide whether
either exchange is worthwhile.

## Required selection evidence

`sai-audit-tokenizer-evidence` now replays the tournament arithmetic and emits a
create-only evidence audit. It deliberately leaves `empirical_winner=null` and
`production_tokenizer_selected=false`.

Before selecting a production tokenizer, Sai requires:

1. a representative English/code/math/science/technical tournament population;
2. identical tokenizer-training records and special-token contracts;
3. matched small-model initialization and body geometry;
4. both equal-admitted-UTF-8-byte and equal-FLOP training contrasts;
5. source-disjoint held-out likelihood measured per domain;
6. source-disjoint real benchmark capability and retention; and
7. a paired decision that treats vocabulary parameters and inference tokens as
   explicit costs.

Losslessness is admission evidence. Fertility is efficiency evidence. Neither
is capability evidence, and neither authorizes 4B training.
