# Sai 4B Benchmark-First Contract

Status: prospective. This document authorizes no GPU work by itself.

## Comparators and changed factors

The external reference is
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It is not the
presumed Sai parent. The primary equal-compute control is the conventional
architecture from the frozen tournament at the same parameter class. Its
iso-data comparison uses the same admitted bytes, data order, sequence
curriculum, updates, and seeds while reporting compute. Its iso-FLOP comparison
uses the same ordered data stream and compute budget but may consume a
different-length prefix. The Sai candidate differs only by the factor set that
won the 100M, 300M, and 1B promotion ladder.

Candidate and control remain single-pass causal language models. Neither
receives a draft, verifier result, benchmark label, or second model call at
inference. Qwen uses its frozen native prompt and decoding contract, with any
cross-model presentation difference declared before results are opened.

## Reasoning curriculum

Only after a base-model architecture wins may an SFT stage combine verified
cold-start reasoning traces with direct
answers and broad parent-behavior replay. Long traces are generated in groups
by a stronger teacher and retained only after rule-based answer or execution
verification. The model is trained to emit a final answer even after long
deliberation and to use a short path when extended reasoning is unnecessary.

An RL stage is not automatic. If SFT passes the public gate, a bounded GRPO
candidate may optimize rule-verifiable math, code, and logic outcomes. It must
retain the same replay objective and face a matched SFT-only control.

## Tokenizer candidate

The architecture tournament compares 64K, 48K, and 32K tokenizers; 16K is a
diagnostic only. Every candidate preserves byte fallback, special tokens,
ASCII, English/Latin, code, whitespace/indentation, identifiers, URLs, numbers,
units, Greek and math symbols, LaTeX, and technical/scientific notation.

Tokenizer-only tests retain body geometry. Parameter-reallocation tests are a
separate contrast that reinvests saved vocabulary parameters into depth or FFN
capacity while matching total parameters. Iso-data tokenizer budgets use
admitted UTF-8 bytes; iso-FLOP budgets use the same ordered stream and compute
budget but may consume a different-length prefix. Validation loss is normalized
per source byte.

## Public decision

The complete official HumanEval+, MBPP+, IFEval, MuSR, and CorrectBench boards
must use identical model-visible prompts and decoding across the unchanged
parent, equal-compute control, and candidate. Each score binds benchmark source,
ordered identities, prompt/decoding contracts, and checkpoint hashes.

Promotion requires every condition:

1. candidate macro is at least `1.0` point above the original;
2. candidate macro is at least `1.0` point above equal compute;
3. no benchmark is more than `1.0` point below either comparator;
4. candidate beats each comparator on at least four of five benchmarks;
5. MuSR is nonnegative against both comparators; and
6. CorrectBench is nonnegative against both comparators.

A pass authorizes broader confirmation. It does not lock the architecture or
authorize a release claim.

## Historical falsification

The predecessor always-revise system scored `42.806%` macro, versus `54.022%`
for the original and `49.911%` for equal compute. MuSR regressed `33.201` points
and CorrectBench `20.839` points against the original. The executable gate must
reject those exact results, which is covered by the test suite.
