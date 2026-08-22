"""Build and measure a matched Sai numeric pre-tokenization ablation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_tree
from sai.tokenizer.build import SPECIAL_TOKENS, _source_receipts, _texts
from sai.tokenizer.qualification import (
    _load_corpus_texts,
    _load_protected_suite,
    _measure_texts,
    _vocabulary,
    declares_byte_fallback,
)

SCHEMA = "sai-numeric-pretokenization-ablation-v1"
PRODUCTION_VOCAB_SIZE = 48_000
MODES = ("individual_digits", "digit_runs")


class NumericPretokenizationError(RuntimeError):
    """The matched tokenizer source, build, or measurement differs."""


def build_numeric_ablation(
    corpora: list[Path],
    protected_suite: Path,
    output_root: Path,
    *,
    vocab_size: int = PRODUCTION_VOCAB_SIZE,
) -> dict[str, Any]:
    """Build two otherwise identical BPEs and measure exact fertility."""

    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 261
        or output_root.exists()
    ):
        raise NumericPretokenizationError("numeric tokenizer geometry differs")
    sources = _source_receipts(tuple(corpora))
    corpus_rows, corpus_domains = _load_corpus_texts(corpora)
    protected_rows, protected_receipt = _load_protected_suite(protected_suite)
    protected_texts = [(category, text) for _, category, text in protected_rows]
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise NumericPretokenizationError(
            "tokenizers and transformers are required for the ablation"
        ) from error

    stage = output_root.with_name(f".{output_root.name}.partial.{os.getpid()}")
    if stage.exists():
        raise NumericPretokenizationError("numeric tokenizer stage already exists")
    stage.mkdir(parents=True)
    candidates: dict[str, Any] = {}
    try:
        for mode in MODES:
            tokenizer = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Digits(
                        individual_digits=mode == "individual_digits"
                    ),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
                ]
            )
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=list(SPECIAL_TOKENS),
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False,
            )
            tokenizer.train_from_iterator(_texts(tuple(corpora)), trainer=trainer)
            if tokenizer.get_vocab_size() != vocab_size:
                raise NumericPretokenizationError(
                    "numeric tokenizer vocab is incomplete"
                )
            root = stage / mode
            wrapped = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token=SPECIAL_TOKENS[0],
                bos_token=SPECIAL_TOKENS[1],
                eos_token=SPECIAL_TOKENS[2],
                clean_up_tokenization_spaces=False,
            )
            wrapped.save_pretrained(root)
            vocabulary_ids, specials = _vocabulary(wrapped, vocab_size)
            corpus_metrics = _measure_texts(wrapped, vocabulary_ids, corpus_rows)
            protected_metrics = _measure_texts(wrapped, vocabulary_ids, protected_texts)
            qualified = declares_byte_fallback(root) and all(
                metrics[field] == 0
                for metrics in (corpus_metrics, protected_metrics)
                for field in (
                    "roundtrip_failures",
                    "unknown_tokens",
                    "empty_encodings",
                )
            )
            candidates[mode] = {
                "mode": mode,
                "vocab_size": vocab_size,
                "root": mode,
                "tree_sha256": sha256_tree(root),
                "special_token_contract_sha256": canonical_sha256(specials),
                "byte_fallback": declares_byte_fallback(root),
                "corpus": corpus_metrics,
                "protected_suite": protected_metrics,
                "qualified": qualified,
            }
        individual = candidates["individual_digits"]
        runs = candidates["digit_runs"]
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": (
                "mechanically_qualified_capability_selection_pending"
                if all(row["qualified"] for row in candidates.values())
                else "candidate_rejected"
            ),
            "training_authorized": False,
            "production_tokenizer_selected": False,
            "vocab_size": vocab_size,
            "production_geometry": vocab_size == PRODUCTION_VOCAB_SIZE,
            "special_tokens": list(SPECIAL_TOKENS),
            "source_receipts": sources,
            "source_identity_sha256": canonical_sha256(sources),
            "domain_documents": dict(sorted(corpus_domains.items())),
            "protected_suite_receipt": protected_receipt,
            "candidates": candidates,
            "digit_runs_minus_individual": {
                "corpus_tokens": runs["corpus"]["tokens"]
                - individual["corpus"]["tokens"],
                "corpus_tokens_percent": 100.0
                * (runs["corpus"]["tokens"] / individual["corpus"]["tokens"] - 1.0),
                "protected_tokens": runs["protected_suite"]["tokens"]
                - individual["protected_suite"]["tokens"],
                "protected_tokens_percent": 100.0
                * (
                    runs["protected_suite"]["tokens"]
                    / individual["protected_suite"]["tokens"]
                    - 1.0
                ),
                "numbers_and_units_tokens_per_1k_utf8_bytes": {
                    mode: candidates[mode]["protected_suite"]["by_domain"][
                        "numbers_and_units"
                    ]["tokens_per_1k_utf8_bytes"]
                    for mode in MODES
                },
                "math_tokens_per_1k_utf8_bytes": {
                    mode: candidates[mode]["protected_suite"]["by_domain"]["math"][
                        "tokens_per_1k_utf8_bytes"
                    ]
                    for mode in MODES
                },
                "science_notation_tokens_per_1k_utf8_bytes": {
                    mode: candidates[mode]["protected_suite"]["by_domain"][
                        "science_notation"
                    ]["tokens_per_1k_utf8_bytes"]
                    for mode in MODES
                },
            },
            "checks": {
                "same_source_records_and_order": True,
                "same_vocab_size_and_special_tokens": True,
                "only_digit_segmentation_changed": True,
                "losslessness_measured": True,
                "fertility_is_not_capability": True,
                "matched_training_and_real_benchmarks_still_required": True,
                "no_training_or_production_selection": True,
            },
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        (stage / "report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--protected-suite", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=PRODUCTION_VOCAB_SIZE)
    args = parser.parse_args()
    payload = build_numeric_ablation(
        args.corpus,
        args.protected_suite,
        args.output_root,
        vocab_size=args.vocab_size,
    )
    print(
        json.dumps(
            {"receipt_sha256": payload["receipt_sha256"], "status": payload["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
