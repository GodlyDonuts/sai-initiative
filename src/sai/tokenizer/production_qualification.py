"""Qualify Sai's fixed-geometry production 48K tokenizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_tree
from sai.tokenizer.build import SCHEMA as BUILD_SCHEMA
from sai.tokenizer.qualification import (
    _load_corpus_texts,
    _load_protected_suite,
    _measure_texts,
    _source_receipts,
    _vocabulary,
    declares_byte_fallback,
)

SCHEMA = "sai-1b-production-tokenizer-qualification-v1"


class ProductionTokenizerQualificationError(RuntimeError):
    """The fixed 48K build, corpus, or lossless qualification differs."""


def qualify_production(
    tokenizer: Any,
    tokenizer_root: Path,
    build_manifest_path: Path,
    corpora: list[Path],
    protected_suite: Path,
) -> dict[str, Any]:
    """Measure exact 48K losslessness and fertility on its training population."""

    try:
        build = json.loads(build_manifest_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionTokenizerQualificationError(
            "tokenizer build differs"
        ) from error
    unsigned = {key: value for key, value in build.items() if key != "manifest_sha256"}
    candidate = build.get("candidates", {}).get("48k")
    identity = sha256_tree(tokenizer_root)
    if (
        build.get("schema") != BUILD_SCHEMA
        or build.get("manifest_sha256") != canonical_sha256(unsigned)
        or not isinstance(candidate, dict)
        or candidate.get("vocab_size") != 48_000
        or candidate.get("tree_sha256") != identity
    ):
        raise ProductionTokenizerQualificationError("tokenizer build differs")
    sources = _source_receipts(corpora)
    if sources != build.get("source_receipts"):
        raise ProductionTokenizerQualificationError("tokenizer source corpus differs")
    corpus_rows, corpus_domains = _load_corpus_texts(corpora)
    protected_rows, protected_receipt = _load_protected_suite(protected_suite)
    vocabulary, specials = _vocabulary(tokenizer, 48_000)
    corpus = _measure_texts(tokenizer, vocabulary, corpus_rows)
    protected = _measure_texts(
        tokenizer,
        vocabulary,
        [(category, text) for _identity, category, text in protected_rows],
    )
    byte_fallback = declares_byte_fallback(tokenizer_root)
    qualified = byte_fallback and all(
        metrics[field] == 0
        for metrics in (corpus, protected)
        for field in ("roundtrip_failures", "unknown_tokens", "empty_encodings")
    )
    payload = {
        "schema": SCHEMA,
        "status": "qualified_production_48k" if qualified else "rejected_48k",
        "build_manifest_sha256": build["manifest_sha256"],
        "tokenizer_identity_sha256": identity,
        "vocab_size": 48_000,
        "source_receipts": sources,
        "corpus_identity_sha256": canonical_sha256(
            {
                "sources": sources,
                "domain_documents": dict(sorted(corpus_domains.items())),
                "protected_suite": protected_receipt,
            }
        ),
        "protected_suite_receipt": protected_receipt,
        "special_token_contract_sha256": canonical_sha256(specials),
        "byte_fallback": byte_fallback,
        "corpus": corpus,
        "protected_suite": protected,
        "fixed_geometry_selection": "48k",
        "empirical_capacity_winner_claimed": False,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    if not qualified:
        raise ProductionTokenizerQualificationError("production tokenizer rejected")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--protected-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ProductionTokenizerQualificationError("qualification output exists")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ProductionTokenizerQualificationError(
            "transformers is required"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ProductionTokenizerQualificationError("fast tokenizer is required")
    tokenizer.byte_fallback = declares_byte_fallback(args.tokenizer_root)
    result = qualify_production(
        tokenizer,
        args.tokenizer_root,
        args.build_manifest,
        args.corpus,
        args.protected_suite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
