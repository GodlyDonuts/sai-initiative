"""Seal exact practical-corpus readiness after local and remote publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_practical_admission import SCHEMA as BOOKS_SCHEMA
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_SCHEMA
from sai.data.practical_hf_publish import METADATA_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-practical-corpus-readiness-audit-v1"


class PracticalCorpusAuditError(RuntimeError):
    """An admission, publication, or corpus accounting invariant differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PracticalCorpusAuditError("signed audit input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PracticalCorpusAuditError("signed audit input is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise PracticalCorpusAuditError("signed audit input differs")
    return payload


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PracticalCorpusAuditError(f"{label} differs")
    return value


def build_audit(
    books_path: Path,
    pleias_path: Path,
    publication_path: Path,
    output: Path,
    minimum_combined_text_bytes: int,
    maximum_combined_text_bytes: int,
) -> dict[str, Any]:
    """Verify the exact practical training corpus and its published custody."""

    if (
        output.exists()
        or output.is_symlink()
        or minimum_combined_text_bytes <= 0
        or maximum_combined_text_bytes < minimum_combined_text_bytes
    ):
        raise PracticalCorpusAuditError("audit arguments differ")
    books = _load_signed(books_path, BOOKS_SCHEMA)
    pleias = _load_signed(pleias_path, PLEIAS_SCHEMA)
    publication = _load_signed(publication_path, METADATA_SCHEMA)
    if (
        books.get("status") != "complete_practical_private_pretraining_admission"
        or books.get("practical_pretraining_ready") is not True
        or books.get("training_ready") is not True
        or books.get("semantic_model_review_required") is not False
        or books.get("official_benchmark_decontamination_complete") is not False
        or pleias.get("status") != "complete_practical_pleias_pretraining_admission"
        or pleias.get("practical_pretraining_ready") is not True
        or pleias.get("training_ready") is not True
        or pleias.get("global_exact_content_deduplication_complete") is not True
        or pleias.get("official_benchmark_decontamination_complete") is not False
        or publication.get("status") != "complete_practical_hf_metadata_publication"
        or publication.get("books_admission_receipt_sha256") != books["receipt_sha256"]
        or publication.get("pleias_admission_receipt_sha256")
        != pleias["receipt_sha256"]
        or publication.get("source_text_uploaded") is not False
    ):
        raise PracticalCorpusAuditError("practical corpus evidence differs")
    book_counts = books.get("counts", {})
    pleias_counts = pleias.get("counts", {})
    book_rows = _count(book_counts.get("admitted_rows"), "book rows")
    book_bytes = _count(book_counts.get("admitted_text_utf8_bytes"), "book text bytes")
    book_tokens = _count(book_counts.get("admitted_enriched_tokens"), "book tokens")
    pleias_rows = _count(pleias_counts.get("admitted_rows"), "PleIAs rows")
    pleias_bytes = _count(
        pleias_counts.get("admitted_text_utf8_bytes"), "PleIAs text bytes"
    )
    pleias_tokens = _count(
        pleias_counts.get("admitted_source_token_count"), "PleIAs tokens"
    )
    combined_bytes = book_bytes + pleias_bytes
    combined_tokens = book_tokens + pleias_tokens
    collections = pleias_counts.get("collections")
    rights = pleias_counts.get("rights")
    if (
        not isinstance(collections, dict)
        or not collections
        or sum(_count(value, "collection rows") for value in collections.values())
        != pleias_rows
        or pleias_counts.get("admitted_collection_count") != len(collections)
        or not isinstance(rights, dict)
        or not rights
        or sum(_count(value, "rights rows") for value in rights.values()) != pleias_rows
        or pleias_counts.get("combined_books_plus_pleias_text_utf8_bytes")
        != combined_bytes
        or not minimum_combined_text_bytes
        <= combined_bytes
        <= maximum_combined_text_bytes
    ):
        raise PracticalCorpusAuditError("practical corpus totals differ")
    payload = {
        "schema": SCHEMA,
        "status": "complete_practical_training_corpus_readiness_audit",
        "inputs": {
            "books_file_sha256": sha256_file(books_path),
            "books_receipt_sha256": books["receipt_sha256"],
            "pleias_file_sha256": sha256_file(pleias_path),
            "pleias_receipt_sha256": pleias["receipt_sha256"],
            "publication_file_sha256": sha256_file(publication_path),
            "publication_receipt_sha256": publication["receipt_sha256"],
        },
        "bounds": {
            "minimum_combined_text_utf8_bytes": minimum_combined_text_bytes,
            "maximum_combined_text_utf8_bytes": maximum_combined_text_bytes,
            "combined_byte_bound_satisfied": True,
        },
        "components": {
            "institutional_books": {
                "rows": book_rows,
                "text_utf8_bytes": book_bytes,
                "source_token_count": book_tokens,
            },
            "pleias_common_corpus": {
                "rows": pleias_rows,
                "text_utf8_bytes": pleias_bytes,
                "source_token_count": pleias_tokens,
                "collections": len(collections),
            },
        },
        "totals": {
            "source_components": 2,
            "rows": book_rows + pleias_rows,
            "text_utf8_bytes": combined_bytes,
            "source_token_count": combined_tokens,
        },
        "quality": {
            "english_only": True,
            "mechanical_non_slop_gate_complete": True,
            "explicit_reusable_rights_complete": True,
            "global_exact_content_deduplication_complete": True,
            "semantic_model_review_required_for_bulk_core": False,
        },
        "custody": {
            "huggingface_repository": publication["remote_repository"],
            "locator_and_manifest_publication_complete": True,
            "private_book_text_uploaded": False,
            "pleias_reconstructable_from_pinned_upstream": True,
        },
        "official_benchmark_decontamination_complete": False,
        "evaluation_claims_allowed": False,
        "practical_training_corpus_ready": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-receipt", type=Path, required=True)
    parser.add_argument("--pleias-receipt", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-combined-text-bytes", type=int, required=True)
    parser.add_argument("--maximum-combined-text-bytes", type=int, required=True)
    args = parser.parse_args()
    result = build_audit(
        args.books_receipt,
        args.pleias_receipt,
        args.publication_receipt,
        args.output,
        args.minimum_combined_text_bytes,
        args.maximum_combined_text_bytes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
