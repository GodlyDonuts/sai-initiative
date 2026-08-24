"""Seal exact post-rewrite foundation bytes under the Sai quality ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as PLEIAS_SCHEMA,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-foundation-corpus-ledger-v1"
DEFAULT_BYTE_CEILING = 2_000_000_000_000


class FoundationCorpusLedgerError(RuntimeError):
    """Component custody, final byte accounting, or quality ceiling differs."""


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FoundationCorpusLedgerError(f"{label} differs")
    return value


def build_ledger(
    book_aggregate_path: Path,
    pleias_aggregate_path: Path,
    output: Path,
    byte_ceiling: int = DEFAULT_BYTE_CEILING,
) -> dict[str, Any]:
    """Bind the exact clean bytes currently admitted to the foundation pool."""

    if (
        output.exists()
        or output.is_symlink()
        or isinstance(byte_ceiling, bool)
        or not isinstance(byte_ceiling, int)
        or byte_ceiling <= 0
    ):
        raise FoundationCorpusLedgerError("ledger arguments differ")
    books = _load_signed(book_aggregate_path, BOOK_SCHEMA)
    pleias = _load_signed(pleias_aggregate_path, PLEIAS_SCHEMA)
    book_bytes = _positive_count(
        books.get("totals", {}).get("output_text_utf8_bytes"), "book bytes"
    )
    pleias_bytes = _positive_count(
        pleias.get("totals", {}).get("output_text_utf8_bytes"), "PleIAs bytes"
    )
    book_documents = _positive_count(
        books.get("totals", {}).get("documents"), "book documents"
    )
    pleias_documents = _positive_count(
        pleias.get("totals", {}).get("documents"), "PleIAs documents"
    )
    train_documents = books.get("totals", {}).get(
        "split::train::documents", 0
    ) + pleias.get("totals", {}).get("split::train::documents", 0)
    development_documents = books.get("totals", {}).get(
        "split::development::documents", 0
    ) + pleias.get("totals", {}).get("split::development::documents", 0)
    train_bytes = books.get("totals", {}).get(
        "split::train::text_utf8_bytes", 0
    ) + pleias.get("totals", {}).get("split::train::text_utf8_bytes", 0)
    development_bytes = books.get("totals", {}).get(
        "split::development::text_utf8_bytes", 0
    ) + pleias.get("totals", {}).get("split::development::text_utf8_bytes", 0)
    if (
        books.get("complete_benchmark_disjoint_book_coverage") is not True
        or books.get("private_storage_only") is not True
        or books.get("huggingface_redistribution_authorized") is not False
        or books.get("benchmark_decontamination_complete") is not True
        or books.get("cross_source_subdocument_deduplication_complete") is not True
        or books.get("token_count_requires_recomputation") is not True
        or books.get("source_disjoint_split_complete") is not True
        or books.get("semantic_quality_metadata_complete") is not True
        or books.get("curriculum_metadata_complete") is not True
        or pleias.get("complete_final_pleias_document_coverage") is not True
        or pleias.get("all_remote_lfs_identities_verified") is not True
        or pleias.get("benchmark_decontamination_complete") is not True
        or pleias.get("cross_source_subdocument_deduplication_complete") is not True
        or pleias.get("token_count_requires_recomputation") is not True
        or pleias.get("source_disjoint_split_complete") is not True
        or pleias.get("semantic_quality_metadata_complete") is not True
        or pleias.get("curriculum_metadata_complete") is not True
    ):
        raise FoundationCorpusLedgerError("component completion differs")
    total_bytes = book_bytes + pleias_bytes
    total_documents = book_documents + pleias_documents
    if total_bytes > byte_ceiling:
        raise FoundationCorpusLedgerError("foundation bytes exceed quality ceiling")
    if (
        train_documents <= 0
        or development_documents <= 0
        or train_documents + development_documents != total_documents
        or train_bytes + development_bytes != total_bytes
    ):
        raise FoundationCorpusLedgerError("source-disjoint split accounting differs")
    components = [
        {
            "component": "institutional_books",
            "custody": "private_storage_only",
            "redistribution_authorized": False,
            "aggregate_path_name": book_aggregate_path.name,
            "aggregate_receipt_sha256": books["receipt_sha256"],
            "documents": book_documents,
            "post_rewrite_text_utf8_bytes": book_bytes,
        },
        {
            "component": "pleias_common_corpus",
            "custody": "verified_huggingface_lfs",
            "redistribution_authorized": True,
            "aggregate_path_name": pleias_aggregate_path.name,
            "aggregate_receipt_sha256": pleias["receipt_sha256"],
            "documents": pleias_documents,
            "post_rewrite_text_utf8_bytes": pleias_bytes,
        },
    ]
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_foundation_corpus_ledger",
        "policy": {
            "byte_ceiling": byte_ceiling,
            "ceiling_unit": "exact_post_rewrite_utf8_text_bytes",
            "ceiling_is_not_a_target": True,
            "padding_for_volume_prohibited": True,
            "quality_over_volume": True,
        },
        "components": components,
        "ordered_components_sha256": canonical_sha256(components),
        "totals": {
            "documents": total_documents,
            "post_rewrite_text_utf8_bytes": total_bytes,
            "remaining_byte_headroom": byte_ceiling - total_bytes,
            "train_documents": train_documents,
            "development_documents": development_documents,
            "train_text_utf8_bytes": train_bytes,
            "development_text_utf8_bytes": development_bytes,
        },
        "byte_ceiling_respected": True,
        "benchmark_decontamination_complete_for_listed_components": True,
        "cross_source_subdocument_deduplication_complete_for_listed_components": True,
        "semantic_quality_metadata_complete_for_listed_components": True,
        "curriculum_metadata_complete_for_listed_components": True,
        "synthetic_bridge_component_admitted": False,
        "final_tokenization_complete": False,
        "source_disjoint_split_complete": True,
        "curriculum_schedule_complete": False,
        "final_corpus_complete": False,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-aggregate", type=Path, required=True)
    parser.add_argument("--pleias-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--byte-ceiling", type=int, default=DEFAULT_BYTE_CEILING)
    args = parser.parse_args()
    result = build_ledger(
        args.book_aggregate,
        args.pleias_aggregate,
        args.output,
        args.byte_ceiling,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
