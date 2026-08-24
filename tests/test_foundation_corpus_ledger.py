import json

import pytest

from sai.data.foundation_corpus_ledger import (
    FoundationCorpusLedgerError,
    build_ledger,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as PLEIAS_SCHEMA,
)
from sai.data.token_stream import canonical_sha256


def _write(path, value):
    value["receipt_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value))


def _components(tmp_path):
    books = tmp_path / "books.json"
    pleias = tmp_path / "pleias.json"
    _write(
        books,
        {
            "schema": BOOK_SCHEMA,
            "totals": {"documents": 3, "output_text_utf8_bytes": 600},
            "complete_benchmark_disjoint_book_coverage": True,
            "private_storage_only": True,
            "huggingface_redistribution_authorized": False,
            "benchmark_decontamination_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "token_count_requires_recomputation": True,
            "training_ready": False,
        },
    )
    _write(
        pleias,
        {
            "schema": PLEIAS_SCHEMA,
            "totals": {"documents": 5, "output_text_utf8_bytes": 1_200},
            "complete_final_pleias_document_coverage": True,
            "all_remote_lfs_identities_verified": True,
            "benchmark_decontamination_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "token_count_requires_recomputation": True,
            "training_ready": False,
        },
    )
    return books, pleias


def test_ledger_uses_exact_post_rewrite_bytes_without_padding(tmp_path):
    books, pleias = _components(tmp_path)
    result = build_ledger(books, pleias, tmp_path / "ledger.json", 2_000)
    assert result["totals"] == {
        "documents": 8,
        "post_rewrite_text_utf8_bytes": 1_800,
        "remaining_byte_headroom": 200,
    }
    assert result["policy"]["ceiling_is_not_a_target"] is True
    assert result["policy"]["padding_for_volume_prohibited"] is True
    assert result["final_corpus_complete"] is False
    assert result["training_ready"] is False


def test_ledger_rejects_volume_above_quality_ceiling(tmp_path):
    books, pleias = _components(tmp_path)
    with pytest.raises(FoundationCorpusLedgerError, match="exceed"):
        build_ledger(books, pleias, tmp_path / "ledger.json", 1_799)
