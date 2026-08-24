import json

import pytest

from sai.data.foundation_source_split import POLICY_SHA256 as SPLIT_POLICY_SHA256
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_virtual_byte_balance import (
    AGGREGATE_SCHEMA as PLEIAS_BALANCE_SCHEMA,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA as PLEIAS_SCHEMA,
)
from sai.data.token_stream import canonical_sha256
from sai.data.virtual_foundation_corpus_ledger import (
    VirtualFoundationCorpusLedgerError,
    build_ledger,
)


def _write(path, payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))


def _components(tmp_path):
    books = tmp_path / "books.json"
    pleias = tmp_path / "pleias.json"
    balance = tmp_path / "balance.json"
    _write(
        books,
        {
            "schema": BOOK_SCHEMA,
            "totals": {
                "documents": 3,
                "output_text_utf8_bytes": 600,
                "split::train::documents": 2,
                "split::development::documents": 1,
                "split::train::text_utf8_bytes": 400,
                "split::development::text_utf8_bytes": 200,
                "semantic_genre::technical::documents": 3,
                "semantic_domain::science::documents": 3,
                "curriculum_band_vote::intermediate::documents": 3,
            },
            "complete_benchmark_disjoint_book_coverage": True,
            "private_storage_only": True,
            "huggingface_redistribution_authorized": False,
            "benchmark_decontamination_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "token_count_requires_recomputation": True,
            "source_disjoint_split_complete": True,
            "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
            "physical_train_development_partition_complete": True,
            "semantic_quality_metadata_complete": True,
            "curriculum_metadata_complete": True,
            "training_ready": False,
        },
    )
    _write(
        pleias,
        {
            "schema": PLEIAS_SCHEMA,
            "totals": {
                "documents": 5,
                "output_text_utf8_bytes": 1_200,
                "split::train::documents": 4,
                "split::development::documents": 1,
                "split::train::text_utf8_bytes": 1_000,
                "split::development::text_utf8_bytes": 200,
                "semantic_stratum::reference::documents": 5,
                "quality_floor_milli::8000::documents": 5,
                "difficulty_mean_milli::2500::documents": 5,
                "curriculum_phase::foundation::documents": 5,
                "semantic_domain::science::documents": 5,
            },
            "complete_final_pleias_document_coverage": True,
            "benchmark_decontamination_complete": True,
            "pleias_internal_subdocument_deduplication_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "source_text_persisted": False,
            "token_count_requires_recomputation": True,
            "source_disjoint_split_complete": True,
            "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
            "physical_train_development_partition_complete": True,
            "semantic_quality_metadata_complete": True,
            "curriculum_metadata_complete": True,
            "training_ready": False,
        },
    )
    book_payload = json.loads(books.read_text())
    pleias_payload = json.loads(pleias.read_text())
    _write(
        balance,
        {
            "schema": PLEIAS_BALANCE_SCHEMA,
            "status": "complete_nontraining_pleias_virtual_byte_balance",
            "source": {
                "book_aggregate_receipt_sha256": book_payload["receipt_sha256"],
                "pleias_aggregate_receipt_sha256": pleias_payload["receipt_sha256"],
            },
            "selected_counts": pleias_payload["totals"],
            "byte_ceiling_respected": True,
            "padding_performed": False,
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    return books, pleias, balance


def test_virtual_ledger_binds_exact_bytes_without_claiming_payload(tmp_path):
    books, pleias, balance = _components(tmp_path)
    result = build_ledger(books, pleias, balance, tmp_path / "ledger.json", 2_000)
    assert result["totals"]["post_rewrite_text_utf8_bytes"] == 1_800
    assert result["totals"]["remaining_byte_headroom"] == 200
    assert result["components"][1]["custody"] == (
        "pinned_source_plus_verified_reconstruction_locators"
    )
    assert result["pleias_virtual_reconstruction_complete"] is True
    assert result["pleias_payload_materialization_complete"] is False
    assert result["final_corpus_complete"] is False
    assert result["training_ready"] is False


def test_virtual_ledger_rejects_unbound_source_text(tmp_path):
    books, pleias, balance = _components(tmp_path)
    payload = json.loads(pleias.read_text())
    payload.pop("receipt_sha256")
    payload["source_text_persisted"] = True
    _write(pleias, payload)
    with pytest.raises(
        VirtualFoundationCorpusLedgerError, match="component completion"
    ):
        build_ledger(books, pleias, balance, tmp_path / "ledger.json", 2_000)


def test_virtual_ledger_rejects_bytes_above_ceiling(tmp_path):
    books, pleias, balance = _components(tmp_path)
    with pytest.raises(VirtualFoundationCorpusLedgerError, match="exceed"):
        build_ledger(books, pleias, balance, tmp_path / "ledger.json", 1_799)
