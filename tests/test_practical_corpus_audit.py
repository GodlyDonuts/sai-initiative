import json
from pathlib import Path

import pytest

from sai.data.institutional_books_practical_admission import SCHEMA as BOOKS_SCHEMA
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_SCHEMA
from sai.data.practical_corpus_audit import PracticalCorpusAuditError, build_audit
from sai.data.practical_hf_publish import METADATA_SCHEMA
from sai.data.token_stream import canonical_sha256


def _write(path: Path, payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return payload


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    books_path = root / "books.json"
    pleias_path = root / "pleias.json"
    publication_path = root / "publication.json"
    books = _write(
        books_path,
        {
            "schema": BOOKS_SCHEMA,
            "status": "complete_practical_private_pretraining_admission",
            "counts": {
                "admitted_rows": 10,
                "admitted_text_utf8_bytes": 1_000,
                "admitted_enriched_tokens": 250,
            },
            "semantic_model_review_required": False,
            "huggingface_redistribution_authorized": False,
            "official_benchmark_decontamination_complete": False,
            "practical_pretraining_ready": True,
            "training_ready": True,
        },
    )
    pleias = _write(
        pleias_path,
        {
            "schema": PLEIAS_SCHEMA,
            "status": "complete_practical_pleias_pretraining_admission",
            "counts": {
                "admitted_rows": 20,
                "admitted_text_utf8_bytes": 3_000,
                "admitted_source_token_count": 750,
                "combined_books_plus_pleias_text_utf8_bytes": 4_000,
                "admitted_collection_count": 2,
                "collections": {"books": 8, "science": 12},
                "rights": {"cc0": 5, "public domain": 15},
            },
            "global_exact_content_deduplication_complete": True,
            "source_text_copied": False,
            "official_benchmark_decontamination_complete": False,
            "practical_pretraining_ready": True,
            "training_ready": True,
        },
    )
    _write(
        publication_path,
        {
            "schema": METADATA_SCHEMA,
            "status": "complete_practical_hf_metadata_publication",
            "books_admission_receipt_sha256": books["receipt_sha256"],
            "pleias_admission_receipt_sha256": pleias["receipt_sha256"],
            "remote_repository": "Godlydonuts/Sai",
            "source_text_uploaded": False,
        },
    )
    return books_path, pleias_path, publication_path


def test_audit_seals_exact_ready_totals(tmp_path: Path) -> None:
    books, pleias, publication = _inputs(tmp_path)
    result = build_audit(
        books,
        pleias,
        publication,
        tmp_path / "audit.json",
        minimum_combined_text_bytes=3_900,
        maximum_combined_text_bytes=4_100,
    )
    assert result["totals"] == {
        "source_components": 2,
        "rows": 30,
        "text_utf8_bytes": 4_000,
        "source_token_count": 1_000,
    }
    assert result["practical_training_corpus_ready"] is True
    assert result["quality"]["row_level_rights_labels_complete"] is True
    assert (
        result["custody"]["institutional_books_public_redistribution_allowed"]
        is False
    )
    assert result["four_b_training_authorized"] is False


def test_audit_rejects_underfilled_corpus(tmp_path: Path) -> None:
    books, pleias, publication = _inputs(tmp_path)
    with pytest.raises(PracticalCorpusAuditError, match="totals differ"):
        build_audit(
            books,
            pleias,
            publication,
            tmp_path / "audit.json",
            minimum_combined_text_bytes=4_001,
            maximum_combined_text_bytes=5_000,
        )
