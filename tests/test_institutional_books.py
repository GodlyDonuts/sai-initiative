import copy

import pytest

from sai.data.institutional_books import (
    ENRICHED_REVISION,
    MAX_EXCERPT_BYTES,
    METADATA_PARQUET_BYTES,
    METADATA_PARQUET_SHA256,
    InstitutionalBooksError,
    build_book_candidate,
    representative_excerpt,
)


def _metadata() -> dict:
    return {
        "barcode_src": "32044000000018",
        "title_src": "Geology of Massachusetts and Rhode Island.",
        "author_src": "Emerson, Benjamin Kendall",
        "date1_src": "1917",
        "date2_src": None,
        "page_count_src": 314,
        "token_count_o200k_base_gen": 201834,
        "language_src": "eng",
        "language_gen": "eng",
        "topic_or_subject_src": "Geology",
        "topic_or_subject_gen": "SCIENCE",
        "genre_or_form_src": None,
        "general_note_src": "Map on one folded leaf.",
        "ocr_score_src": 94,
        "ocr_score_gen": 98,
        "likely_duplicates_barcodes_gen": None,
        "identifiers_src": {"lccn": [], "isbn": [], "ocolc": ["272423"]},
        "hathitrust_data_ext": {
            "url": "https://hdl.handle.net/2027/hvd.32044000000018",
            "rights_code": "pd",
            "reason_code": "bib",
            "last_check": "2025-05-12",
        },
    }


def _enriched() -> dict:
    return {
        "barcode_src": "32044000000018",
        "primary_language_gen": "eng",
        "processed_middlematter_gen": "Geology describes rock and time. " * 20_000,
    }


def test_institutional_source_pins_and_representative_sampling() -> None:
    assert len(ENRICHED_REVISION) == 40
    assert METADATA_PARQUET_BYTES == 306_251_508
    assert len(METADATA_PARQUET_SHA256) == 64
    excerpt = representative_excerpt(_enriched()["processed_middlematter_gen"])
    assert excerpt.count("[SAI REPRESENTATIVE EXCERPT BOUNDARY]") == 2
    assert len(excerpt.encode()) <= MAX_EXCERPT_BYTES == 32_768


def test_build_book_candidate_joins_exact_barcode_and_keeps_rights_evidence() -> None:
    candidate = build_book_candidate(_metadata(), _enriched())
    assert candidate["source"]["barcode_src"] == "32044000000018"
    assert candidate["bibliographic"]["rights_evidence"]["status_code"] == "pd"
    assert candidate["bibliographic"]["likely_duplicates_barcodes_gen"] == []
    assert len(candidate["candidate_identity_sha256"]) == 64


def test_build_book_candidate_rejects_cross_book_join_and_metadata_tamper() -> None:
    enriched = copy.deepcopy(_enriched())
    enriched["barcode_src"] = "other"
    with pytest.raises(InstitutionalBooksError, match="join"):
        build_book_candidate(_metadata(), enriched)
    metadata = _metadata()
    metadata.pop("hathitrust_data_ext")
    with pytest.raises(InstitutionalBooksError, match="metadata fields"):
        build_book_candidate(metadata, _enriched())


def test_build_book_candidate_bounds_pathological_identifier_clusters() -> None:
    metadata = _metadata()
    metadata["identifiers_src"]["isbn"] = [
        f"978000000{i:03d}" for i in range(70)
    ] + ["978000000000"]
    metadata["likely_duplicates_barcodes_gen"] = [
        f"book-{index:03d}" for index in range(260)
    ] + ["book-000"]
    candidate = build_book_candidate(metadata, _enriched())
    assert candidate["bibliographic"]["identifiers_src"]["isbn"] == [
        f"978000000{i:03d}" for i in range(64)
    ]
    assert candidate["bibliographic"]["likely_duplicates_barcodes_gen"] == [
        f"book-{index:03d}" for index in range(256)
    ]


def test_build_book_candidate_rejects_invalid_archive_identifier_values() -> None:
    metadata = _metadata()
    metadata["identifiers_src"]["isbn"] = ["valid", ""]
    with pytest.raises(InstitutionalBooksError, match="isbn"):
        build_book_candidate(metadata, _enriched())
