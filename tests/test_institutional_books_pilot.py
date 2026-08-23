import copy

import pytest

from sai.data.institutional_books_pilot import (
    InstitutionalBooksPilotError,
    build_candidate_population,
)


def _metadata(barcode: str, language: str = "eng", ocr: int = 95) -> dict:
    return {
        "barcode_src": barcode,
        "title_src": "A Book",
        "author_src": "An Author",
        "date1_src": "1900",
        "date2_src": None,
        "page_count_src": 100,
        "token_count_o200k_base_gen": 10_000,
        "language_src": language,
        "language_gen": language,
        "topic_or_subject_src": "Science",
        "topic_or_subject_gen": "SCIENCE",
        "genre_or_form_src": "Book",
        "general_note_src": None,
        "ocr_score_src": ocr,
        "ocr_score_gen": ocr,
        "likely_duplicates_barcodes_gen": [],
        "identifiers_src": {"lccn": [], "isbn": [], "ocolc": []},
        "hathitrust_data_ext": {
            "url": "https://example.test/book",
            "rights_code": "pd",
            "reason_code": "bib",
            "last_check": "2025-01-01",
        },
    }


def _enriched(barcode: str, language: str = "eng") -> dict:
    return {
        "barcode_src": barcode,
        "primary_language_gen": language,
        "processed_middlematter_gen": "A grounded passage about science. " * 100,
    }


def test_pilot_builds_english_and_translation_candidates() -> None:
    candidates, statistics = build_candidate_population(
        [_metadata("a"), _metadata("b", "fra")],
        [_enriched("a"), _enriched("b", "fra")],
    )
    assert len(candidates) == 2
    assert statistics["candidate_english_rows"] == 1
    assert statistics["candidate_non_english_rows"] == 1
    assert all(row["source"]["barcode_src"] in {"a", "b"} for row in candidates)


def test_pilot_filters_low_ocr_before_model_calls() -> None:
    candidates, statistics = build_candidate_population(
        [_metadata("a", ocr=70), _metadata("b")],
        [_enriched("a"), _enriched("b")],
    )
    assert len(candidates) == 1
    assert statistics["rejection_reasons"] == {"ocr_below_minimum": 1}


def test_pilot_rejects_missing_or_duplicate_joins() -> None:
    with pytest.raises(InstitutionalBooksPilotError, match="incomplete"):
        build_candidate_population([_metadata("a")], [_enriched("b")])
    duplicate = copy.deepcopy(_enriched("a"))
    with pytest.raises(InstitutionalBooksPilotError, match="enriched barcode"):
        build_candidate_population([_metadata("a")], [_enriched("a"), duplicate])
