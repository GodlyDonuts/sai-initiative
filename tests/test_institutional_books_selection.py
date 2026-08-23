import pytest

from sai.data.institutional_books_selection import (
    InstitutionalBooksSelectionError,
    select_metadata_rows,
)


def _row(
    barcode: str,
    language: str,
    topic: str,
    *,
    ocr: int = 95,
    duplicates: list[str] | None = None,
) -> dict:
    return {
        "barcode_src": barcode,
        "title_src": f"Title {barcode}",
        "author_src": "Author",
        "date1_src": "1900",
        "token_count_o200k_base_gen": 20_000,
        "language_src": language,
        "language_gen": language,
        "topic_or_subject_src": topic,
        "topic_or_subject_gen": topic,
        "topic_or_subject_score_gen": 0.99,
        "genre_or_form_src": "book",
        "general_note_src": None,
        "ocr_score_gen": ocr,
        "likely_duplicates_barcodes_gen": duplicates or [],
        "text_analysis_gen": {"text_by_page_gen": {"tokenizability_score": 95.0}},
        "hathitrust_data_ext": {"rights_code": "pd"},
    }


def test_selection_is_duplicate_safe_and_coverage_first() -> None:
    rows = [
        _row("a", "eng", "SCIENCE", ocr=90, duplicates=["b"]),
        _row("b", "eng", "SCIENCE", ocr=99, duplicates=["a"]),
        _row("c", "rus", "LANGUAGE AND LITERATURE"),
        _row("d", "fra", "PHILOSOPHY"),
        _row("e", "eng", "SCIENCE", ocr=98),
    ]
    selected, statistics = select_metadata_rows(rows, 3)
    barcodes = {row["barcode_src"] for row in selected}
    assert "a" not in barcodes
    assert "b" in barcodes
    assert {row["language_gen"] for row in selected} == {"eng", "rus", "fra"}
    assert statistics["duplicate_components"] == 4
    assert statistics["selected_non_english_rows"] == 2
    assert all(row["training_ready"] is False for row in selected)


def test_selection_rejects_bad_rights_ocr_and_oversized_target() -> None:
    bad = _row("a", "eng", "SCIENCE", ocr=70)
    good = _row("b", "eng", "SCIENCE")
    good["hathitrust_data_ext"] = {"rights_code": "ic"}
    with pytest.raises(InstitutionalBooksSelectionError, match="exceeds"):
        select_metadata_rows([bad, good], 1)


def test_selection_rejects_duplicate_primary_barcodes() -> None:
    with pytest.raises(InstitutionalBooksSelectionError, match="barcode"):
        select_metadata_rows([_row("a", "eng", "SCIENCE")] * 2, 1)
