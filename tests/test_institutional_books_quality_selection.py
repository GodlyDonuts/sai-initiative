from sai.data.institutional_books_quality_selection import select_rows


def _row(barcode, language, ocr, tokens, duplicates=None, rights="pd"):
    return {
        "barcode_src": barcode,
        "title_src": barcode,
        "author_src": "author",
        "date1_src": "1900",
        "topic_or_subject_gen": "LANGUAGE AND LITERATURE",
        "topic_or_subject_score_gen": 0.9,
        "genre_or_form_src": "books",
        "language_gen": language,
        "ocr_score_gen": ocr,
        "token_count_o200k_base_gen": tokens,
        "likely_duplicates_barcodes_gen": duplicates or [],
        "text_analysis_gen": {},
        "hathitrust_data_ext": {
            "rights_code": rights,
            "reason_code": "bib",
            "last_check": "2026-01-01",
        },
    }


def test_selects_only_strict_english_rights_bounded_representatives():
    rows = [
        _row("best", "eng", 98, 10_000, ["duplicate"]),
        _row("duplicate", "eng", 96, 9_000, ["best"]),
        _row("french", "fra", 99, 20_000),
        _row("ocr94", "eng", 94, 30_000),
        _row("rights", "eng", 99, 40_000, rights="ic"),
    ]
    selected = select_rows(rows)
    assert [row["barcode_src"] for row in selected] == ["best"]
    assert selected[0]["token_count_o200k_base_gen"] == 10_000
    assert selected[0]["source_text_persisted"] is False
    assert selected[0]["training_ready"] is False
