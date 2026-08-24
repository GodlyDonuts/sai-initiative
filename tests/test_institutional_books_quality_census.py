from sai.data.institutional_books_quality_census import build_payload


def _row(barcode, language, ocr, tokens, duplicates=None, rights="pd"):
    return {
        "barcode_src": barcode,
        "title_src": barcode,
        "author_src": "author",
        "date1_src": "1900",
        "topic_or_subject_gen": "SCIENCE",
        "topic_or_subject_score_gen": 0.9,
        "language_gen": language,
        "ocr_score_gen": ocr,
        "token_count_o200k_base_gen": tokens,
        "likely_duplicates_barcodes_gen": duplicates or [],
        "text_analysis_gen": {},
        "hathitrust_data_ext": {"rights_code": rights},
    }


def test_measures_nested_english_and_translation_tiers_after_deduplication():
    rows = [
        _row("eng-best", "eng", 97, 10_000, ["eng-worse"]),
        _row("eng-worse", "eng", 90, 12_000, ["eng-best"]),
        _row("fra", "fra", 92, 20_000),
        _row("low", "eng", 79, 30_000),
        _row("unclear", "eng", 99, 40_000, rights="ic"),
    ]
    result = build_payload(rows)
    counts = result["counts"]
    assert result["duplicate_components"] == 4
    assert counts["english_ocr_95_rows"] == 1
    assert counts["english_ocr_95_tokens"] == 10_000
    assert counts["english_ocr_90_rows"] == 1
    assert counts["translation_ocr_90_rows"] == 1
    assert counts["translation_ocr_90_tokens"] == 20_000
    assert "translation_ocr_95_rows" not in counts
    assert result["rows_by_language"]["ocr_90"] == {"eng": 1, "fra": 1}
    assert result["training_ready"] is False


def test_excludes_out_of_range_token_counts():
    result = build_payload(
        [
            _row("short", "eng", 99, 1_999),
            _row("long", "eng", 99, 2_000_001),
            _row("usable", "eng", 95, 2_000),
        ]
    )
    assert result["counts"]["english_ocr_95_rows"] == 1
    assert result["counts"]["english_ocr_95_tokens"] == 2_000
