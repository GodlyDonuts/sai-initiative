from sai.data.institutional_books_materializer import filter_source_rows


def test_filters_one_parent_and_preserves_text_free_lineage():
    source = {"path": "train/data/p.parquet", "sha256": "a" * 64}
    selection = {
        "good": {"row_sha256": "b" * 64, "tokens": 3_000},
        "foreign": {"row_sha256": "c" * 64, "tokens": 4_000},
    }
    rows = [
        {
            "barcode_src": "good",
            "primary_language_gen": "eng",
            "token_count_gen": 3_100,
            "char_count_gen": 12_000,
            "word_count_gen": 2_000,
            "tokenizability_ratio_gen": 99.0,
            "processed_middlematter_gen": "A" * 1_000,
        },
        {
            "barcode_src": "foreign",
            "primary_language_gen": "fra",
            "token_count_gen": 4_000,
            "char_count_gen": 16_000,
            "word_count_gen": 3_000,
            "tokenizability_ratio_gen": 98.0,
            "processed_middlematter_gen": "B" * 1_000,
        },
        {"barcode_src": "not-selected"},
    ]
    outputs, lineage = filter_source_rows(rows, selection, source)
    assert [row["barcode_src"] for row in outputs] == ["good"]
    assert outputs[0]["metadata_token_count_o200k_base_gen"] == 3_000
    assert [row["disposition"] for row in lineage] == [
        "materialized",
        "excluded",
    ]
    assert lineage[1]["exclusion_reason"] == "enriched_primary_language_mismatch"
    assert all(row["source_text_persisted"] is False for row in lineage)
