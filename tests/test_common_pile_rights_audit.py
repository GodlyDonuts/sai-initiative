from __future__ import annotations

from sai.data.common_pile_rights_audit import summarize_rights


def test_rights_summary_separates_recognized_and_ambiguous_declarations() -> None:
    result = summarize_rights(
        [
            {
                "source_id": "common_pile_libretexts",
                "declared_license": "Creative Commons - Attribution - "
                "https://creativecommons.org/licenses/by/4.0/",
            },
            {
                "source_id": "common_pile_libretexts",
                "declared_license": "GNU Free Documentation License",
            },
            {
                "source_id": "common_pile_peps",
                "declared_license": "Public Domain",
            },
        ]
    )
    libretexts = result["by_source"]["common_pile_libretexts"]
    assert result["rows"] == 3
    assert libretexts["recognized_declaration_rows"] == 1
    assert libretexts["rights_hold_rows"] == 1
    assert libretexts["attribution_required_rows"] == 1
    assert result["canonical_license_counts"]["<rights_hold>"] == 1
    assert result["source_wide_rights_clearance_established"] is False
    assert result["legal_clearance_established"] is False
