from __future__ import annotations

from sai.data.license_policy import classify_declared_license


def test_exact_creative_commons_declaration_is_normalized_with_obligations() -> None:
    result = classify_declared_license(
        "Creative Commons - Attribution Share-Alike - "
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    assert result["canonical_license"] == "CC-BY-SA-4.0"
    assert result["attribution_required"] is True
    assert result["share_alike_required"] is True
    assert result["rights_hold"] is False
    assert result["legal_clearance_established"] is False


def test_ambiguous_gfdl_declaration_fails_closed() -> None:
    result = classify_declared_license("GNU Free Documentation License")
    assert result["canonical_license"] is None
    assert result["declaration_recognized"] is False
    assert result["rights_hold"] is True


def test_case_only_spdx_alias_is_canonicalized() -> None:
    result = classify_declared_license("apache-2.0")
    assert result["canonical_license"] == "Apache-2.0"
    assert result["rights_hold"] is False
