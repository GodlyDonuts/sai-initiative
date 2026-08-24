from __future__ import annotations

from sai.data.independent_review_population import classify, select_rows


def _candidate(identity: str) -> dict:
    return {"candidate_identity_sha256": identity}


def _receipt(verdict: str, *active: str) -> dict:
    risks = {
        "seo_or_content_farm": False,
        "incoherent_or_corrupted": False,
        "factual_unreliability": False,
        "duplicated_boilerplate": False,
        "answer_farm_without_teaching": False,
        "personal_or_secret_data": False,
        "ocr_or_extraction_damage": False,
        "translation_loss": False,
        "cultural_flattening": False,
        "weak_source_grounding": False,
        "generic_synthetic_style": False,
        "license_or_provenance_unclear": False,
    }
    for key in active:
        risks[key] = True
    return {"judgment": {"verdict": verdict, "risks": risks}}


def test_review_strata_are_mutually_exclusive_and_risk_first() -> None:
    assert classify(_receipt("reject")["judgment"]) == "nonretain"
    assert (
        classify(_receipt("retain", "personal_or_secret_data")["judgment"])
        == "severe_risk_retain"
    )
    assert (
        classify(_receipt("retain", "ocr_or_extraction_damage")["judgment"])
        == "cleanup_risk_retain"
    )
    assert classify(_receipt("retain")["judgment"]) == "clean_retain"
    assert classify(_receipt("retain", "factual_unreliability")["judgment"]) is None


def test_review_selection_is_lowest_identity_per_lane_and_stratum() -> None:
    rows = [
        ("a", _candidate("3" * 64), _receipt("review")),
        ("a", _candidate("1" * 64), _receipt("review")),
        ("a", _candidate("2" * 64), _receipt("review")),
        ("b", _candidate("4" * 64), _receipt("retain")),
    ]
    selected = select_rows(rows, 2)
    identities = [item[2]["candidate_identity_sha256"] for item in selected]
    assert identities == ["1" * 64, "2" * 64, "4" * 64]
