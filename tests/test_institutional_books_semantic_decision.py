from __future__ import annotations

from sai.data.institutional_books_semantic_decision import classify_judgment


def _judgment() -> dict:
    return {
        "verdict": "retain",
        "current_language": "english",
        "genre": "science",
        "confidence_ppm": 950_000,
        "quality": {
            "overall_quality": 4,
            "ocr_quality": 4,
            "knowledge_density": 4,
            "literary_value": 1,
            "factual_reliability": 4,
            "historical_value": 2,
        },
        "rights_evidence": {"status_code": "pdus"},
        "risks": {
            "rights_evidence_incomplete": False,
            "duplicate_or_near_duplicate_edition": False,
            "ocr_damage": False,
            "factual_unreliability": False,
            "outdated_or_harmful_claims": False,
            "bibliographic_ambiguity": False,
            "translation_loss": False,
            "cultural_flattening": False,
            "generic_model_voice": False,
        },
    }


def test_only_exceptional_clean_book_advances_to_independent_verification() -> None:
    assert classify_judgment(_judgment()) == ("independent_verification", [])


def test_rights_and_risk_holds_precede_quality() -> None:
    judgment = _judgment()
    judgment["rights_evidence"]["status_code"] = "unknown"
    assert classify_judgment(judgment) == (
        "rights_hold",
        ["rights_code_not_allowed"],
    )
    judgment = _judgment()
    judgment["risks"]["bibliographic_ambiguity"] = True
    assert classify_judgment(judgment) == (
        "risk_hold",
        ["active_risk:bibliographic_ambiguity"],
    )


def test_technical_factual_reliability_and_confidence_fail_closed() -> None:
    judgment = _judgment()
    judgment["quality"]["factual_reliability"] = 3
    judgment["confidence_ppm"] = 899_999
    disposition, reasons = classify_judgment(judgment)
    assert disposition == "quality_hold"
    assert reasons == [
        "confidence_below_floor",
        "technical_factual_reliability_below_floor",
    ]


def test_literature_uses_literary_value_not_factual_density() -> None:
    judgment = _judgment()
    judgment["genre"] = "literature"
    judgment["quality"]["knowledge_density"] = 1
    judgment["quality"]["literary_value"] = 4
    judgment["quality"]["factual_reliability"] = 1
    assert classify_judgment(judgment) == ("independent_verification", [])
