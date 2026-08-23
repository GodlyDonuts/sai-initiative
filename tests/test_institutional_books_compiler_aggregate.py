from __future__ import annotations

import pytest

from sai.data.institutional_books_compiler_aggregate import (
    InstitutionalBooksAggregateError,
    triage_route,
)


def _judgment() -> dict:
    return {
        "verdict": "retain",
        "current_language": "english",
        "quality": {
            "overall_quality": 4,
            "ocr_quality": 4,
            "knowledge_density": 4,
            "literary_value": 1,
            "historical_value": 3,
        },
        "risks": {
            "rights_evidence_incomplete": False,
            "duplicate_or_near_duplicate_edition": False,
            "ocr_damage": False,
            "factual_unreliability": False,
            "outdated_or_harmful_claims": False,
        },
    }


def test_high_quality_book_routes_to_representation_verification() -> None:
    assert triage_route(_judgment()) == "representation_verification"


def test_historical_risk_routes_before_quality_score() -> None:
    judgment = _judgment()
    judgment["risks"]["outdated_or_harmful_claims"] = True
    assert triage_route(judgment) == "historical_context_transformation"


def test_non_english_routes_to_translation_after_ocr_and_rights() -> None:
    judgment = _judgment()
    judgment["current_language"] = "russian"
    assert triage_route(judgment) == "translation_review"
    judgment["risks"]["rights_evidence_incomplete"] = True
    assert triage_route(judgment) == "rights_hold"


def test_missing_quality_geometry_fails_closed() -> None:
    with pytest.raises(InstitutionalBooksAggregateError, match="route"):
        triage_route({"risks": {}})
