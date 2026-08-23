import copy

import pytest

from sai.data.opencoder_promotion_screen import (
    EXPECTED_SCREEN_ROWS,
    OpenCoderPromotionScreenError,
    _metric_gates,
    summarize_screen,
)


def _receipt(*, route: str = "representation", domain: str = "computer_science"):
    judgment = {
        "verdict": "retain",
        "domains": [domain],
        "scores": {
            "educational_value": 3,
            "technical_depth": 3,
            "information_density": 3,
            "source_reliability": 3,
            "coherence": 3,
            "writing_quality": 3,
            "human_expression_value": 0,
            "reasoning_density": 3,
            "cross_domain_bridge_value": 0,
            "factual_reference_value": 3,
            "formatting_quality": 3,
            "novelty_potential": 2,
            "cultural_context_value": 0,
        },
        "risks": {
            "personal_or_secret_data": False,
            "incoherent_or_corrupted": False,
            "seo_or_content_farm": False,
            "answer_farm_without_teaching": False,
            "license_or_provenance_unclear": False,
            "factual_unreliability": False,
            "weak_source_grounding": False,
            "ocr_or_extraction_damage": False,
            "duplicated_boilerplate": False,
            "translation_loss": False,
            "cultural_flattening": False,
            "generic_synthetic_style": False,
        },
        "source_language": "english",
        "epistemic_functions": ["procedural_reasoning"],
    }
    if route == "quarantine":
        judgment["verdict"] = "reject"
    elif route == "quality":
        judgment["scores"]["reasoning_density"] = 2
        judgment["scores"]["educational_value"] = 2
    return {
        "receipt_sha256": "a" * 64,
        "judgment": judgment,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "attempts": [{"outcome": "valid"}],
    }


def test_metric_gates_use_exact_integer_thresholds() -> None:
    result = _metric_gates(
        rows=100,
        representation_rows=30,
        quarantine_rows=25,
        computer_science_rows=60,
        educational_value_sum=250,
        technical_depth_sum=250,
    )
    assert all(value["passed"] for value in result.values())
    result["representation_verification"] = _metric_gates(
        rows=100,
        representation_rows=29,
        quarantine_rows=25,
        computer_science_rows=60,
        educational_value_sum=250,
        technical_depth_sum=250,
    )["representation_verification"]
    assert result["representation_verification"]["passed"] is False


def test_screen_promotes_only_when_every_gate_passes() -> None:
    receipts = [_receipt() for _ in range(EXPECTED_SCREEN_ROWS)]
    result = summarize_screen(receipts)
    assert result["all_gates_passed"] is True
    assert result["decision"] == "promote_full_2048_row_audit"
    assert result["gates"]["computer_science"]["observed_milli"] == 1000

    failed = copy.deepcopy(receipts)
    for index in range(70):
        failed[index] = _receipt(route="quarantine")
    result = summarize_screen(failed)
    assert result["gates"]["quarantine"]["passed"] is False
    assert result["all_gates_passed"] is False
    assert result["decision"] == "stop_full_audit_and_reallocate_hermes_capacity"


def test_screen_rejects_partial_coverage() -> None:
    with pytest.raises(OpenCoderPromotionScreenError, match="row count differs"):
        summarize_screen([_receipt()])
