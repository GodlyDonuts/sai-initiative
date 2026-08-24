from __future__ import annotations

from sai.data.independent_review_compare import summarize_row


def _judgment(verdict: str, *active: str) -> dict:
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
    return {
        "verdict": verdict,
        "risks": risks,
        "source_language": "english",
        "epistemic_functions": ["reality_anchor"],
        "scores": {
            "writing_quality": 4,
            "information_density": 4,
            "educational_value": 4,
            "reasoning_density": 4,
            "factual_reference_value": 4,
            "source_reliability": 4,
            "technical_depth": 4,
            "coherence": 4,
            "formatting_quality": 4,
            "human_expression_value": 4,
            "cultural_context_value": 4,
            "cross_domain_bridge_value": 4,
            "novelty_potential": 4,
        },
    }


def test_comparison_requires_complete_exact_agreement_for_no_adjudication() -> None:
    primary = _judgment("retain", "duplicated_boilerplate")
    same = _judgment("retain", "duplicated_boilerplate")
    summary = summarize_row(primary, {"a": same, "b": same})
    assert summary["complete_review_coverage"] is True
    assert summary["all_available_route_agree"] is True
    assert summary["all_available_active_risks_agree"] is True
    assert summary["manual_adjudication_required"] is False
    assert summary["automatic_training_admission"] is False


def test_comparison_routes_missing_or_disagreement_to_adjudication() -> None:
    primary = _judgment("retain")
    rejected = _judgment("reject", "incoherent_or_corrupted")
    disagreement = summarize_row(primary, {"a": rejected, "b": None})
    assert disagreement["complete_review_coverage"] is False
    assert disagreement["all_available_verdict_agree"] is False
    assert disagreement["all_available_route_agree"] is False
    assert disagreement["manual_adjudication_required"] is True
