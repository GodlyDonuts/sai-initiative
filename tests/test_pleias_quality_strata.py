from __future__ import annotations

import pytest

from sai.data.pleias_quality_strata import PleiasQualityStrataError, summarize


def _source(index: int, collection: str, language: str) -> dict:
    return {
        "full_text_bytes": 100 + index,
        "locator": {
            "collection": collection,
            "language": language,
            "open_type": "Open Culture",
        },
    }


def _judgment(verdict: str, route_risk: str | None = None) -> dict:
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
    if route_risk is not None:
        risks[route_risk] = True
    return {
        "verdict": verdict,
        "preservation_policy": "reject" if verdict == "reject" else "preserve",
        "source_language": "english",
        "translation_disposition": "not_needed_english",
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
        "risks": risks,
    }


def test_strata_report_is_source_safe_and_conservative() -> None:
    lineage = [_source(i, "Books", "English") for i in range(8)]
    judgments = [_judgment("retain") for _ in range(8)]
    result = summarize(lineage, judgments)

    collection = result["collection"][0]
    assert collection["value"] == "Books"
    assert collection["rows"] == 8
    assert collection["full_text_bytes"] == sum(100 + i for i in range(8))
    assert collection["route_counts"] == {"representation_verification": 8}
    assert collection["screen_priority"] == "priority_targeted_materialization_screen"
    assert collection["bulk_training_admission"] is False
    assert all("text" not in row for rows in result.values() for row in rows)


def test_strata_report_holds_quarantine_heavy_groups() -> None:
    lineage = [_source(i, "Archive", "English") for i in range(10)]
    judgments = [
        _judgment("reject", "incoherent_or_corrupted") if i < 2 else _judgment("retain")
        for i in range(10)
    ]
    collection = summarize(lineage, judgments)["collection"][0]
    assert collection["route_counts"]["quarantine"] == 2
    assert collection["screen_priority"] == "hold_for_row_level_recovery"


def test_strata_report_rejects_missing_metadata() -> None:
    with pytest.raises(PleiasQualityStrataError, match="metadata"):
        summarize(
            [
                {
                    "full_text_bytes": 100,
                    "locator": {
                        "collection": "Books",
                        "language": 7,
                        "open_type": "Open Culture",
                    },
                }
            ],
            [_judgment("retain")],
        )
