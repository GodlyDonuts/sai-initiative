import pytest

from sai.data.data_compiler_labeling import RISK_KEYS, SCORE_KEYS
from sai.data.reservoir_audit_aggregate import (
    ReservoirAuditAggregateError,
    summarize,
)


def _receipt(
    verdict: str,
    *,
    language: str = "english",
    bridge: bool = False,
    score: int = 3,
) -> dict:
    return {
        "judgment": {
            "verdict": verdict,
            "epistemic_functions": ["reality_anchor"],
            "domains": ["history"],
            "curriculum_phase": "integration" if verdict != "reject" else "reject",
            "source_language": language,
            "translation_disposition": (
                "not_needed_english"
                if language == "english"
                else "translate_preserve_meaning"
            ),
            "translation_priority": 0 if language == "english" else 4,
            "preservation_policy": (
                "preserve_plus_derivatives" if verdict != "reject" else "reject"
            ),
            "recommended_representations": [
                "original_english" if language == "english" else "english_translation"
            ],
            "style": "exposition",
            "likely_origin": "organic_human",
            "grounding_type": "direct_primary",
            "difficulty": 2,
            "prerequisite_burden": 1,
            "cross_domain_bridges": ["history::economics"] if bridge else [],
            "risks": {
                key: key == "translation_loss" and language != "english"
                for key in RISK_KEYS
            },
            "scores": {key: score for key in SCORE_KEYS},
        },
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "attempts": [
            {"outcome": "invalid_model_output"},
            {"outcome": "valid"},
        ],
    }


def test_aggregate_reports_model_evidence_without_promoting_it() -> None:
    lineage = [
        {"source_id": "finepdfs", "stratum": "english"},
        {"source_id": "finepdfs", "stratum": "chinese"},
    ]
    result = summarize(
        lineage,
        [
            _receipt("retain", bridge=True),
            _receipt("review", language="chinese", score=2),
        ],
    )
    assert result["rows"] == 2
    assert result["by_source_verdict"]["finepdfs"] == {"retain": 1, "review": 1}
    assert result["rows_with_cross_domain_bridges"] == 1
    assert result["potential_translation_rows"] == 1
    assert result["mean_scores_milli"]["writing_quality"] == 2500
    assert result["usage"]["total_tokens"] == 240
    assert result["rows_requiring_repair"] == 2
    assert result["model_judgments_are_verified_admissions"] is False


def test_aggregate_rejects_missing_receipt() -> None:
    with pytest.raises(ReservoirAuditAggregateError, match="inputs"):
        summarize([{"source_id": "one", "stratum": "one"}], [])
