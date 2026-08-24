from __future__ import annotations

from sai.data.common_pile_full_source_promotion import _source_evidence
from sai.data.data_compiler_labeling import RISK_KEYS, SCORE_KEYS


def _receipt(*, verdict: str = "retain", quarantine: bool = False) -> dict:
    risks = {key: False for key in RISK_KEYS}
    risks["incoherent_or_corrupted"] = quarantine
    scores = {key: 3 for key in SCORE_KEYS}
    scores.update(
        {
            "coherence": 4,
            "educational_value": 4,
            "source_reliability": 4,
            "information_density": 4,
        }
    )
    return {
        "judgment": {
            "verdict": verdict,
            "source_language": "english",
            "epistemic_functions": ["reality_anchor"],
            "risks": risks,
            "scores": scores,
        }
    }


def test_strong_source_authorizes_candidate_materialization_only() -> None:
    lineage = [{"source_id": "pressbooks"} for _ in range(1_024)]
    receipts = [_receipt() for _ in range(900)] + [
        _receipt(verdict="review") for _ in range(124)
    ]
    result = _source_evidence("pressbooks", lineage, receipts)
    assert result["bounded_rows"] == 1_024
    assert result["retain_ppm"] == 878_906
    assert result["full_source_candidate_materialization_authorized"] is True
    assert result["failed_checks"] == []


def test_quarantine_heavy_source_fails_closed() -> None:
    lineage = [{"source_id": "noisy"} for _ in range(1_024)]
    receipts = [_receipt() for _ in range(800)] + [
        _receipt(verdict="reject", quarantine=True) for _ in range(224)
    ]
    result = _source_evidence("noisy", lineage, receipts)
    assert result["quarantine_ppm"] == 218_750
    assert result["full_source_candidate_materialization_authorized"] is False
    assert "maximum_quarantine_ppm" in result["failed_checks"]
    assert "minimum_retain_ppm" in result["failed_checks"]
