from __future__ import annotations

from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.confirmation_promotion import decide_sources
from sai.data.cross_population_duplicates import SCHEMA as DUPLICATE_SCHEMA
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256


def _signed(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _evidence(*, contaminated: int = 0, quarantine: int = 0):
    source = "common_pile_example"
    population = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "by_source": {source: 32},
            "identity_disjoint_from_discovery": True,
            "exact_content_disjoint_from_discovery": True,
            "training_ready": False,
        }
    )
    aggregate = _signed(
        {
            "schema": AGGREGATE_SCHEMA,
            "status": "complete",
            "population_receipt_sha256": population["receipt_sha256"],
            "summary": {
                "by_source_conservative_triage": {
                    source: {
                        "representation_verification": 32 - quarantine,
                        "quarantine": quarantine,
                    }
                }
            },
            "coverage_first_not_statistical_acceptance_estimate": True,
            "training_ready": False,
        }
    )
    screen = _signed(
        {
            "schema": SCREEN_SCHEMA,
            "status": "complete",
            "population": {"receipt_sha256": population["receipt_sha256"]},
            "summary": {
                "by_source": {
                    source: {"rows": 32, "contaminated_rows": contaminated}
                }
            },
            "benchmark_contamination_screen_complete": True,
            "training_ready": False,
        }
    )
    duplicates = _signed(
        {
            "schema": DUPLICATE_SCHEMA,
            "status": "complete",
            "populations": [
                {
                    "population": "confirmation",
                    "receipt_sha256": population["receipt_sha256"],
                }
            ],
            "flagged_pairs": 0,
            "cross_population_flagged_pairs": 0,
            "sample_exact_duplicate_audit_complete": True,
            "training_ready": False,
        }
    )
    return population, aggregate, screen, duplicates


def test_clean_confirmation_authorizes_only_a_bounded_pilot() -> None:
    decision = decide_sources(*_evidence())[0]
    assert decision["bounded_streaming_source_pilot_authorized"] is True
    assert decision["next_action"] == "build_bounded_streaming_source_pilot"
    assert decision["bulk_training_admission"] is False
    assert decision["training_ready"] is False


def test_contamination_blocks_source_pilot() -> None:
    decision = decide_sources(*_evidence(contaminated=1))[0]
    assert decision["bounded_streaming_source_pilot_authorized"] is False
    assert decision["failed_checks"] == ["zero_benchmark_contamination"]


def test_quarantine_blocks_source_pilot() -> None:
    decision = decide_sources(*_evidence(quarantine=1))[0]
    assert decision["bounded_streaming_source_pilot_authorized"] is False
    assert "zero_quarantine" in decision["failed_checks"]
