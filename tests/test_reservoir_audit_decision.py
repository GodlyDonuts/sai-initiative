from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.reservoir_audit_decision import (
    ReservoirAuditDecisionError,
    build_decision,
)
from sai.data.token_stream import canonical_sha256


def _aggregate(path: Path) -> Path:
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "coverage_first_not_statistical_acceptance_estimate": True,
        "independent_factual_verification_complete": False,
        "cross_source_deduplication_complete": False,
        "benchmark_decontamination_complete": False,
        "training_ready": False,
        "summary": {
            "rows": 30,
            "by_source_conservative_triage": {
                "cleaner": {"representation_verification": 8, "cleanup_review": 2},
                "noisy": {"quarantine": 6, "cleanup_review": 4},
                "rights": {"rights_hold": 8, "quarantine": 2},
            },
            "by_source_verdict": {
                "cleaner": {"retain": 10},
                "noisy": {"retain": 4, "reject": 6},
                "rights": {"review": 8, "reject": 2},
            },
            "model_judgments_are_verified_admissions": False,
            "representation_verification_is_training_admission": False,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def test_decision_routes_sources_without_claiming_acceptance(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path / "aggregate.json")
    result = build_decision(aggregate, tmp_path / "decision.json")
    by_source = {row["source_id"]: row for row in result["sources"]}
    assert by_source["cleaner"]["next_action"] == "priority_targeted_verification"
    assert (
        by_source["noisy"]["next_action"]
        == "bulk_expansion_paused_pending_stratified_confirmation"
    )
    assert (
        by_source["rights"]["next_action"]
        == "rights_blocked_pending_source_specific_resolution"
    )
    assert all(row["bulk_training_admission"] is False for row in result["sources"])
    assert result["training_ready"] is False
    assert result["four_b_training_authorized"] is False


def test_decision_rejects_rehashed_aggregate_semantic_tamper(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path / "aggregate.json")
    payload = json.loads(aggregate.read_text())
    payload["coverage_first_not_statistical_acceptance_estimate"] = False
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    aggregate.write_text(json.dumps(payload))
    with pytest.raises(ReservoirAuditDecisionError, match="receipt"):
        build_decision(aggregate, tmp_path / "decision.json")


def test_decision_rejects_source_row_coverage_tamper(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path / "aggregate.json")
    payload = json.loads(aggregate.read_text())
    payload["summary"]["rows"] = 31
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    aggregate.write_text(json.dumps(payload))
    with pytest.raises(ReservoirAuditDecisionError, match="cover"):
        build_decision(aggregate, tmp_path / "decision.json")
