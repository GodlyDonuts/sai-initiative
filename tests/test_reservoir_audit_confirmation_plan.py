from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.decontamination import POLICY
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.reservoir_audit_confirmation_plan import (
    ReservoirAuditConfirmationPlanError,
    build_plan,
)
from sai.data.token_stream import canonical_sha256


def _write(path: Path, payload: dict) -> Path:
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _aggregate(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": AGGREGATE_SCHEMA,
            "status": "complete",
            "summary": {
                "rows": 12,
                "by_source_conservative_triage": {
                    "clean": {"representation_verification": 3, "cleanup_review": 1},
                    "contaminated": {"representation_verification": 4},
                    "quarantined": {
                        "representation_verification": 2,
                        "quarantine": 2,
                    },
                },
                "model_judgments_are_verified_admissions": False,
                "representation_verification_is_training_admission": False,
            },
            "coverage_first_not_statistical_acceptance_estimate": True,
            "independent_factual_verification_complete": False,
            "cross_source_deduplication_complete": False,
            "benchmark_decontamination_complete": False,
            "training_ready": False,
        },
    )


def _screen(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": SCREEN_SCHEMA,
            "status": "complete",
            "policy": POLICY,
            "policy_sha256": canonical_sha256(POLICY),
            "summary": {
                "rows": 12,
                "by_source": {
                    "clean": {"rows": 4, "contaminated_rows": 0},
                    "contaminated": {"rows": 4, "contaminated_rows": 1},
                    "quarantined": {"rows": 4, "contaminated_rows": 0},
                },
                "individual_decisions_persisted": False,
                "source_text_persisted": False,
            },
            "benchmark_contamination_screen_complete": True,
            "full_source_population_decontaminated": False,
            "training_ready": False,
        },
    )


def test_plan_requires_independent_clean_and_quality_signals(tmp_path: Path) -> None:
    result = build_plan(
        _aggregate(tmp_path / "aggregate.json"),
        _screen(tmp_path / "screen.json"),
        tmp_path / "plan.json",
    )
    assert result["selected_source_ids"] == ["clean"]
    assert result["target_confirmation_rows"] == 32
    rows = {row["source_id"]: row for row in result["sources"]}
    assert rows["contaminated"]["exclusion_reasons"] == ["benchmark_overlap_observed"]
    assert rows["quarantined"]["exclusion_reasons"] == ["quarantine_observed"]
    assert result["bulk_training_admission"] is False
    assert result["training_ready"] is False


def test_plan_rejects_tampered_screen(tmp_path: Path) -> None:
    aggregate = _aggregate(tmp_path / "aggregate.json")
    screen = _screen(tmp_path / "screen.json")
    payload = json.loads(screen.read_text())
    payload["summary"]["by_source"]["clean"]["contaminated_rows"] = 1
    screen.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ReservoirAuditConfirmationPlanError, match="screen"):
        build_plan(aggregate, screen, tmp_path / "plan.json")
