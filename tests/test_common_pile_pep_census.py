import copy

import pytest

from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.common_pile_pep_census import (
    SOURCE_ID,
    CommonPilePepCensusError,
    validate_recovery_evidence,
)
from sai.data.common_pile_rights_audit import SCHEMA as RIGHTS_SCHEMA
from sai.data.confirmation_promotion import SCHEMA as PROMOTION_SCHEMA
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA


def _evidence() -> tuple[dict, dict, dict, dict, list[dict]]:
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "summary": {
            "by_source_verdict": {SOURCE_ID: {"retain": 32}},
            "by_source_conservative_triage": {
                SOURCE_ID: {"representation_verification": 27, "quarantine": 2}
            },
        },
        "training_ready": False,
    }
    screen = {
        "schema": SCREEN_SCHEMA,
        "status": "complete",
        "benchmark_contamination_screen_complete": True,
        "summary": {
            "by_source": {SOURCE_ID: {"rows": 32, "contaminated_rows": 0}}
        },
    }
    rights = {
        "schema": RIGHTS_SCHEMA,
        "status": "complete_declaration_audit_not_legal_clearance",
        "summary": {
            "by_source": {
                SOURCE_ID: {
                    "rows": 32,
                    "rights_hold_rows": 0,
                    "recognized_declaration_rows": 32,
                    "canonical_license:LicenseRef-Public-Domain": 32,
                }
            }
        },
        "training_ready": False,
    }
    promotion = {
        "schema": PROMOTION_SCHEMA,
        "status": "complete",
        "sources": [
            {
                "source_id": SOURCE_ID,
                "bounded_streaming_source_pilot_authorized": False,
                "failed_checks": ["zero_quarantine"],
            }
        ],
    }
    rows = [{"source_id": SOURCE_ID, "physical_bytes": 3_723_467}]
    return aggregate, screen, rights, promotion, rows


def test_recovery_evidence_authorizes_only_filtered_census() -> None:
    result = validate_recovery_evidence(*_evidence())
    assert result["retain_ppm"] == 1_000_000
    assert result["representation_verification_ppm"] == 843_750
    assert result["quarantine_ppm"] == 62_500
    assert result["decision_scope"] == (
        "complete_filtered_nontraining_parent_census_only"
    )
    assert result["bulk_training_admission"] is False
    assert result["training_ready"] is False


def test_recovery_evidence_rejects_benchmark_overlap() -> None:
    evidence = list(_evidence())
    screen = copy.deepcopy(evidence[1])
    screen["summary"]["by_source"][SOURCE_ID]["contaminated_rows"] = 1
    evidence[1] = screen
    with pytest.raises(CommonPilePepCensusError, match="evidence"):
        validate_recovery_evidence(*evidence)
