import json

import pytest

from sai.data.pleias_quality_core_policy import (
    PleiasQualityCorePolicyError,
    build_policy_payload,
)


def _census():
    return {
        "schema": "sai-pleias-metadata-census-aggregate-v1",
        "status": "complete_nontraining_pleias_metadata_census",
        "axes": {
            "collection_language": {
                json.dumps(["Books", "English"], separators=(",", ":")): {
                    "files": 2,
                    "rows": 100,
                    "word_count": 1000,
                    "token_count": 1400,
                },
                json.dumps(["Courts", "Chinese"], separators=(",", ":")): {
                    "files": 1,
                    "rows": 50,
                    "word_count": 500,
                    "token_count": 800,
                },
            }
        },
        "totals": {"rows": 150, "word_count": 1500, "token_count": 2200},
        "receipt_sha256": "a" * 64,
        "training_ready": False,
    }


def _quality():
    return {
        "schema": "sai-pleias-quality-strata-report-v1",
        "status": "complete_nontraining_quality_strata_report",
        "axes": {
            "collection_language": [
                {
                    "value": "Books::English",
                    "rows": 10,
                    "route_counts": {"representation_verification": 9, "quarantine": 1},
                    "active_risk_counts": {},
                },
                {
                    "value": "Courts::Chinese",
                    "rows": 8,
                    "route_counts": {"quarantine": 8},
                    "active_risk_counts": {"personal_or_secret_data": 8},
                },
            ]
        },
        "receipt_sha256": "b" * 64,
        "training_ready": False,
    }


def _calibration():
    return {
        "schema": "sai-independent-review-comparison-v1",
        "status": "complete_nontraining_independent_review_comparison",
        "receipt_sha256": "c" * 64,
        "training_ready": False,
    }


def test_routes_groups_without_admitting_or_excluding():
    result = build_policy_payload(_census(), _quality(), _calibration())
    by_collection = {row["collection"]: row for row in result["groups"]}
    assert (
        by_collection["Books"]["work_route"]
        == "priority_direct_representation_verification"
    )
    assert by_collection["Courts"]["work_route"] == "hold_high_blocking_signal"
    assert result["totals"] == {
        "rows": 150,
        "token_count": 2200,
        "word_count": 1500,
    }
    assert result["automatic_exclusion"] is False
    assert result["automatic_training_admission"] is False
    assert result["training_ready"] is False


def test_rejects_group_total_drift():
    census = _census()
    census["totals"]["rows"] = 151
    with pytest.raises(PleiasQualityCorePolicyError, match="totals"):
        build_policy_payload(census, _quality(), _calibration())
