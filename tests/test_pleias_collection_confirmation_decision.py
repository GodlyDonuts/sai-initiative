import pytest

from sai.data.pleias_collection_confirmation_decision import (
    PleiasCollectionConfirmationDecisionError,
    build_payload,
)


def _comparison(collection, primary, independent):
    rows = []
    for index, (left, right) in enumerate(zip(primary, independent, strict=True)):
        rows.append(
            {
                "stratum": f"collection::{collection}",
                "primary": {"route": left},
                "reviews": {
                    "gemini-3.5-flash-lite": {"route": right},
                    "aux": None if index else {"route": right},
                },
            }
        )
    return {
        "receipt_sha256": collection,
        "by_provider": {"gemini-3.5-flash-lite": {}},
        "rows": rows,
    }


def test_routes_priority_hold_translation_and_recovery_cells():
    comparisons = [
        _comparison("Priority", ["cleanup_review"] * 8, ["cleanup_review"] * 8),
        _comparison("Hold", ["quarantine"] * 8, ["quarantine"] * 8),
        _comparison(
            "Translate",
            ["factual_grounding_review"] * 8,
            ["translation_review"] * 8,
        ),
        _comparison(
            "Mixed",
            ["quarantine"] * 4 + ["cleanup_review"] * 4,
            ["quarantine"] + ["cleanup_review"] * 7,
        ),
    ]
    result = build_payload(comparisons)
    decisions = {row["collection"]: row["work_route"] for row in result["decisions"]}
    assert decisions == {
        "Hold": "hold_high_blocking_confirmation",
        "Mixed": "targeted_recovery_confirmation",
        "Priority": "priority_targeted_verification",
        "Translate": "translation_or_grounding_adjudication",
    }
    assert result["automatic_training_admission"] is False


def test_rejects_duplicate_comparison_receipt():
    comparison = _comparison(
        "Priority", ["cleanup_review"] * 8, ["cleanup_review"] * 8
    )
    with pytest.raises(
        PleiasCollectionConfirmationDecisionError,
        match="comparison receipt is duplicated",
    ):
        build_payload([comparison, comparison])


def test_rejects_nonexact_collection_coverage():
    comparison = _comparison(
        "Priority", ["cleanup_review"] * 7, ["cleanup_review"] * 7
    )
    with pytest.raises(
        PleiasCollectionConfirmationDecisionError,
        match="collection confirmation coverage differs",
    ):
        build_payload([comparison])
