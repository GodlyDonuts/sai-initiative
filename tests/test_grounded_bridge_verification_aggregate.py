from __future__ import annotations

from copy import deepcopy

from sai.data.grounded_bridge_verification_aggregate import (
    REJECTION_SCHEMA,
    RETAINED_SCHEMA,
    REVISION_SCHEMA,
    route_candidate,
)
from tests.test_grounded_bridge_verifier_labeling import _candidate, _retain


def _receipt(verdict: str) -> dict:
    candidate = _candidate()
    judgment = _retain(candidate)
    judgment.update(
        {
            "verdict": verdict,
            "judgment_sha256": "8" * 64,
            "confidence_ppm": 900_000,
        }
    )
    if verdict != "retain":
        judgment["defects"] = ["superficial_connection"]
        judgment["revision_brief"] = (
            "Rebuild the connection around the exact shared structure."
        )
    return {"receipt_sha256": "7" * 64, "judgment": judgment}


def test_retain_route_strips_anchor_text_and_remains_nontraining() -> None:
    candidate = _candidate()
    route, row = route_candidate(candidate, _receipt("retain"))
    assert route == "retain"
    assert row["schema"] == RETAINED_SCHEMA
    assert row["same_family_retention_passed"] is True
    assert row["source_text_persisted"] is False
    assert candidate["anchor_a_text"] not in str(row)
    assert candidate["anchor_b_text"] not in str(row)
    assert row["independent_model_family_verification_complete"] is False
    assert row["bridge_verified"] is False
    assert row["training_ready"] is False


def test_revision_route_preserves_work_but_not_source_quotes() -> None:
    candidate = _candidate()
    route, row = route_candidate(candidate, _receipt("revise"))
    assert route == "revise"
    assert row["schema"] == REVISION_SCHEMA
    assert row["revision_complete"] is False
    assert row["representations"] == candidate["generated"]["representations"]
    assert candidate["anchor_a_text"] not in str(row)


def test_rejection_route_drops_generated_prose() -> None:
    candidate = _candidate()
    receipt = _receipt("reject")
    receipt = deepcopy(receipt)
    route, row = route_candidate(candidate, receipt)
    assert route == "reject"
    assert row["schema"] == REJECTION_SCHEMA
    assert row["generated_text_persisted"] is False
    assert "representations" not in row
    assert candidate["generated_text"] not in str(row)
