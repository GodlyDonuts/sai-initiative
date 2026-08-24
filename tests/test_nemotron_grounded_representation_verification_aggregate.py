from __future__ import annotations

from sai.data.nemotron_grounded_representation_verification_aggregate import (
    REJECTION_SCHEMA,
    RETAINED_SCHEMA,
    REVISION_SCHEMA,
    route_candidate,
)
from tests.test_grounded_representation_verification_aggregate import (
    _candidate,
    _receipt,
)


def _independent_receipt(verdict: str) -> dict:
    receipt = _receipt(verdict)
    receipt["receipt_sha256"] = "9" * 64
    receipt["judgment"]["judgment_sha256"] = "a" * 64
    return receipt


def test_cross_model_retain_requires_two_retains() -> None:
    route, row = route_candidate(
        _candidate(), _receipt("retain"), _independent_receipt("retain")
    )
    assert route == "retain"
    assert row["schema"] == RETAINED_SCHEMA
    assert row["cross_model_retention_passed"] is True
    assert row["representation_verified"] is True
    assert row["source_claims_independently_verified"] is False
    assert row["training_ready"] is False


def test_cross_model_disagreement_routes_to_revision() -> None:
    route, row = route_candidate(
        _candidate(), _receipt("retain"), _independent_receipt("revise")
    )
    assert route == "revise"
    assert row["schema"] == REVISION_SCHEMA
    assert row["representation_verified"] is False
    assert row["text"] == _candidate()["generated_text"]


def test_either_rejection_removes_generated_text() -> None:
    route, row = route_candidate(
        _candidate(), _receipt("revise"), _independent_receipt("reject")
    )
    assert route == "reject"
    assert row["schema"] == REJECTION_SCHEMA
    assert row["generated_text_persisted"] is False
    assert "text" not in row
