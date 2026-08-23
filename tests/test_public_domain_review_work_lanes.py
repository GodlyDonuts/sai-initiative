from __future__ import annotations

import pytest

from sai.data.public_domain_review_scope_audit import SOURCE_ID
from sai.data.public_domain_review_work_lanes import (
    PublicDomainReviewWorkLaneError,
    build_lane_record,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    return {
        "original_candidate_identity_sha256": "1" * 64,
        "record_sha256": "2" * 64,
    }


def _work(*, route: str = "representation_verification") -> dict:
    return {
        "retained_document_identity_sha256": "1" * 64,
        "source_id": SOURCE_ID,
        "candidate_identity_sha256": "3" * 64,
        "compiler_receipt_sha256": "4" * 64,
        "compiler_judgment_sha256": "5" * 64,
        "record_sha256": "6" * 64,
        "compiler_verdict": "retain",
        "content_route": route,
        "content_work_lane": "independent_representation_verification",
        "rights_record_sha256": "7" * 64,
        "rights_route": "review_pdr_scope",
        "expected_license_evidence_observed": True,
        "representation_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }


def test_representation_route_creates_priority_not_admission() -> None:
    row = build_lane_record(_candidate(), _work())
    assert row["representation_priority_candidate"] is True
    assert row["compiler_route_is_verified_admission"] is False
    assert row["representation_verified"] is False
    assert row["training_ready"] is False
    assert row["record_sha256"] == canonical_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )


def test_review_route_is_not_representation_priority() -> None:
    row = build_lane_record(_candidate(), _work(route="quality_review"))
    assert row["representation_priority_candidate"] is False


def test_source_mismatch_fails_closed() -> None:
    work = _work()
    work["source_id"] = "another_source"
    with pytest.raises(PublicDomainReviewWorkLaneError, match="work lane identity"):
        build_lane_record(_candidate(), work)
