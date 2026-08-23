from __future__ import annotations

import json

import pytest

from sai.data.bounded_pilot_compiler_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.bounded_pilot_work_queue import (
    BoundedPilotWorkQueueError,
    _load_aggregate,
    build_records,
)
from sai.data.data_compiler_labeling import RISK_KEYS, SCORE_KEYS
from sai.data.token_stream import canonical_sha256


def _receipt(identity: str, *, risks: tuple[str, ...] = ()) -> dict:
    judgment = {
        "judgment_sha256": "2" * 64,
        "verdict": "retain",
        "epistemic_functions": ["reality_anchor"],
        "source_language": "english",
        "risks": {key: key in risks for key in RISK_KEYS},
        "scores": {key: 3 for key in SCORE_KEYS},
    }
    return {
        "candidate_identity_sha256": identity,
        "receipt_sha256": "3" * 64,
        "judgment": judgment,
    }


def _rights(identity: str, source_id: str, *, observed: bool = True) -> dict:
    return {
        "identity_sha256": identity,
        "source_id": source_id,
        "record_sha256": "4" * 64,
        "adjudication_route": "review_source_scope",
        "expected_license_evidence_observed": observed,
    }


def _aggregate() -> dict:
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_joint_evidence",
        "compiler_judgments_are_verified_admissions": False,
        "independent_representation_verification_complete": False,
        "rights_provenance_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def test_build_records_preserves_independent_content_and_rights_lanes() -> None:
    retained = "1" * 64
    records = build_records(
        [
            {
                "retained_document_identity_sha256": retained,
                "source_id": "pressbooks",
            }
        ],
        [_receipt("0" * 64, risks=("weak_source_grounding",))],
        {retained: _rights(retained, "pressbooks")},
    )
    assert len(records) == 1
    record = records[0]
    assert record["content_route"] == "factual_grounding_review"
    assert record["content_work_lane"] == "source_grounding_verification"
    assert record["rights_route"] == "review_source_scope"
    assert record["expected_license_evidence_observed"] is True
    assert record["content_and_rights_lanes_are_independent"] is True
    assert record["model_retain_overrides_content_or_rights_lane"] is False
    assert record["training_ready"] is False
    assert record["record_sha256"] == canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )


def test_build_records_rejects_extra_rights_identity() -> None:
    retained = "1" * 64
    extra = "2" * 64
    with pytest.raises(BoundedPilotWorkQueueError, match="rights coverage"):
        build_records(
            [
                {
                    "retained_document_identity_sha256": retained,
                    "source_id": "pressbooks",
                }
            ],
            [_receipt("0" * 64)],
            {
                retained: _rights(retained, "pressbooks"),
                extra: _rights(extra, "pressbooks"),
            },
        )


def test_build_records_rejects_source_mismatch() -> None:
    retained = "1" * 64
    with pytest.raises(BoundedPilotWorkQueueError, match="identity"):
        build_records(
            [
                {
                    "retained_document_identity_sha256": retained,
                    "source_id": "pressbooks",
                }
            ],
            [_receipt("0" * 64)],
            {retained: _rights(retained, "another_source")},
        )


def test_load_aggregate_replays_self_hash(tmp_path) -> None:
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(_aggregate()) + "\n")
    assert _load_aggregate(path) == _aggregate()


def test_load_aggregate_rejects_tamper(tmp_path) -> None:
    path = tmp_path / "aggregate.json"
    payload = _aggregate()
    payload["training_ready"] = True
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(BoundedPilotWorkQueueError, match="aggregate receipt"):
        _load_aggregate(path)
