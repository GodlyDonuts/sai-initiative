from __future__ import annotations

import pytest

from sai.data.bounded_pilot_compiler_aggregate import (
    BoundedPilotCompilerAggregateError,
    combine_rights_and_model_routes,
)
from sai.data.data_compiler_labeling import RISK_KEYS, SCORE_KEYS


def _receipt(verdict: str = "retain", *, enabled_risks: tuple[str, ...] = ()) -> dict:
    return {
        "judgment": {
            "verdict": verdict,
            "source_language": "english",
            "risks": {key: key in enabled_risks for key in RISK_KEYS},
            "scores": {key: 3 for key in SCORE_KEYS},
        }
    }


def _rights(identity: str, source_id: str, route: str, *, observed: bool) -> dict:
    return {
        "identity_sha256": identity,
        "source_id": source_id,
        "adjudication_route": route,
        "expected_license_evidence_observed": observed,
    }


def test_cross_tabs_model_triage_without_overriding_rights() -> None:
    lineage = [
        {
            "retained_document_identity_sha256": "1" * 64,
            "source_id": "pressbooks",
        },
        {
            "retained_document_identity_sha256": "2" * 64,
            "source_id": "pdr",
        },
    ]
    rights = {
        "1"
        * 64: _rights("1" * 64, "pressbooks", "source_access_review", observed=False),
        "2" * 64: _rights("2" * 64, "pdr", "editorial_scope_review", observed=True),
    }
    result = combine_rights_and_model_routes(
        lineage,
        [_receipt(), _receipt(enabled_risks=("weak_source_grounding",))],
        rights,
    )
    assert result["compiler_verdict_by_rights_route"] == {
        "editorial_scope_review": {"retain": 1},
        "source_access_review": {"retain": 1},
    }
    assert result["model_triage_by_rights_route"] == {
        "editorial_scope_review": {"factual_grounding_review": 1},
        "source_access_review": {"representation_verification": 1},
    }
    assert result["rows_with_observed_evidence_and_representation_route"] == 0
    assert result["rights_route_overrides_model_retain_for_admission"] is True
    assert result["joint_route_is_training_admission"] is False


def test_rejects_extra_rights_identity() -> None:
    lineage = [
        {
            "retained_document_identity_sha256": "1" * 64,
            "source_id": "pressbooks",
        }
    ]
    rights = {
        "1" * 64: _rights("1" * 64, "pressbooks", "review", observed=False),
        "2" * 64: _rights("2" * 64, "pressbooks", "review", observed=False),
    }
    with pytest.raises(BoundedPilotCompilerAggregateError, match="coverage"):
        combine_rights_and_model_routes(lineage, [_receipt()], rights)
