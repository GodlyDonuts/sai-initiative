from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from sai.data.nemotron_grounded_bridge_verification_aggregate import (
    REJECTION_SCHEMA,
    RETAINED_SCHEMA,
    REVISION_SCHEMA,
    NemotronBridgeVerificationAggregateError,
    route_candidate,
    validate_receipt,
)
from sai.data.nemotron_grounded_bridge_verifier import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RECEIPT_SCHEMA,
    execute_one,
)
from sai.data.nemotron_grounded_bridge_verifier_labeling import (
    normalize_model_judgment,
)
from sai.data.token_stream import canonical_sha256
from tests.test_grounded_bridge_verifier_labeling import _candidate, _retain


def _judgment(candidate: dict, verdict: str) -> dict:
    payload = _retain(candidate)
    if verdict != "retain":
        payload["verdict"] = verdict
        payload["defects"] = ["superficial_connection"]
        payload["revision_brief"] = (
            "Rebuild the connection around the exact shared structure."
        )
    return normalize_model_judgment(payload, candidate)


def _receipt(verdict: str) -> dict:
    candidate = _candidate()
    judgment = _judgment(candidate, verdict)
    judgment["judgment_sha256"] = "8" * 64
    return {"receipt_sha256": "7" * 64, "judgment": judgment}


def _sealed_nvidia_receipt() -> dict:
    candidate = _candidate()
    payload = _retain(candidate)

    def request_function(**_kwargs):
        return (
            {
                "id": "independent-bridge-verification-1",
                "model": DEFAULT_MODEL,
                "provider": "nvidia",
                "created": 1,
                "choices": [
                    {
                        "message": {"content": json.dumps(payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
            200,
        )

    return execute_one(
        candidate,
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key="nvidia-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )


def test_validate_receipt_accepts_exact_nvidia_binding_then_fails_on_drift() -> None:
    candidate = _candidate()
    receipt = _sealed_nvidia_receipt()
    assert validate_receipt(receipt, candidate)["schema"] == RECEIPT_SCHEMA
    drifted = deepcopy(receipt)
    drifted["endpoint_origin"] = "https://openrouter.ai/api/v1"
    drifted["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in drifted.items() if key != "receipt_sha256"}
    )
    with pytest.raises(NemotronBridgeVerificationAggregateError, match="receipt"):
        validate_receipt(drifted, candidate)


def test_retain_route_strips_anchor_text_and_remains_nontraining() -> None:
    candidate = _candidate()
    route, row = route_candidate(candidate, _receipt("retain"))
    assert route == "retain"
    assert row["schema"] == RETAINED_SCHEMA
    assert row["same_family_retention_passed"] is True
    assert row["independent_family_retention_passed"] is True
    assert row["source_text_persisted"] is False
    assert candidate["anchor_a_text"] not in str(row)
    assert candidate["anchor_b_text"] not in str(row)
    assert row["same_model_family_verification_complete"] is True
    assert row["independent_model_family_verification_complete"] is True
    assert row["benchmark_decontamination_complete"] is False
    assert row["bridge_verified"] is False
    assert row["training_ready"] is False
    seal = row.pop("record_sha256")
    assert seal == canonical_sha256(row)


def test_revision_route_preserves_work_but_not_source_quotes() -> None:
    candidate = _candidate()
    route, row = route_candidate(candidate, _receipt("revise"))
    assert route == "revise"
    assert row["schema"] == REVISION_SCHEMA
    assert row["revision_complete"] is False
    assert row["representations"] == candidate["generated"]["representations"]
    assert row["defects"] == ["superficial_connection"]
    assert candidate["anchor_a_text"] not in str(row)
    assert row["bridge_verified"] is False
    assert row["training_ready"] is False


def test_rejection_route_drops_generated_prose() -> None:
    candidate = _candidate()
    receipt = deepcopy(_receipt("reject"))
    route, row = route_candidate(candidate, receipt)
    assert route == "reject"
    assert row["schema"] == REJECTION_SCHEMA
    assert row["generated_text_persisted"] is False
    assert "representations" not in row
    expected_reason = hashlib.sha256(
        receipt["judgment"]["rationale"].encode()
    ).hexdigest()
    assert row["rejection_reason_sha256"] == expected_reason
    assert candidate["generated_text"] not in str(row)
