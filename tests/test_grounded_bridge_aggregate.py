import pytest

from sai.data.grounded_bridge_aggregate import (
    GroundedBridgeAggregateError,
    build_candidate_row,
    validate_receipt,
)
from sai.data.grounded_bridge_labeling import (
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.nous_grounded_bridge_worker import RECEIPT_SCHEMA
from sai.data.token_stream import canonical_sha256
from tests.test_grounded_bridge_labeling import model_output, paired_candidate


def receipt(candidate):
    judgment = normalize_model_judgment(model_output(candidate), candidate)
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "rubric_sha256": RUBRIC_SHA256,
        "requested_model": "stealth/ox-alpha",
        "request_reasoning_effort": "low",
        "attempt_request_sha256s": ["a" * 64],
        "successful_request_sha256": "a" * 64,
        "attempts": [
            {
                "attempt": 1,
                "http_status": 200,
                "outcome": "valid",
                "request_sha256": "a" * 64,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "judgment": judgment,
        "api_key_persisted": False,
        "tools_enabled": False,
        "raw_source_is_training_data": False,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def test_candidate_row_strips_source_quotes_and_remains_unverified() -> None:
    candidate = paired_candidate()
    generated = build_candidate_row(candidate, receipt(candidate))
    assert generated["source_quotes_retained_in_candidate"] is False
    assert generated["claims"][0]["evidence_quote_sha256"]
    assert "evidence_quote" not in generated["claims"][0]
    assert generated["source_disjoint"] is True
    assert generated["independent_claim_verification_complete"] is False
    assert generated["independent_transfer_verification_complete"] is False
    assert generated["training_ready"] is False


def test_receipt_replays_and_rejects_generated_text_tamper() -> None:
    candidate = paired_candidate()
    payload = receipt(candidate)
    assert validate_receipt(payload, candidate) == payload
    payload["judgment"]["representations"][0]["text"] += " Unsupported drift."
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(GroundedBridgeAggregateError, match="receipt"):
        validate_receipt(payload, candidate)


def test_receipt_rejects_identity_tamper_even_if_rehashed() -> None:
    candidate = paired_candidate()
    payload = receipt(candidate)
    payload["candidate_identity_sha256"] = "f" * 64
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(GroundedBridgeAggregateError, match="receipt"):
        validate_receipt(payload, candidate)
