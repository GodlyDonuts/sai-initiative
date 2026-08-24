from __future__ import annotations

from copy import deepcopy
import json

import pytest

from sai.data.nemotron_grounded_bridge_verifier import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RECEIPT_SCHEMA,
    execute_one,
)
from sai.data.nemotron_grounded_bridge_verifier_labeling import (
    NemotronBridgeVerifierError,
    build_messages,
    normalize_model_judgment,
    repair_evidence_quotes,
)
from tests.test_grounded_bridge_verifier_labeling import _candidate, _retain


def test_execute_one_seals_independent_family_bridge_verification() -> None:
    candidate = _candidate()
    payload = _retain(candidate)
    seen = {}

    def request_function(**kwargs):
        seen.update(kwargs)
        return (
            {
                "id": "independent-bridge-verification-1",
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
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

    result = execute_one(
        candidate,
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key="nvidia-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert seen["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert seen["body"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert "reasoning" not in seen["body"]
    assert result["schema"] == RECEIPT_SCHEMA
    assert result["requested_model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert result["endpoint_origin"] == "https://integrate.api.nvidia.com/v1"
    assert result["credential_transport"] == "direct_portal_bearer"
    assert result["judgment"] == normalize_model_judgment(payload, candidate)
    assert result["judgment"]["independent_request_verification_complete"] is True
    assert result["judgment"]["independent_model_family_verification_complete"] is True
    assert result["judgment"]["bridge_verified"] is False
    assert result["training_ready"] is False


def test_bridge_evidence_repair_recovers_unique_literal_spans() -> None:
    candidate = _candidate()
    payload = deepcopy(_retain(candidate))
    literal = candidate["anchor_a_text"]
    payload["anchor_a_evidence_quotes"] = [literal.upper()]
    for check in payload["claim_checks"]:
        if check["anchor_side"] == "A":
            check["evidence_quote"] = literal.upper()
    repaired, repairs = repair_evidence_quotes(payload, candidate)
    assert repaired["anchor_a_evidence_quotes"] == [literal]
    assert all(
        check["evidence_quote"] == literal
        for check in repaired["claim_checks"]
        if check["anchor_side"] == "A"
    )
    assert any(row["path"] == "anchor_a_evidence_quotes[0]" for row in repairs)
    assert normalize_model_judgment(repaired, candidate)["verdict"] == "retain"


def test_bridge_messages_offer_only_literal_evidence_candidates() -> None:
    candidate = _candidate()
    envelope = json.loads(build_messages(candidate)[1]["content"])
    for key, source_key in (
        ("anchor_a_evidence_quote_candidates", "anchor_a_text"),
        ("anchor_b_evidence_quote_candidates", "anchor_b_text"),
        ("generated_evidence_quote_candidates", "generated_text"),
    ):
        assert envelope[key]
        assert all(quote in candidate[source_key] for quote in envelope[key])


def test_bridge_evidence_repair_rejects_invented_quote() -> None:
    candidate = _candidate()
    payload = deepcopy(_retain(candidate))
    payload["anchor_a_evidence_quotes"] = ["invented source evidence"]
    with pytest.raises(RuntimeError, match="no exact source span"):
        repair_evidence_quotes(payload, candidate)


def test_execute_one_seals_quote_repair_receipt() -> None:
    candidate = _candidate()
    payload = deepcopy(_retain(candidate))
    literal = candidate["anchor_a_text"]
    payload["anchor_a_evidence_quotes"] = [literal.upper()]

    def request_function(**_kwargs):
        return (
            {
                "id": "independent-bridge-verification-repair-1",
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

    result = execute_one(
        candidate,
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key="nvidia-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert result["judgment"]["anchor_a_evidence_quotes"] == [literal]
    assert result["deterministic_evidence_quote_repairs"][0]["path"] == (
        "anchor_a_evidence_quotes[0]"
    )
