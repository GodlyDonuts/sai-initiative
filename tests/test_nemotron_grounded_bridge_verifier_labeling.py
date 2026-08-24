from __future__ import annotations

import json
from copy import deepcopy

import pytest

from sai.data.grounded_bridge_verifier_labeling import (
    JUDGMENT_SCHEMA as SAME_FAMILY_JUDGMENT_SCHEMA,
)
from sai.data.grounded_bridge_verifier_labeling import (
    RUBRIC_SHA256 as SAME_FAMILY_RUBRIC_SHA256,
)
from sai.data.nemotron_grounded_bridge_verifier_labeling import (
    INDEPENDENT_RUBRIC_SHA256,
    JUDGMENT_SCHEMA,
    NemotronBridgeVerifierError,
    build_messages,
    normalize_model_judgment,
    validation_hint,
)
from sai.data.token_stream import canonical_sha256
from tests.test_grounded_bridge_verifier_labeling import _candidate, _retain


def test_validation_hints_preserve_the_strict_verifier_contract() -> None:
    assert "exactly one object" in validation_hint(
        "bridge claim-check coverage differs"
    )
    assert "byte-for-byte" in validation_hint(
        "bridge claim evidence is not exact"
    )
    assert "verdict=retain" in validation_hint("retained bridge is inconsistent")
    assert validation_hint("unrecognized validation error") == ""


def test_messages_bind_prior_contract_and_all_hashes() -> None:
    candidate = _candidate()
    messages = build_messages(candidate)
    assert len(messages) == 2
    envelope = json.loads(messages[1]["content"])
    assert "anchor_a" in envelope
    assert "anchor_b" in envelope
    assert "generated_bridge" in envelope
    assert envelope["same_family_rubric_sha256"] == SAME_FAMILY_RUBRIC_SHA256
    assert envelope["same_family_judgment_schema"] == SAME_FAMILY_JUDGMENT_SCHEMA
    assert envelope["independent_rubric_sha256"] == INDEPENDENT_RUBRIC_SHA256
    assert INDEPENDENT_RUBRIC_SHA256 != SAME_FAMILY_RUBRIC_SHA256
    for key, value in envelope["bindings"].items():
        assert value == candidate[key]


def test_conservative_retain_is_independent_family_only_and_nontraining() -> None:
    candidate = _candidate()
    result = normalize_model_judgment(_retain(candidate), candidate)
    assert result["schema"] == JUDGMENT_SCHEMA
    assert result["same_family_rubric_sha256"] == SAME_FAMILY_RUBRIC_SHA256
    assert result["same_family_judgment_schema"] == SAME_FAMILY_JUDGMENT_SCHEMA
    assert result["rubric_sha256"] == INDEPENDENT_RUBRIC_SHA256
    for key in (
        "candidate_identity_sha256",
        "pair_identity_sha256",
        "anchor_a_source_content_sha256",
        "anchor_a_candidate_identity_sha256",
        "anchor_b_source_content_sha256",
        "anchor_b_candidate_identity_sha256",
        "generated_text_sha256",
        "generated_candidate_identity_sha256",
        "generator_receipt_sha256",
        "generator_judgment_sha256",
    ):
        assert result[key] == candidate[key]
    assert result["verdict"] == "retain"
    assert result["same_model_family_as_generator"] is False
    assert result["independent_request_verification_complete"] is True
    assert result["independent_model_family_verification_complete"] is True
    assert result["bridge_verified"] is False
    assert result["training_ready"] is False
    unsigned = {key: value for key, value in result.items() if key != "judgment_sha256"}
    assert result["judgment_sha256"] == canonical_sha256(unsigned)


def test_nonliteral_claim_evidence_fails_closed() -> None:
    candidate = _candidate()
    payload = _retain(candidate)
    payload["claim_checks"][0]["evidence_quote"] = "invented evidence"
    with pytest.raises(NemotronBridgeVerifierError, match="not exact"):
        normalize_model_judgment(payload, candidate)


def test_missing_claim_coverage_fails_closed() -> None:
    candidate = _candidate()
    payload = deepcopy(_retain(candidate))
    payload["claim_checks"] = payload["claim_checks"][:-1]
    with pytest.raises(NemotronBridgeVerifierError, match="coverage"):
        normalize_model_judgment(payload, candidate)


def test_retain_with_failed_transfer_check_fails_closed() -> None:
    candidate = _candidate()
    payload = deepcopy(_retain(candidate))
    payload["worked_transfer_problem_correct"] = False
    payload["defects"] = ["incorrect_transfer_solution"]
    with pytest.raises(NemotronBridgeVerifierError, match="inconsistent"):
        normalize_model_judgment(payload, candidate)


def test_rehashed_anchor_tamper_fails_closed() -> None:
    candidate = deepcopy(_candidate())
    candidate["anchor_a_text"] += " Tampered."
    candidate["candidate_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in candidate.items()
            if key != "candidate_identity_sha256"
        }
    )
    with pytest.raises(RuntimeError, match="differs"):
        build_messages(candidate)
