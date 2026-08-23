from __future__ import annotations

from copy import deepcopy

import pytest

from sai.data.grounded_bridge_aggregate import build_candidate_row
from sai.data.grounded_bridge_verification_population import build_candidate
from sai.data.grounded_bridge_verifier_labeling import (
    GroundedBridgeVerifierError,
    build_messages,
    normalize_model_judgment,
)
from sai.data.token_stream import canonical_sha256
from tests.test_grounded_bridge_aggregate import receipt
from tests.test_grounded_bridge_verification_population import _bound_pair


def _candidate() -> dict:
    pair = _bound_pair()
    generator = receipt(pair)
    generated = build_candidate_row(pair, generator)
    return build_candidate(pair, generated, generator)


def _retain(candidate: dict) -> dict:
    return {
        "verdict": "retain",
        "claim_checks": [
            {
                "claim_index": index,
                "anchor_side": claim["anchor_side"],
                "supported": True,
                "evidence_quote": candidate[
                    "anchor_a_text" if claim["anchor_side"] == "A" else "anchor_b_text"
                ],
                "rationale": "The assigned source anchor contains the exact evidence.",
            }
            for index, claim in enumerate(candidate["generated"]["claims"])
        ],
        "shared_structure_supported": True,
        "domain_connection_substantive": True,
        "worked_transfer_problem_correct": True,
        "counterexample_valid": True,
        "analogy_limits_adequate": True,
        "unsupported_generated_claims": [],
        "defects": [],
        "anchor_a_evidence_quotes": [candidate["anchor_a_text"]],
        "anchor_b_evidence_quotes": [candidate["anchor_b_text"]],
        "generated_evidence_quotes": [candidate["generated"]["bridge_thesis"]],
        "revision_brief": "",
        "confidence_ppm": 900_000,
        "rationale": "Every claim and transfer component is grounded and coherent.",
    }


def test_messages_bind_both_sources_and_generated_bridge() -> None:
    messages = build_messages(_candidate())
    assert len(messages) == 2
    assert "anchor_a" in messages[1]["content"]
    assert "anchor_b" in messages[1]["content"]
    assert "generated_bridge" in messages[1]["content"]


def test_conservative_retain_is_same_family_only_and_nontraining() -> None:
    candidate = _candidate()
    result = normalize_model_judgment(_retain(candidate), candidate)
    assert result["verdict"] == "retain"
    assert result["independent_request_verification_complete"] is True
    assert result["independent_model_family_verification_complete"] is False
    assert result["bridge_verified"] is False
    assert result["training_ready"] is False


def test_nonliteral_claim_evidence_fails_closed() -> None:
    candidate = _candidate()
    payload = _retain(candidate)
    payload["claim_checks"][0]["evidence_quote"] = "invented evidence"
    with pytest.raises(GroundedBridgeVerifierError, match="not exact"):
        normalize_model_judgment(payload, candidate)


def test_retain_with_failed_transfer_check_fails_closed() -> None:
    candidate = _candidate()
    payload = deepcopy(_retain(candidate))
    payload["worked_transfer_problem_correct"] = False
    payload["defects"] = ["incorrect_transfer_solution"]
    with pytest.raises(GroundedBridgeVerifierError, match="inconsistent"):
        normalize_model_judgment(payload, candidate)


def test_rehashed_generated_object_tamper_fails_closed() -> None:
    candidate = deepcopy(_candidate())
    candidate["generated"]["representations"][0]["text"] += " Tampered."
    candidate["candidate_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in candidate.items()
            if key != "candidate_identity_sha256"
        }
    )
    with pytest.raises(GroundedBridgeVerifierError, match="candidate"):
        build_messages(candidate)
