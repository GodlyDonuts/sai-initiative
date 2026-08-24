from __future__ import annotations

import json

import pytest

from sai.data.grounded_representation_verifier_labeling import (
    GroundedRepresentationVerifierError,
)
from sai.data.nemotron_grounded_representation_verifier import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RECEIPT_SCHEMA,
    execute_one,
)
from sai.data.nemotron_grounded_representation_verifier_labeling import (
    build_messages,
    normalize_model_judgment,
    validation_hint,
)
from tests.test_nous_grounded_representation_verifier import _candidate


def _retain() -> dict:
    return {
        "verdict": "retain",
        "scores": {
            "source_entailment": 4,
            "factual_fidelity": 4,
            "pedagogical_value": 4,
            "linguistic_quality": 4,
            "cultural_fidelity": 3,
            "uncertainty_fidelity": 4,
        },
        "external_claims_present": False,
        "source_uncertainty_preserved": True,
        "cultural_specificity_preserved": True,
        "generic_model_style": False,
        "excessive_source_copying": False,
        "defects": [],
        "source_evidence_quotes": [
            "The source records uncertainty about a historical technique."
        ],
        "representation_evidence_quotes": [
            "The source records uncertainty about the historical technique"
        ],
        "revision_brief": "",
        "rationale": "The generated text is faithful and preserves uncertainty.",
    }


def test_independent_messages_and_judgment_bind_model_family() -> None:
    messages = build_messages(_candidate())
    assert "independently verify" in messages[0]["content"]
    result = normalize_model_judgment(_retain(), _candidate())
    assert result["same_model_family_as_generator"] is False
    assert result["independent_request_verification_complete"] is True
    assert result["independent_model_family_verification_complete"] is True
    assert result["training_ready"] is False


def test_independent_retain_remains_fail_closed() -> None:
    payload = _retain()
    payload["scores"]["factual_fidelity"] = 3
    with pytest.raises(GroundedRepresentationVerifierError, match="retain"):
        normalize_model_judgment(payload, _candidate())
    assert "source_entailment=4" in validation_hint("retain conditions differs")


def test_execute_one_seals_independent_representation_receipt() -> None:
    payload = _retain()
    seen = {}

    def request_function(**kwargs):
        seen.update(kwargs)
        return (
            {
                "id": "independent-representation-verification-1",
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

    receipt = execute_one(
        _candidate(),
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key="nvidia-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert seen["body"]["model"] == DEFAULT_MODEL
    assert "reasoning" not in seen["body"]
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["credential_transport"] == "direct_portal_bearer"
    assert receipt["judgment"] == normalize_model_judgment(payload, _candidate())
    assert receipt["training_ready"] is False
