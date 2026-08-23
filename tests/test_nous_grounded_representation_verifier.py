from __future__ import annotations

import hashlib
import json

from sai.data.nous_grounded_representation_verifier import (
    RECEIPT_SCHEMA,
    execute_one,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    source_quote = "The source records uncertainty about a historical technique."
    source_text = source_quote + (" It preserves context for later readers." * 6)
    generated_text = (
        "The source records uncertainty about the historical technique and "
        "preserves its context for readers."
    )
    row = {
        "schema": "sai-grounded-representation-verification-candidate-v1",
        "source_text": source_text,
        "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "generated_text": generated_text,
        "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
        "source_evidence_quotes": [source_quote],
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "historical-technique",
            "source_url": "https://publicdomainreview.org/collection/technique/",
            "source_type": "collection",
            "license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
        },
        "source_candidate_identity_sha256": "1" * 64,
        "generated_record_sha256": "2" * 64,
        "clean_record_sha256": "3" * 64,
        "generator_receipt_sha256": "4" * 64,
        "generator_judgment_sha256": "5" * 64,
        "representation_index": 0,
        "representation_type": "conceptual_summary",
        "title": "A historical technique",
        "concepts": ["historical technique"],
        "difficulty": 1,
        "benchmark_decontamination_complete": True,
        "same_model_family_as_generator": True,
        "independent_request_verification_complete": False,
        "independent_model_family_verification_complete": False,
        "representation_verified": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def test_execute_one_seals_same_family_independent_request() -> None:
    payload = {
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

    def request_function(**_kwargs):
        return (
            {
                "id": "verification-1",
                "model": "stealth/ox-alpha",
                "provider": "nous",
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
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="loopback-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["judgment"]["verdict"] == "retain"
    assert receipt["judgment"]["independent_request_verification_complete"] is True
    assert (
        receipt["judgment"]["independent_model_family_verification_complete"] is False
    )
    assert receipt["training_ready"] is False
