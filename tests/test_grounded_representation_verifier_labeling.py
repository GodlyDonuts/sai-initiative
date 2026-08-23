from __future__ import annotations

import hashlib

import pytest

from sai.data.grounded_representation_verifier_labeling import (
    GroundedRepresentationVerifierError,
    build_messages,
    normalize_candidate,
    normalize_model_judgment,
    repair_evidence_quotes,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    source_quote = (
        "The archive documents geometry as one influence on historical design."
    )
    source_text = source_quote + (" It also records uncertainty and context." * 6)
    generated_text = (
        "The source presents geometry as one documented influence on historical "
        "design while retaining uncertainty about the broader context."
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
            "row_id": "geometry-design",
            "source_url": "https://publicdomainreview.org/collection/geometry/",
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
        "title": "Geometry and design",
        "concepts": ["historical design"],
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
            "The archive documents geometry as one influence on historical design."
        ],
        "representation_evidence_quotes": [
            "The source presents geometry as one documented influence on "
            "historical design"
        ],
        "revision_brief": "",
        "rationale": "The representation is entailed, specific, and uncertainty-aware.",
    }


def test_candidate_and_messages_bind_both_texts() -> None:
    candidate = normalize_candidate(_candidate())
    messages = build_messages(candidate)
    assert len(messages) == 2
    assert "source_document" in messages[1]["content"]
    assert "generated_representation" in messages[1]["content"]


def test_conservative_retain_is_valid() -> None:
    result = normalize_model_judgment(_retain(), _candidate())
    assert result["verdict"] == "retain"
    assert result["independent_request_verification_complete"] is True
    assert result["independent_model_family_verification_complete"] is False
    assert result["training_ready"] is False


def test_retain_with_generic_style_fails_closed() -> None:
    payload = _retain()
    payload["generic_model_style"] = True
    payload["defects"] = ["generic_model_style"]
    with pytest.raises(GroundedRepresentationVerifierError, match="retain"):
        normalize_model_judgment(payload, _candidate())


def test_revision_requires_defect_and_brief() -> None:
    payload = _retain()
    payload["verdict"] = "revise"
    payload["scores"]["source_entailment"] = 3
    payload["defects"] = ["not_entailed"]
    payload["revision_brief"] = "Remove the implication not supported by the source."
    result = normalize_model_judgment(payload, _candidate())
    assert result["verdict"] == "revise"
    payload["revision_brief"] = ""
    with pytest.raises(GroundedRepresentationVerifierError, match="non-retain"):
        normalize_model_judgment(payload, _candidate())


def test_quote_repair_uses_correct_compared_text() -> None:
    payload = _retain()
    payload["source_evidence_quotes"] = [
        "the archive documents geometry as one influence on historical design."
    ]
    repaired, repairs = repair_evidence_quotes(payload, _candidate())
    assert repaired["source_evidence_quotes"][0].startswith("The archive")
    assert repairs[0]["path"] == "source_evidence_quotes[0]"
