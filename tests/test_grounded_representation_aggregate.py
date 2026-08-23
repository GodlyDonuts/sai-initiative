from __future__ import annotations

import hashlib

import pytest

from sai.data.grounded_representation_aggregate import (
    GroundedRepresentationAggregateError,
    build_candidate_rows,
    validate_receipt,
)
from sai.data.grounded_representation_labeling import (
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    quote = "The archive documents a relationship between design and engineering."
    text = quote + (" It preserves historical context and uncertainty." * 6)
    row = {
        "schema": "sai-public-domain-review-representation-candidate-v1",
        "text": text,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": "1" * 64,
        "original_candidate_identity_sha256": "2" * 64,
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "design-engineering",
            "source_url": "https://publicdomainreview.org/collection/design/",
            "source_type": "collection",
            "license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
        },
        "compiler": {
            "candidate_identity_sha256": "3" * 64,
            "receipt_sha256": "4" * 64,
            "judgment_sha256": "5" * 64,
            "work_record_sha256": "6" * 64,
            "content_route": "representation_verification",
            "rights_route": "editorial_scope_review",
            "verdict": "retain",
            "preservation_policy": "preserve_plus_derivatives",
            "requested_representations": ["conceptual_summary"],
            "domains": ["architecture_design", "engineering"],
            "subdomains": ["design history"],
            "concepts_taught": ["historical design"],
            "prerequisites_assumed": [],
            "cross_domain_bridges": ["architecture_design::engineering"],
            "difficulty": 1,
            "curriculum_phase": "integration",
        },
        "compiler_route_is_verified_admission": False,
        "representation_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _judgment(candidate: dict) -> dict:
    quote = "The archive documents a relationship between design and engineering."
    return normalize_model_judgment(
        {
            "representations": [
                {
                    "type": "conceptual_summary",
                    "title": "Design and engineering in historical context",
                    "text": (
                        "The document presents design and engineering as related "
                        "practices whose historical context should be retained."
                    ),
                    "evidence_quotes": [quote],
                    "concepts": ["historical design"],
                    "difficulty": 1,
                }
            ],
            "prerequisite_edges": [],
            "cross_domain_bridge_candidates": [
                {
                    "bridge_label": "architecture_design::engineering",
                    "connection": (
                        "The relationship can be tested against a separate "
                        "engineering source before becoming a verified bridge."
                    ),
                    "source_evidence_quotes": [quote],
                    "external_anchor_required": True,
                }
            ],
            "coverage_note": "The representation covers the central relation.",
        },
        candidate,
    )


def _receipt(candidate: dict) -> dict:
    receipt = {
        "schema": "sai-nous-grounded-representation-receipt-v1",
        "status": "complete",
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "rubric_sha256": RUBRIC_SHA256,
        "requested_model": "stealth/ox-alpha",
        "request_reasoning_effort": "low",
        "api_key_persisted": False,
        "tools_enabled": False,
        "raw_source_is_training_data": False,
        "training_ready": False,
        "judgment": _judgment(candidate),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def test_build_candidate_rows_excludes_source_quote_text() -> None:
    candidate = _candidate()
    receipt = _receipt(candidate)
    representations, edges, bridges = build_candidate_rows(candidate, receipt)
    assert len(representations) == 1
    assert edges == []
    assert len(bridges) == 1
    assert "evidence_quotes" not in representations[0]
    assert representations[0]["evidence_quote_sha256s"]
    assert representations[0]["source_claims_independently_verified"] is False
    assert bridges[0]["external_anchor_verified"] is False


def test_validate_receipt_replays_normalized_judgment() -> None:
    candidate = _candidate()
    receipt = _receipt(candidate)
    assert validate_receipt(receipt, candidate) == receipt


def test_validate_receipt_rejects_generated_text_tamper() -> None:
    candidate = _candidate()
    receipt = _receipt(candidate)
    receipt["judgment"]["representations"][0]["text"] += " Fabricated drift."
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(GroundedRepresentationAggregateError, match="receipt"):
        validate_receipt(receipt, candidate)
