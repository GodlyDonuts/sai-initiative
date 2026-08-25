from __future__ import annotations

import hashlib

import pytest

from sai.data.grounded_representation_labeling import (
    GroundedRepresentationError,
    build_messages,
    normalize_candidate,
    normalize_model_judgment,
    repair_evidence_quotes,
    validation_hint,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    quote = (
        "The archive connects the history of pigments with chemistry and "
        "artistic practice."
    )
    text = quote + (" It distinguishes documented evidence from later legend. " * 5)
    row = {
        "schema": "sai-public-domain-review-representation-candidate-v1",
        "text": text,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": "1" * 64,
        "original_candidate_identity_sha256": "2" * 64,
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "pigment-history",
            "source_url": "https://publicdomainreview.org/collection/pigment-history/",
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
            "requested_representations": [
                "conceptual_summary",
                "concise_reference",
            ],
            "domains": ["history", "chemistry_materials"],
            "subdomains": ["pigment history"],
            "concepts_taught": ["historical pigments"],
            "prerequisites_assumed": ["material properties"],
            "cross_domain_bridges": ["history::chemistry_materials"],
            "difficulty": 2,
            "curriculum_phase": "integration",
        },
        "compiler_route_is_verified_admission": False,
        "representation_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _payload() -> dict:
    quote = (
        "The archive connects the history of pigments with chemistry and "
        "artistic practice."
    )
    return {
        "representations": [
            {
                "type": "conceptual_summary",
                "title": "Pigments as historical materials",
                "text": (
                    "Pigment history joins documented cultural practice to the "
                    "material behavior studied by chemistry."
                ),
                "evidence_quotes": [quote],
                "concepts": ["historical pigments"],
                "difficulty": 1,
            },
            {
                "type": "concise_reference",
                "title": "Pigment history reference",
                "text": (
                    "A source-grounded reference should separate documented "
                    "evidence about pigments from later legend and attribution."
                ),
                "evidence_quotes": [
                    "It distinguishes documented evidence from later legend."
                ],
                "concepts": ["source criticism"],
                "difficulty": 2,
            },
        ],
        "prerequisite_edges": [
            {
                "prerequisite": "material properties",
                "concept": "historical pigments",
                "relation": "helpful_before",
                "evidence_quotes": [quote],
            }
        ],
        "cross_domain_bridge_candidates": [
            {
                "bridge_label": "history::chemistry_materials",
                "connection": (
                    "The historical use of pigments may be compared with their "
                    "material chemistry after an external chemistry anchor is added."
                ),
                "source_evidence_quotes": [quote],
                "external_anchor_required": True,
            }
        ],
        "coverage_note": "The representations cover the document's central claim.",
    }


def test_candidate_and_messages_preserve_rights_and_exact_plan() -> None:
    candidate = normalize_candidate(_candidate())
    messages = build_messages(candidate)
    assert len(messages) == 2
    assert "CC-BY-SA-4.0" in messages[1]["content"]
    assert '"requested_representation_types"' in messages[1]["content"]


def test_normalize_grounded_representations() -> None:
    result = normalize_model_judgment(_payload(), _candidate())
    assert [row["type"] for row in result["representations"]] == [
        "conceptual_summary",
        "concise_reference",
    ]
    assert result["external_bridge_anchors_verified"] is False
    assert result["share_alike_required"] is True
    assert result["training_ready"] is False


def test_nested_quote_repair_returns_literal_source_span() -> None:
    payload = _payload()
    payload["representations"][0]["evidence_quotes"] = [
        "the archive connects the history of pigments with chemistry and "
        "artistic practice."
    ]
    repaired, repairs = repair_evidence_quotes(payload, _candidate())
    assert repaired["representations"][0]["evidence_quotes"] == [
        "The archive connects the history of pigments with chemistry and "
        "artistic practice."
    ]
    assert repairs[0]["path"] == "representations[0].evidence_quotes[0]"


def test_validation_hint_names_exact_prerequisite_relations() -> None:
    hint = validation_hint("prerequisite edge differs")
    assert "required_before" in hint
    assert "helpful_before" in hint
    assert "revisited_with" in hint


def test_validation_hint_preserves_requested_representation_order() -> None:
    assert "supplied order" in validation_hint("representation order differs")


def test_validation_hint_pins_representation_concept_geometry() -> None:
    hint = validation_hint("representation concepts differs")
    assert "concepts to a JSON list" in hint
    assert "1..8 unique, nonempty, lowercase strings" in hint


def test_rejects_representation_order_drift() -> None:
    payload = _payload()
    payload["representations"].reverse()
    with pytest.raises(GroundedRepresentationError, match="order"):
        normalize_model_judgment(payload, _candidate())


def test_rejects_unanchored_external_bridge() -> None:
    payload = _payload()
    payload["cross_domain_bridge_candidates"][0]["external_anchor_required"] = False
    with pytest.raises(GroundedRepresentationError, match="bridge candidate"):
        normalize_model_judgment(payload, _candidate())
