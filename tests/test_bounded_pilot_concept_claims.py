from __future__ import annotations

import pytest

from sai.data.bounded_pilot_concept_claims import (
    BoundedPilotConceptClaimError,
    build_claim,
    normalize_label,
    summarize_claims,
)
from sai.data.token_stream import canonical_sha256


def _claim(identity: str, *, source_id: str = "pdr") -> dict:
    source = {
        "retained_document_identity_sha256": "9" * 64,
        "source_id": source_id,
    }
    judgment = {
        "judgment_sha256": "2" * 64,
        "curriculum_phase": "grounding",
        "difficulty": 2,
        "prerequisite_burden": 1,
        "domains": ["mathematics", "natural_sciences"],
        "subdomains": ["Geometry", "geometry"],
        "epistemic_functions": ["reality_anchor"],
        "concepts_taught": ["Triangles", "shared"],
        "prerequisites_assumed": ["Algebra", "shared"],
        "cross_domain_bridges": ["mathematics::natural_sciences"],
    }
    receipt = {
        "candidate_identity_sha256": identity,
        "receipt_sha256": "3" * 64,
        "judgment": judgment,
    }
    work = {
        "candidate_identity_sha256": identity,
        "retained_document_identity_sha256": source[
            "retained_document_identity_sha256"
        ],
        "source_id": source_id,
        "compiler_receipt_sha256": receipt["receipt_sha256"],
        "compiler_judgment_sha256": judgment["judgment_sha256"],
        "record_sha256": "4" * 64,
        "content_route": "representation_verification",
        "rights_route": "review_pdr_scope",
        "expected_license_evidence_observed": True,
    }
    return build_claim(source, receipt, work)


def test_build_claim_normalizes_labels_without_claiming_verification() -> None:
    claim = _claim("1" * 64)
    assert claim["domains"] == ["mathematics", "natural_sciences"]
    assert claim["subdomains"] == ["geometry"]
    assert claim["concepts_taught"] == ["shared", "triangles"]
    assert claim["model_annotations_independently_verified"] is False
    assert claim["semantic_edges_verified"] is False
    assert claim["training_ready"] is False
    assert claim["claim_sha256"] == canonical_sha256(
        {key: value for key, value in claim.items() if key != "claim_sha256"}
    )


def test_summarize_claims_preserves_support_and_skips_self_edge() -> None:
    first = _claim("1" * 64)
    second = _claim("2" * 64, source_id="pressbooks")
    second["claim_sha256"] = canonical_sha256(
        {key: value for key, value in second.items() if key != "claim_sha256"}
    )
    nodes, edges, bridges = summarize_claims([first, second])
    algebra = next(row for row in nodes if row["label"] == "algebra")
    assert algebra["assumed_candidate_identity_sha256s"] == ["1" * 64, "2" * 64]
    assert {row["source_ids"][0] for row in edges} <= {"pdr", "pressbooks"}
    assert not any(row["prerequisite_label"] == row["concept_label"] for row in edges)
    edge = next(
        row
        for row in edges
        if row["prerequisite_label"] == "algebra"
        and row["concept_label"] == "triangles"
    )
    assert edge["supporting_candidate_identity_sha256s"] == ["1" * 64, "2" * 64]
    assert edge["pairing_inferred_from_document_level_cooccurrence"] is True
    assert edge["semantic_edge_verified"] is False
    assert len(bridges) == 1
    assert bridges[0]["bridge_claim"] == "mathematics::natural_sciences"


def test_label_normalization_rejects_control_characters() -> None:
    with pytest.raises(BoundedPilotConceptClaimError, match="concept label"):
        normalize_label("algebra\x00geometry")
