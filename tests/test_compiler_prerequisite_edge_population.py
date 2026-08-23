from __future__ import annotations

import hashlib

import pytest

from sai.data.compiler_prerequisite_edge_population import (
    CompilerPrerequisiteEdgePopulationError,
    build_edge_plan,
    judgment_qualifies,
)
from sai.data.token_stream import canonical_sha256


def _candidate(index: int, text: str) -> dict:
    row = {
        "schema": "sai-agent-data-candidate-v1",
        "text": text,
        "source": {
            "dataset": "unit/source",
            "revision": "immutable",
            "row_id": str(index),
            "license": "test",
            "source_type": "reference",
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(f"provenance-{index}".encode()).hexdigest(),
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _judgment(index: int, prerequisite: str, concept: str, domain: str) -> dict:
    quote = f"Document {index} explains {concept} after assuming {prerequisite}."
    row = {
        "verdict": "retain",
        "source_language": "english",
        "confidence_ppm": 900_000,
        "scores": {"source_reliability": 4, "educational_value": 4},
        "risks": {
            "seo_or_content_farm": False,
            "incoherent_or_corrupted": False,
            "factual_unreliability": False,
            "answer_farm_without_teaching": False,
            "personal_or_secret_data": False,
            "weak_source_grounding": False,
            "license_or_provenance_unclear": False,
        },
        "domains": [domain],
        "concepts_taught": [concept],
        "prerequisites_assumed": [prerequisite],
        "evidence_quotes": [quote],
    }
    row["judgment_sha256"] = canonical_sha256(row)
    return row


def _anchor(index: int, prerequisite: str, concept: str, domain: str) -> dict:
    judgment = _judgment(index, prerequisite, concept, domain)
    return {
        "candidate": _candidate(index, judgment["evidence_quotes"][0]),
        "judgment": judgment,
    }


def test_selects_only_repeated_source_disjoint_edges() -> None:
    anchors = [
        _anchor(1, "basic arithmetic", "unit conversion", "mathematics"),
        _anchor(2, "basic arithmetic", "unit conversion", "mathematics"),
        _anchor(3, "newton's laws", "torque", "physics_astronomy"),
        _anchor(4, "newton's laws", "torque", "physics_astronomy"),
        _anchor(5, "one-off premise", "one-off concept", "history"),
    ]
    rows = build_edge_plan(anchors, target_edges=2, seed=7)
    assert {(row["prerequisite"], row["concept"]) for row in rows} == {
        ("basic arithmetic", "unit conversion"),
        ("newton's laws", "torque"),
    }
    assert {row["primary_domain"] for row in rows} == {
        "mathematics",
        "physics_astronomy",
    }
    for row in rows:
        assert row["supporting_documents"] == 2
        assert len(row["supporting_anchors"]) == 2
        assert (
            len(
                {
                    anchor["source_content_sha256"]
                    for anchor in row["supporting_anchors"]
                }
            )
            == 2
        )
        assert row["compiler_cooccurrence_only"] is True
        assert row["directional_prerequisite_verified"] is False
        assert row["training_ready"] is False


def test_underfilled_repeated_edge_population_fails_closed() -> None:
    anchors = [
        _anchor(1, "basic arithmetic", "unit conversion", "mathematics"),
        _anchor(2, "basic arithmetic", "unit conversion", "mathematics"),
    ]
    with pytest.raises(CompilerPrerequisiteEdgePopulationError, match="underfilled"):
        build_edge_plan(anchors, target_edges=2)


def test_disqualifying_source_risk_blocks_edge_support() -> None:
    judgment = _judgment(1, "basic arithmetic", "unit conversion", "mathematics")
    assert judgment_qualifies(judgment)
    judgment["risks"]["weak_source_grounding"] = True
    assert not judgment_qualifies(judgment)
