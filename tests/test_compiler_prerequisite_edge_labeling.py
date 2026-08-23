from __future__ import annotations

from copy import deepcopy

import pytest

from sai.data.compiler_prerequisite_edge_labeling import (
    CompilerPrerequisiteEdgeLabelingError,
    build_messages,
    normalize_candidate,
    normalize_model_judgment,
)
from sai.data.compiler_prerequisite_edge_population import build_edge_plan
from tests.test_compiler_prerequisite_edge_population import _anchor


def _candidate() -> dict:
    return build_edge_plan(
        [
            _anchor(1, "basic arithmetic", "unit conversion", "mathematics"),
            _anchor(2, "basic arithmetic", "unit conversion", "mathematics"),
        ],
        target_edges=1,
        seed=7,
    )[0]


def _strict(candidate: dict) -> dict:
    return {
        "verdict": "strict_prerequisite",
        "source_checks": [
            {
                "anchor_index": index,
                "concept_present": True,
                "prerequisite_assumed": True,
                "concept_quote": anchor["text"],
                "prerequisite_quote": anchor["text"],
                "rationale": "The exact source explicitly binds both labels.",
            }
            for index, anchor in enumerate(candidate["supporting_anchors"])
        ],
        "direction_supported": True,
        "reverse_direction_plausible": False,
        "prerequisite_definition": (
            "Basic arithmetic is the ability to perform elementary numerical "
            "operations."
        ),
        "concept_definition": (
            "Unit conversion maps a quantity between compatible measurement units."
        ),
        "limitations": [
            "The edge applies to numerical unit conversions requiring calculation."
        ],
        "defects": [],
        "confidence_ppm": 900_000,
        "rationale": "Both independent source excerpts support the same direction.",
    }


def test_candidate_and_messages_bind_every_exact_source() -> None:
    candidate = normalize_candidate(_candidate())
    messages = build_messages(candidate)
    assert len(messages) == 2
    assert "basic arithmetic" in messages[1]["content"]
    assert "unit conversion" in messages[1]["content"]
    for anchor in candidate["supporting_anchors"]:
        assert anchor["text"] in messages[1]["content"]


def test_malformed_anchor_fails_closed_without_attribute_error() -> None:
    candidate = _candidate()
    candidate["supporting_anchors"][0] = "untrusted-anchor"
    with pytest.raises(
        CompilerPrerequisiteEdgeLabelingError, match="candidate differs"
    ):
        normalize_candidate(candidate)


def test_strict_edge_remains_same_family_and_nontraining() -> None:
    candidate = _candidate()
    result = normalize_model_judgment(_strict(candidate), candidate)
    assert result["verdict"] == "strict_prerequisite"
    assert result["independent_request_verification_complete"] is True
    assert result["independent_model_family_verification_complete"] is False
    assert result["directional_prerequisite_verified"] is False
    assert result["training_ready"] is False


def test_nonliteral_source_quote_fails_closed() -> None:
    candidate = _candidate()
    payload = _strict(candidate)
    payload["source_checks"][0]["concept_quote"] = "invented quote"
    with pytest.raises(CompilerPrerequisiteEdgeLabelingError, match="not exact"):
        normalize_model_judgment(payload, candidate)


def test_strict_route_rejects_missing_directional_support() -> None:
    candidate = _candidate()
    payload = deepcopy(_strict(candidate))
    payload["direction_supported"] = False
    with pytest.raises(CompilerPrerequisiteEdgeLabelingError, match="inconsistent"):
        normalize_model_judgment(payload, candidate)
