from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.foundational_syllabus import (
    FoundationalSyllabusError,
    build,
    validate,
)

ROOT = Path(__file__).parents[1]


def _inputs(tmp_path: Path) -> dict:
    return {
        "base_concepts": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "additions": ROOT
        / "docs"
        / "SAI_FOUNDATIONAL_SYLLABUS_ADDITIONS_CANDIDATE.json",
        "concept_output": tmp_path / "concepts.json",
        "receipt_output": tmp_path / "receipt.json",
    }


def test_builds_acyclic_balanced_expanded_candidate(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = build(**inputs)
    assert payload["composed"]["concepts"] == 125
    assert payload["composed"]["concepts_by_domain"] == {
        "code": 25,
        "english": 25,
        "math": 25,
        "science": 25,
        "technical": 25,
    }
    assert sum(payload["composed"]["concepts_by_earliest_phase"].values()) == 125
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate(**inputs) == payload


def test_rejects_forward_phase_dependency_and_cycle(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    additions = json.loads(inputs["additions"].read_text())
    additions["concepts"][0]["prerequisites"] = ["technical.machine-learning"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(additions) + "\n")
    with pytest.raises(FoundationalSyllabusError, match="addition|phase"):
        build(**{**inputs, "additions": bad})


def test_outputs_are_create_only_and_tamper_evident(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    build(**inputs)
    with pytest.raises(FoundationalSyllabusError, match="output boundary"):
        build(**inputs)
    inputs["concept_output"].chmod(0o644)
    inputs["concept_output"].write_text(inputs["concept_output"].read_text() + " ")
    with pytest.raises(FoundationalSyllabusError, match="output differs"):
        validate(**inputs)
