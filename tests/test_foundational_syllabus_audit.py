from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.foundational_syllabus_audit import (
    FoundationalSyllabusAuditError,
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
        "output": tmp_path / "audit.json",
    }


def test_measures_exact_graph_risks_without_qualifying_progression(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = build(**inputs)
    assert payload["status"] == "graph_risk_review_required"
    assert payload["summary"]["roots"] == 2
    assert payload["summary"]["root_concept_ids"] == [
        "english.symbols",
        "math.number",
    ]
    assert payload["summary"]["hard_edges"] == 263
    assert payload["summary"]["cross_domain_hard_edges"] == 66
    assert payload["summary"]["hard_prerequisite_phase_inversions"] == 1
    assert payload["summary"]["maximum_hard_prerequisite_depth"] == 12
    assert payload["summary"]["flagged_concepts"] > 0
    assert payload["progression_qualified"] is False
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate(**inputs) == payload


def test_flags_deep_central_and_cross_domain_concepts(tmp_path: Path) -> None:
    payload = build(**_inputs(tmp_path))
    rows = {row["concept_id"]: row for row in payload["concept_graph_rows"]}
    assert rows["technical.cybersecurity"]["hard_prerequisite_depth"] == 12
    assert (
        "deep_hard_prerequisite_chain" in rows["technical.cybersecurity"]["risk_flags"]
    )
    assert rows["english.reference"]["direct_dependents"] == 12
    assert "high_direct_dependent_centrality" in rows["english.reference"]["risk_flags"]
    assert (
        "math.number" in rows["english.quantifier"]["cross_domain_hard_prerequisites"]
    )
    assert rows["code.testing"]["hard_prerequisite_phase_inversions"] == [
        "english.evidence"
    ]
    assert (
        "hard_prerequisite_starts_after_dependent" in rows["code.testing"]["risk_flags"]
    )


def test_audit_is_create_only_and_rejects_source_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    build(**inputs)
    with pytest.raises(FoundationalSyllabusAuditError, match="output differs"):
        build(**inputs)
    additions = json.loads(inputs["additions"].read_text())
    additions["concepts"][0]["name"] += " changed"
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(additions) + "\n")
    with pytest.raises(FoundationalSyllabusAuditError, match="output differs"):
        validate(**{**inputs, "additions": drifted})
