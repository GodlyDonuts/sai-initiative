from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.foundational_syllabus import _prepare as _prepare_syllabus
from sai.data.foundational_syllabus_audit import _prepare as _prepare_audit
from sai.data.foundational_syllabus_review_apply import (
    FoundationalSyllabusReviewApplyError,
    apply_reviews,
)
from sai.data.foundational_syllabus_review_compare import compare
from sai.data.foundational_syllabus_review_workspace import _review_rows

ROOT = Path(__file__).parents[1]


def _encoded(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _inputs(
    tmp_path: Path, *, supporting_edge: bool = False, disagreement: bool = False
) -> dict:
    base = ROOT / "docs" / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json"
    additions = ROOT / "docs" / "SAI_FOUNDATIONAL_SYLLABUS_ADDITIONS_CANDIDATE.json"
    _, concept_encoded, _ = _prepare_syllabus(base_concepts=base, additions=additions)
    audit, _ = _prepare_audit(base_concepts=base, additions=additions)
    source_rows = _review_rows(json.loads(concept_encoded), audit)
    reviews = []
    for reviewer in ("reviewer-a", "reviewer-b"):
        rows = []
        for source in source_rows:
            changed = (
                supporting_edge
                and source["concept_id"] == "english.words"
                and not (disagreement and reviewer == "reviewer-a")
            )
            proposed_phase = (
                "reasoning"
                if source["concept_id"] == "code.testing"
                else source["earliest_phase"]
            )
            material_change = changed or proposed_phase != source["earliest_phase"]
            edges = [
                {
                    "prerequisite_id": edge["concept_id"],
                    "classification": "supporting" if changed else "hard",
                    "rationale": "This edge has the declared pedagogical relationship.",
                }
                for edge in source["prerequisites"]
            ]
            rows.append(
                {
                    "schema": "sai-foundational-syllabus-subject-review-v1",
                    "reviewer_id": reviewer,
                    "concept_id": source["concept_id"],
                    "concept_verdict": "revise" if material_change else "accept",
                    "proposed_name": None,
                    "proposed_earliest_phase": proposed_phase,
                    "granularity": "appropriate",
                    "edge_reviews": edges,
                    "missing_prerequisites": [],
                    "rationale": (
                        "The concept and its prerequisite relationships have been "
                        "reviewed against the candidate syllabus."
                    ),
                }
            )
        path = tmp_path / f"{reviewer}.jsonl"
        path.write_bytes(_encoded(rows))
        reviews.append(path)
    comparison = tmp_path / "comparison.json"
    compare(
        base_concepts=base,
        additions=additions,
        review_a=reviews[0],
        expected_review_a_sha256=hashlib.sha256(reviews[0].read_bytes()).hexdigest(),
        review_b=reviews[1],
        expected_review_b_sha256=hashlib.sha256(reviews[1].read_bytes()).hexdigest(),
        output=comparison,
    )
    return {
        "base_concepts": base,
        "additions": additions,
        "review_a": reviews[0],
        "expected_review_a_sha256": hashlib.sha256(reviews[0].read_bytes()).hexdigest(),
        "review_b": reviews[1],
        "expected_review_b_sha256": hashlib.sha256(reviews[1].read_bytes()).hexdigest(),
        "comparison": comparison,
        "expected_comparison_sha256": hashlib.sha256(
            comparison.read_bytes()
        ).hexdigest(),
        "concept_output": tmp_path / "reviewed-concepts.json",
        "supporting_output": tmp_path / "supporting.json",
        "receipt_output": tmp_path / "receipt.json",
    }


def test_applies_unanimous_hard_graph_and_revalidates(tmp_path: Path) -> None:
    payload = apply_reviews(**_inputs(tmp_path))
    assert payload["changes"] == {
        "hard_edges": 263,
        "supporting_edges": 0,
        "removed_existing_edges": 0,
        "added_hard_edges": 0,
        "added_supporting_edges": 0,
        "renamed_concepts": 0,
        "rephased_concepts": 1,
    }
    assert payload["hard_graph_revalidated"] is True
    assert payload["subject_review_consensus_applied"] is True
    assert payload["subject_review_qualified"] is False
    assert payload["training_authorized"] is False


def test_moves_agreed_supporting_edge_out_of_hard_graph(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, supporting_edge=True)
    payload = apply_reviews(**inputs)
    assert payload["changes"]["hard_edges"] == 262
    assert payload["changes"]["supporting_edges"] == 1
    concepts = json.loads(inputs["concept_output"].read_text())["concepts"]
    words = next(row for row in concepts if row["concept_id"] == "english.words")
    assert words["prerequisites"] == []
    supporting = json.loads(inputs["supporting_output"].read_text())["rows"]
    words_support = next(
        row for row in supporting if row["concept_id"] == "english.words"
    )
    assert words_support["supporting_concepts"] == ["english.symbols"]


def test_refuses_unresolved_comparison(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, supporting_edge=True, disagreement=True)
    with pytest.raises(
        FoundationalSyllabusReviewApplyError, match="consensus is unresolved"
    ):
        apply_reviews(**inputs)
