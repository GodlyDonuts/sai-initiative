from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.foundational_syllabus import _prepare as _prepare_syllabus
from sai.data.foundational_syllabus_audit import _prepare as _prepare_audit
from sai.data.foundational_syllabus_review_compare import (
    FoundationalSyllabusReviewCompareError,
    compare,
)
from sai.data.foundational_syllabus_review_workspace import _review_rows

ROOT = Path(__file__).parents[1]


def _encoded(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _inputs(tmp_path: Path, *, disagreement: bool = False) -> dict:
    base = ROOT / "docs" / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json"
    additions = ROOT / "docs" / "SAI_FOUNDATIONAL_SYLLABUS_ADDITIONS_CANDIDATE.json"
    _, concept_encoded, _ = _prepare_syllabus(base_concepts=base, additions=additions)
    audit, _ = _prepare_audit(base_concepts=base, additions=additions)
    source_rows = _review_rows(json.loads(concept_encoded), audit)
    review_paths = []
    for reviewer in ("reviewer-a", "reviewer-b"):
        rows = []
        for source in source_rows:
            edge_reviews = [
                {
                    "prerequisite_id": edge["concept_id"],
                    "classification": "hard",
                    "rationale": "This dependency is necessary before the concept.",
                }
                for edge in source["prerequisites"]
            ]
            if (
                disagreement
                and reviewer == "reviewer-b"
                and source["concept_id"] == "english.words"
            ):
                edge_reviews[0]["classification"] = "supporting"
            rows.append(
                {
                    "schema": "sai-foundational-syllabus-subject-review-v1",
                    "reviewer_id": reviewer,
                    "concept_id": source["concept_id"],
                    "concept_verdict": "accept",
                    "proposed_name": None,
                    "proposed_earliest_phase": source["earliest_phase"],
                    "granularity": "appropriate",
                    "edge_reviews": edge_reviews,
                    "missing_prerequisites": [],
                    "rationale": (
                        "The concept scope and phase are appropriate for the "
                        "candidate syllabus."
                    ),
                }
            )
        path = tmp_path / f"{reviewer}.jsonl"
        path.write_bytes(_encoded(rows))
        review_paths.append(path)
    return {
        "base_concepts": base,
        "additions": additions,
        "review_a": review_paths[0],
        "expected_review_a_sha256": hashlib.sha256(
            review_paths[0].read_bytes()
        ).hexdigest(),
        "review_b": review_paths[1],
        "expected_review_b_sha256": hashlib.sha256(
            review_paths[1].read_bytes()
        ).hexdigest(),
        "output": tmp_path / "comparison.json",
    }


def test_complete_structured_consensus_is_measured_but_not_qualified(
    tmp_path: Path,
) -> None:
    payload = compare(**_inputs(tmp_path))
    assert payload["status"] == "complete_consensus_requires_final_syllabus_application"
    assert payload["summary"] == {
        "concepts": 125,
        "structured_concept_agreements": 125,
        "structured_concept_agreement_ppm": 1_000_000,
        "concept_verdict_agreements": 125,
        "concept_verdict_agreement_ppm": 1_000_000,
        "existing_edges": 263,
        "edge_classification_agreements": 263,
        "edge_classification_agreement_ppm": 1_000_000,
        "unresolved_concepts": 0,
    }
    assert payload["subject_review_qualified"] is False
    assert payload["training_authorized"] is False


def test_edge_disagreement_is_published_not_silently_selected(tmp_path: Path) -> None:
    payload = compare(**_inputs(tmp_path, disagreement=True))
    assert payload["status"] == "disagreements_require_independent_adjudication"
    assert payload["summary"]["unresolved_concepts"] == 1
    assert payload["summary"]["edge_classification_agreements"] == 262
    row = next(
        row
        for row in payload["concept_comparisons"]
        if row["concept_id"] == "english.words"
    )
    assert row["structured_agreement"] is False
    assert row["consensus"] is None


def test_rejects_same_reviewer_and_hash_tamper(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs["review_b"].read_text().splitlines()]
    for row in rows:
        row["reviewer_id"] = "reviewer-a"
    inputs["review_b"].write_bytes(_encoded(rows))
    inputs["expected_review_b_sha256"] = hashlib.sha256(
        inputs["review_b"].read_bytes()
    ).hexdigest()
    with pytest.raises(
        FoundationalSyllabusReviewCompareError, match="independent reviewer"
    ):
        compare(**inputs)
