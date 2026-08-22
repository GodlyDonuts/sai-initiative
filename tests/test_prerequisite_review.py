from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sai.data.prerequisite_review as review_module
from sai.data.prerequisite import build_taxonomy
from sai.data.prerequisite_review import (
    PrerequisiteReviewError,
    build_review_receipt,
    validate_review_receipt,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.write_bytes(encoded)
    return encoded


def _artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, disagreements: int = 6
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    population = []
    phases = ("grounding", "integration", "reasoning", "specialization")
    for index in range(120):
        text = f"Evidence token for semantic document {index}."
        population.append(
            {
                "schema": "sai-semantic-prerequisite-audit-document-v1",
                "document_index": index,
                "phase": phases[index // 30],
                "surface_band": ("basic", "intermediate", "advanced", "specialized")[
                    index % 4
                ],
                "selection_rank_sha256": f"{index + 1:064x}",
                "document_identity_sha256": hashlib.sha256(
                    f"document-{index}".encode()
                ).hexdigest(),
                "source": {
                    "dataset": "semantic-review-test",
                    "revision": "r1",
                    "record_id": str(index),
                },
                "text": text,
            }
        )
    population_output = tmp_path / "population.jsonl"
    population_bytes = _write_jsonl(population_output, population)
    population_payload = {
        "schema": "sai-semantic-prerequisite-audit-population-v1",
        "receipt_sha256": "a" * 64,
        "selection": {
            "per_stratum": 8,
            "strata": 15,
            "selected_documents": 120,
            "excluded_structurally_empty_strata": ["grounding:specialization"],
        },
        "output": {
            "path": str(population_output.resolve()),
            "bytes": len(population_bytes),
            "sha256": hashlib.sha256(population_bytes).hexdigest(),
            "ordered_population_sha256": canonical_sha256(population),
        },
    }
    population_receipt = tmp_path / "population-receipt.json"
    population_receipt.write_text(json.dumps(population_payload) + "\n")
    monkeypatch.setattr(
        review_module,
        "validate_audit_population",
        lambda *_args, **_kwargs: population_payload,
    )

    concepts = ROOT / "docs" / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json"
    annotator_identity = tmp_path / "annotator-identity.json"
    policy = ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json"
    reviewer_identity = tmp_path / "reviewer-identity.json"
    annotator_identity.write_text('{"identity":"prospective-annotator-v1"}\n')
    reviewer_identity.write_text('{"identity":"independent-human-reviewer-v1"}\n')

    proposed = []
    reviewed = []
    for index, document in enumerate(population):
        span_text = "Evidence"
        evidence = {
            "concept_id": "english.symbols",
            "confidence_ppm": 950_000,
            "evidence_spans": [
                {
                    "start": 0,
                    "end": len(span_text),
                    "text_sha256": hashlib.sha256(span_text.encode()).hexdigest(),
                }
            ],
        }
        base = {
            "schema": "sai-prerequisite-document-annotation-v1",
            "document_identity_sha256": document["document_identity_sha256"],
            "phase": document["phase"],
        }
        proposed.append({**base, "concepts": [evidence]})
        reviewed.append(
            {**base, "concepts": [] if index < disagreements else [evidence]}
        )
    annotator_annotations = tmp_path / "annotator.jsonl"
    reviewer_annotations = tmp_path / "reviewer.jsonl"
    _write_jsonl(annotator_annotations, proposed)
    _write_jsonl(reviewer_annotations, reviewed)
    return {
        "population_receipt": population_receipt,
        "concepts": concepts,
        "annotator_identity": annotator_identity,
        "policy": policy,
        "reviewer_identity": reviewer_identity,
        "annotator_annotations": annotator_annotations,
        "reviewer_annotations": reviewer_annotations,
    }


def _build(paths: dict[str, Path], output: Path) -> dict:
    return build_review_receipt(
        paths["population_receipt"],
        paths["concepts"],
        paths["annotator_identity"],
        paths["policy"],
        paths["annotator_annotations"],
        paths["reviewer_identity"],
        paths["reviewer_annotations"],
        output,
        curriculum_workers=3,
    )


def test_builds_replays_and_qualifies_independent_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _artifacts(tmp_path, monkeypatch)
    output = tmp_path / "review-receipt.json"
    payload = _build(paths, output)
    assert payload["status"] == "passed"
    assert payload["audit_qualified"] is True
    assert payload["reviewed_documents"] == 120
    assert payload["disagreement_documents"] == 6
    assert payload["observed_disagreement_ppm"] == 50_000
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert (
        validate_review_receipt(
            output,
            expected_annotator_identity_sha256=sha256_file(paths["annotator_identity"]),
            expected_annotation_policy_sha256=sha256_file(paths["policy"]),
            expected_concept_list_sha256=sha256_file(paths["concepts"]),
            curriculum_workers=2,
        )
        == payload
    )
    taxonomy = build_taxonomy(
        paths["concepts"],
        paths["annotator_identity"],
        paths["policy"],
        output,
        tmp_path / "taxonomy.json",
        annotation_method="hybrid",
        minimum_annotation_confidence_ppm=800_000,
        maximum_new_concepts_per_document=2,
    )
    assert taxonomy["annotation_method"]["audit_sample_receipt_sha256"] == sha256_file(
        output
    )


def test_failed_disagreement_receipt_cannot_build_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _artifacts(tmp_path, monkeypatch, disagreements=7)
    output = tmp_path / "failed-review.json"
    payload = _build(paths, output)
    assert payload["status"] == "failed"
    assert payload["audit_qualified"] is False
    with pytest.raises(RuntimeError, match="audit sample qualification differs"):
        build_taxonomy(
            paths["concepts"],
            paths["annotator_identity"],
            paths["policy"],
            output,
            tmp_path / "taxonomy.json",
            annotation_method="hybrid",
            minimum_annotation_confidence_ppm=800_000,
            maximum_new_concepts_per_document=2,
        )


def test_rejects_evidence_and_post_receipt_annotation_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _artifacts(tmp_path, monkeypatch)
    rows = [
        json.loads(line)
        for line in paths["annotator_annotations"].read_text().splitlines()
    ]
    rows[0]["concepts"][0]["evidence_spans"][0]["text_sha256"] = "f" * 64
    _write_jsonl(paths["annotator_annotations"], rows)
    with pytest.raises(PrerequisiteReviewError, match="evidence text differs"):
        _build(paths, tmp_path / "bad-evidence.json")

    paths = _artifacts(tmp_path / "second", monkeypatch)
    output = tmp_path / "review.json"
    _build(paths, output)
    paths["reviewer_annotations"].write_text(
        paths["reviewer_annotations"].read_text() + "{}\n"
    )
    with pytest.raises(PrerequisiteReviewError, match="descriptor differs"):
        validate_review_receipt(
            output,
            expected_annotator_identity_sha256=sha256_file(paths["annotator_identity"]),
            expected_annotation_policy_sha256=sha256_file(paths["policy"]),
            expected_concept_list_sha256=sha256_file(paths["concepts"]),
        )
