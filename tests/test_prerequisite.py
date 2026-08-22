from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import sai.data.prerequisite as prerequisite
from sai.data.prerequisite import (
    ANNOTATION_SCHEMA,
    CONCEPT_LIST_SCHEMA,
    TAXONOMY_SCHEMA,
    PrerequisiteError,
    _read_taxonomy,
    analyze_curriculum_annotation_files,
    analyze_progression,
    build_taxonomy,
    validate_taxonomy_payload,
)
from sai.data.prerequisite_review import SCHEMA as AUDIT_SAMPLE_SCHEMA
from sai.data.token_stream import (
    ROW_SCHEMA,
    canonical_sha256,
    normalize_document,
    sha256_file,
)


def _taxonomy() -> dict:
    concepts = [
        {
            "concept_id": "language.color",
            "name": "color words",
            "domain": "english",
            "prerequisites": [],
            "minimum_prior_documents": 0,
            "minimum_phase_documents": {
                "grounding": 1,
                "integration": 0,
                "reasoning": 0,
                "specialization": 0,
            },
        },
        {
            "concept_id": "code.variable",
            "name": "program variables",
            "domain": "code",
            "prerequisites": [],
            "minimum_prior_documents": 0,
            "minimum_phase_documents": {
                "grounding": 0,
                "integration": 1,
                "reasoning": 0,
                "specialization": 0,
            },
        },
        {
            "concept_id": "math.addition",
            "name": "addition",
            "domain": "math",
            "prerequisites": [],
            "minimum_prior_documents": 0,
            "minimum_phase_documents": {
                "grounding": 1,
                "integration": 0,
                "reasoning": 0,
                "specialization": 0,
            },
        },
        {
            "concept_id": "science.primary-colors",
            "name": "primary colors",
            "domain": "science",
            "prerequisites": ["language.color"],
            "minimum_prior_documents": 1,
            "minimum_phase_documents": {
                "grounding": 0,
                "integration": 0,
                "reasoning": 1,
                "specialization": 0,
            },
        },
        {
            "concept_id": "technical.color-mixing",
            "name": "subtractive color mixing",
            "domain": "technical",
            "prerequisites": ["science.primary-colors", "math.addition"],
            "minimum_prior_documents": 1,
            "minimum_phase_documents": {
                "grounding": 0,
                "integration": 0,
                "reasoning": 0,
                "specialization": 1,
            },
        },
    ]
    payload = {
        "schema": TAXONOMY_SCHEMA,
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "minimum_annotation_confidence_ppm": 800_000,
        "annotation_method": {
            "method": "hybrid",
            "annotator_identity_sha256": "1" * 64,
            "policy_sha256": "2" * 64,
            "audit_sample_receipt_sha256": "3" * 64,
        },
        "concepts": concepts,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _evidence(concept_id: str, index: int, confidence: int = 900_000) -> dict:
    text = "evidence"
    return {
        "concept_id": concept_id,
        "confidence_ppm": confidence,
        "evidence_spans": [
            {
                "start": 0,
                "end": len(text),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        ],
    }


def _annotations(valid: bool = True) -> tuple[list[dict], list[str], list[str]]:
    identities = [f"{index + 100:064x}" for index in range(4)]
    concepts = [
        [_evidence("language.color", 0), _evidence("math.addition", 1)],
        [_evidence("code.variable", 2)],
        [_evidence("science.primary-colors", 3)],
        [_evidence("technical.color-mixing", 4)],
    ]
    if not valid:
        concepts[0].append(_evidence("technical.color-mixing", 5))
    rows = [
        {
            "schema": ANNOTATION_SCHEMA,
            "document_identity_sha256": identity,
            "phase": phase,
            "concepts": evidence,
        }
        for identity, phase, evidence in zip(
            identities,
            ("grounding", "integration", "reasoning", "specialization"),
            concepts,
            strict=True,
        )
    ]
    return rows, identities, ["evidence"] * len(rows)


def _resign(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


def test_taxonomy_and_progression_pass_with_prior_exposure() -> None:
    taxonomy = _taxonomy()
    annotations, identities, texts = _annotations()
    assert validate_taxonomy_payload(taxonomy) == taxonomy
    report = analyze_progression(taxonomy, annotations, identities, texts)
    assert report["status"] == "qualified"
    assert report["progression_qualified"] is True
    assert report["violations"] == []
    assert report["concepts"]["technical.color-mixing"] == {
        "confident_documents": 1,
        "first_document_index": 3,
        "phase_documents": {
            "grounding": 0,
            "integration": 0,
            "reasoning": 0,
            "specialization": 1,
        },
    }
    assert report["phase_coverage_violations"] == []
    assert report["ordered_document_identity_sha256"] == canonical_sha256(identities)
    assert report["annotations_sha256"] == canonical_sha256(annotations)
    assert report["training_authorized"] is False
    assert report["four_b_training_authorized"] is False


def test_builds_taxonomy_from_real_evidence_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    concepts = tmp_path / "concepts.json"
    concepts.write_text(
        json.dumps(
            {
                "schema": CONCEPT_LIST_SCHEMA,
                "status": "candidate",
                "concepts": _taxonomy()["concepts"],
            }
        )
        + "\n"
    )
    annotator = tmp_path / "annotator.json"
    policy = tmp_path / "policy.json"
    audit = tmp_path / "audit.json"
    annotator.write_text('{"model":"frozen-annotator"}\n')
    policy.write_text('{"policy":"frozen-semantic-evidence-v1"}\n')
    audit_payload = {"schema": AUDIT_SAMPLE_SCHEMA, "evidence": "replayed elsewhere"}
    audit.write_text(json.dumps(audit_payload) + "\n")
    monkeypatch.setattr(
        prerequisite,
        "validate_review_payload",
        lambda *_args, **_kwargs: {"status": "passed", "audit_qualified": True},
    )
    output = tmp_path / "taxonomy.json"
    payload = build_taxonomy(
        concepts,
        annotator,
        policy,
        audit,
        output,
        annotation_method="hybrid",
        minimum_annotation_confidence_ppm=800_000,
    )
    assert _read_taxonomy(output) == payload
    assert payload["annotation_method"] == {
        "method": "hybrid",
        "annotator_identity_sha256": sha256_file(annotator),
        "policy_sha256": sha256_file(policy),
        "audit_sample_receipt_sha256": sha256_file(audit),
    }
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    with pytest.raises(PrerequisiteError, match="already exists"):
        build_taxonomy(
            concepts,
            annotator,
            policy,
            audit,
            output,
            annotation_method="hybrid",
            minimum_annotation_confidence_ppm=800_000,
        )
    duplicate_audit = tmp_path / "duplicate-audit.json"
    duplicate_audit.write_text(json.dumps(audit_payload) + "\n")
    with pytest.raises(PrerequisiteError, match="not distinct"):
        build_taxonomy(
            concepts,
            policy,
            policy,
            duplicate_audit,
            tmp_path / "duplicate-evidence-taxonomy.json",
            annotation_method="hybrid",
            minimum_annotation_confidence_ppm=800_000,
        )


def test_same_document_prerequisites_do_not_count_as_prior() -> None:
    annotations, identities, texts = _annotations(valid=False)
    report = analyze_progression(_taxonomy(), annotations, identities, texts)
    assert report["status"] == "not_qualified"
    assert report["progression_qualified"] is False
    violation = report["violations"][0]
    assert violation["concept_id"] == "technical.color-mixing"
    assert violation["observed_prior_documents"] == 0


def test_missing_later_rehearsal_fails_progression() -> None:
    taxonomy = deepcopy(_taxonomy())
    taxonomy["concepts"][0]["minimum_phase_documents"]["specialization"] = 1
    _resign(taxonomy)
    annotations, identities, texts = _annotations()
    report = analyze_progression(taxonomy, annotations, identities, texts)
    assert report["status"] == "not_qualified"
    assert report["progression_qualified"] is False
    assert report["violations"] == []
    assert report["phase_coverage_violations"] == [
        {
            "concept_id": "language.color",
            "phase": "specialization",
            "required_documents": 1,
            "observed_documents": 0,
        }
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["concepts"][0].update(
            prerequisites=["science.primary-colors"], minimum_prior_documents=1
        ),
        lambda value: value["concepts"][4].update(prerequisites=["missing.concept"]),
        lambda value: value["concepts"][0].update(domain="math"),
        lambda value: value["annotation_method"].update(
            audit_sample_receipt_sha256="0" * 64
        ),
        lambda value: value.update(training_authorized=True),
    ],
)
def test_rejects_resigned_taxonomy_drift(mutate) -> None:
    taxonomy = deepcopy(_taxonomy())
    mutate(taxonomy)
    _resign(taxonomy)
    with pytest.raises(PrerequisiteError):
        validate_taxonomy_payload(taxonomy)


def test_rejects_annotation_identity_phase_and_evidence_tamper() -> None:
    annotations, identities, texts = _annotations()
    annotations[0]["document_identity_sha256"] = "f" * 64
    with pytest.raises(PrerequisiteError, match="document order"):
        analyze_progression(_taxonomy(), annotations, identities, texts)

    annotations, identities, texts = _annotations()
    annotations[2]["phase"] = "grounding"
    with pytest.raises(PrerequisiteError, match="not monotonic"):
        analyze_progression(_taxonomy(), annotations, identities, texts)

    annotations, identities, texts = _annotations()
    annotations[0]["concepts"][0]["evidence_spans"][0]["text_sha256"] = "0" * 64
    with pytest.raises(PrerequisiteError, match="placeholder"):
        analyze_progression(_taxonomy(), annotations, identities, texts)


def test_taxonomy_file_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "taxonomy.json"
    target.write_text(json.dumps(_taxonomy()))
    assert _read_taxonomy(target) == _taxonomy()

    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(PrerequisiteError, match="unsafe"):
        _read_taxonomy(link)

    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(target)
    with pytest.raises(PrerequisiteError, match="unsafe"):
        _read_taxonomy(hardlink)


def test_streaming_audit_reopens_exact_curriculum_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(_taxonomy()) + "\n")
    curriculum_path = tmp_path / "curriculum.jsonl"
    rows = []
    for index, domain in enumerate(("english", "code", "math", "science")):
        rows.append(
            normalize_document(
                {
                    "schema": ROW_SCHEMA,
                    "text": f"curriculum document {index} " + "clear words " * 80,
                    "source": {
                        "dataset": "prerequisite-test",
                        "row_id": str(index),
                        "license": "CC0",
                        "domain": domain,
                    },
                    "verification": {
                        "benchmark_disjoint": True,
                        "evidence_sha256": f"{index + 200:064x}",
                    },
                }
            )
        )
    curriculum_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    annotations, _, _ = _annotations()
    for annotation, row in zip(annotations, rows, strict=True):
        annotation["document_identity_sha256"] = row["identity_sha256"]
        for evidence in annotation["concepts"]:
            span_text = row["text"][:10]
            evidence["evidence_spans"] = [
                {
                    "start": 0,
                    "end": len(span_text),
                    "text_sha256": hashlib.sha256(span_text.encode()).hexdigest(),
                }
            ]
    annotations_path = tmp_path / "annotations.jsonl"
    annotations_path.write_text(
        "".join(json.dumps(annotation) + "\n" for annotation in annotations)
    )
    curriculum_payload = {
        "receipt_sha256": "a" * 64,
        "output": {
            "path": str(curriculum_path),
            "bytes": curriculum_path.stat().st_size,
            "sha256": sha256_file(curriculum_path),
        },
        "phases": {
            phase: {"documents": 1}
            for phase in ("grounding", "integration", "reasoning", "specialization")
        },
    }
    monkeypatch.setattr(
        prerequisite,
        "validate_curriculum",
        lambda receipt, workers=1: curriculum_payload,
    )
    output = tmp_path / "progression.json"
    report = analyze_curriculum_annotation_files(
        taxonomy_path,
        tmp_path / "curriculum.receipt.json",
        annotations_path,
        output,
        workers=2,
    )
    assert report["status"] == "qualified"
    assert json.loads(output.read_text()) == report
    assert output.stat().st_mode & 0o777 == 0o444
    assert report["ordered_document_identity_sha256"] == canonical_sha256(
        [row["identity_sha256"] for row in rows]
    )
    assert report["annotations_sha256"] == canonical_sha256(annotations)
    assert report["curriculum_lineage"] == {
        "curriculum_receipt_sha256": "a" * 64,
        "curriculum_output_bytes": curriculum_path.stat().st_size,
        "curriculum_output_sha256": sha256_file(curriculum_path),
        "annotations_path": str(annotations_path.resolve()),
        "annotations_bytes": annotations_path.stat().st_size,
        "annotations_file_sha256": sha256_file(annotations_path),
    }
    with pytest.raises(PrerequisiteError, match="already exists"):
        analyze_curriculum_annotation_files(
            taxonomy_path,
            tmp_path / "curriculum.receipt.json",
            annotations_path,
            output,
        )
