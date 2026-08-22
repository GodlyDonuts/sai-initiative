from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sai.data.prerequisite import (
    ANNOTATION_SCHEMA,
    TAXONOMY_SCHEMA,
    PrerequisiteError,
    _read_taxonomy,
    analyze_progression,
    validate_taxonomy_payload,
)
from sai.data.token_stream import canonical_sha256


def _taxonomy() -> dict:
    concepts = [
        {
            "concept_id": "language.color",
            "name": "color words",
            "domain": "english",
            "prerequisites": [],
            "minimum_prior_documents": 0,
        },
        {
            "concept_id": "code.variable",
            "name": "program variables",
            "domain": "code",
            "prerequisites": [],
            "minimum_prior_documents": 0,
        },
        {
            "concept_id": "math.addition",
            "name": "addition",
            "domain": "math",
            "prerequisites": [],
            "minimum_prior_documents": 0,
        },
        {
            "concept_id": "science.primary-colors",
            "name": "primary colors",
            "domain": "science",
            "prerequisites": ["language.color"],
            "minimum_prior_documents": 1,
        },
        {
            "concept_id": "technical.color-mixing",
            "name": "subtractive color mixing",
            "domain": "technical",
            "prerequisites": ["science.primary-colors", "math.addition"],
            "minimum_prior_documents": 1,
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
    return {
        "concept_id": concept_id,
        "confidence_ppm": confidence,
        "evidence_sha256": f"{index + 20:064x}",
    }


def _annotations(valid: bool = True) -> tuple[list[dict], list[str]]:
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
    return rows, identities


def _resign(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


def test_taxonomy_and_progression_pass_with_prior_exposure() -> None:
    taxonomy = _taxonomy()
    annotations, identities = _annotations()
    assert validate_taxonomy_payload(taxonomy) == taxonomy
    report = analyze_progression(taxonomy, annotations, identities)
    assert report["status"] == "qualified"
    assert report["progression_qualified"] is True
    assert report["violations"] == []
    assert report["concepts"]["technical.color-mixing"] == {
        "confident_documents": 1,
        "first_document_index": 3,
    }
    assert report["ordered_document_identity_sha256"] == canonical_sha256(identities)
    assert report["annotations_sha256"] == canonical_sha256(annotations)
    assert report["training_authorized"] is False
    assert report["four_b_training_authorized"] is False


def test_same_document_prerequisites_do_not_count_as_prior() -> None:
    annotations, identities = _annotations(valid=False)
    report = analyze_progression(_taxonomy(), annotations, identities)
    assert report["status"] == "not_qualified"
    assert report["progression_qualified"] is False
    violation = report["violations"][0]
    assert violation["concept_id"] == "technical.color-mixing"
    assert violation["observed_prior_documents"] == 0


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
    annotations, identities = _annotations()
    annotations[0]["document_identity_sha256"] = "f" * 64
    with pytest.raises(PrerequisiteError, match="document order"):
        analyze_progression(_taxonomy(), annotations, identities)

    annotations, identities = _annotations()
    annotations[2]["phase"] = "grounding"
    with pytest.raises(PrerequisiteError, match="not monotonic"):
        analyze_progression(_taxonomy(), annotations, identities)

    annotations, identities = _annotations()
    annotations[0]["concepts"][0]["evidence_sha256"] = "0" * 64
    with pytest.raises(PrerequisiteError, match="placeholder"):
        analyze_progression(_taxonomy(), annotations, identities)


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
