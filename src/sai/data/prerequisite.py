"""Validate concept prerequisites and audit an ordered curriculum annotation set."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.curriculum import PHASES
from sai.data.token_stream import ALLOWED_DOMAINS, canonical_sha256

TAXONOMY_SCHEMA = "sai-semantic-prerequisite-taxonomy-v1"
ANNOTATION_SCHEMA = "sai-prerequisite-document-annotation-v1"
REPORT_SCHEMA = "sai-semantic-prerequisite-progression-report-v1"
ANNOTATION_METHODS = {"deterministic", "model", "hybrid", "human"}
_CONCEPT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{1,95}")
_MAX_TAXONOMY_BYTES = 8 << 20
_TAXONOMY_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "minimum_annotation_confidence_ppm",
    "annotation_method",
    "concepts",
    "receipt_sha256",
}
_METHOD_KEYS = {
    "method",
    "annotator_identity_sha256",
    "policy_sha256",
    "audit_sample_receipt_sha256",
}
_CONCEPT_KEYS = {
    "concept_id",
    "name",
    "domain",
    "prerequisites",
    "minimum_prior_documents",
}
_ANNOTATION_KEYS = {"schema", "document_identity_sha256", "phase", "concepts"}
_EVIDENCE_KEYS = {"concept_id", "confidence_ppm", "evidence_sha256"}


class PrerequisiteError(RuntimeError):
    """A taxonomy, annotation, document identity, or prerequisite order differs."""


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PrerequisiteError(f"{label} fields differ")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PrerequisiteError(f"{label} differs")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise PrerequisiteError(f"{label} differs") from error
    if not any(raw):
        raise PrerequisiteError(f"{label} is a placeholder")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrerequisiteError(f"{label} differs")
    return value


def _ppm(value: Any, label: str, *, positive: bool = False) -> int:
    result = _nonnegative_int(value, label)
    if result > 1_000_000 or (positive and result == 0):
        raise PrerequisiteError(f"{label} differs")
    return result


def _assert_acyclic(concepts: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(concept_id: str) -> None:
        if concept_id in visiting:
            raise PrerequisiteError("concept prerequisite graph contains a cycle")
        if concept_id in visited:
            return
        visiting.add(concept_id)
        for prerequisite in concepts[concept_id]["prerequisites"]:
            visit(prerequisite)
        visiting.remove(concept_id)
        visited.add(concept_id)

    for concept_id in concepts:
        visit(concept_id)


def validate_taxonomy_payload(payload: Any) -> dict[str, Any]:
    """Validate one immutable prospective semantic-prerequisite taxonomy."""

    taxonomy = _exact_keys(payload, _TAXONOMY_KEYS, "prerequisite taxonomy")
    if (
        taxonomy["schema"] != TAXONOMY_SCHEMA
        or taxonomy["status"] != "prospective"
        or taxonomy["training_authorized"] is not False
        or taxonomy["four_b_training_authorized"] is not False
    ):
        raise PrerequisiteError("taxonomy authorization boundary differs")
    receipt_sha256 = _sha256(taxonomy["receipt_sha256"], "taxonomy receipt")
    unsigned = {
        key: value for key, value in taxonomy.items() if key != "receipt_sha256"
    }
    if receipt_sha256 != canonical_sha256(unsigned):
        raise PrerequisiteError("taxonomy receipt hash differs")
    _ppm(
        taxonomy["minimum_annotation_confidence_ppm"],
        "minimum annotation confidence",
        positive=True,
    )

    method = _exact_keys(taxonomy["annotation_method"], _METHOD_KEYS, "annotation")
    if method["method"] not in ANNOTATION_METHODS:
        raise PrerequisiteError("annotation method differs")
    for field in (
        "annotator_identity_sha256",
        "policy_sha256",
        "audit_sample_receipt_sha256",
    ):
        _sha256(method[field], field)

    raw_concepts = taxonomy["concepts"]
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise PrerequisiteError("taxonomy concepts differ")
    concepts: dict[str, dict[str, Any]] = {}
    domains: set[str] = set()
    for raw_concept in raw_concepts:
        concept = _exact_keys(raw_concept, _CONCEPT_KEYS, "concept")
        concept_id = concept["concept_id"]
        if (
            not isinstance(concept_id, str)
            or _CONCEPT_ID.fullmatch(concept_id) is None
            or concept_id in concepts
        ):
            raise PrerequisiteError("concept identity differs or is duplicated")
        if not isinstance(concept["name"], str) or not concept["name"].strip():
            raise PrerequisiteError("concept name differs")
        if concept["domain"] not in ALLOWED_DOMAINS:
            raise PrerequisiteError("concept domain differs")
        domains.add(concept["domain"])
        prerequisites = concept["prerequisites"]
        if (
            not isinstance(prerequisites, list)
            or any(not isinstance(item, str) for item in prerequisites)
            or len(prerequisites) != len(set(prerequisites))
            or concept_id in prerequisites
        ):
            raise PrerequisiteError("concept prerequisites differ")
        minimum = _nonnegative_int(
            concept["minimum_prior_documents"], "minimum prior documents"
        )
        if bool(prerequisites) != bool(minimum):
            raise PrerequisiteError("concept prerequisite threshold differs")
        concepts[concept_id] = concept

    if domains != ALLOWED_DOMAINS:
        raise PrerequisiteError("taxonomy does not cover every Sai domain")
    for concept in concepts.values():
        if any(item not in concepts for item in concept["prerequisites"]):
            raise PrerequisiteError("concept prerequisite is absent from taxonomy")
    _assert_acyclic(concepts)
    return taxonomy


def _read_taxonomy(path: Path) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise PrerequisiteError("taxonomy is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_TAXONOMY_BYTES
        ):
            raise PrerequisiteError("taxonomy is missing or unsafe")
        chunks = []
        remaining = _MAX_TAXONOMY_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(encoded) != before.st_size or len(encoded) > _MAX_TAXONOMY_BYTES:
        raise PrerequisiteError("taxonomy size differs")
    if (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PrerequisiteError("taxonomy changed while reading")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteError("taxonomy JSON differs") from error
    return validate_taxonomy_payload(payload)


def analyze_progression(
    taxonomy_payload: Any,
    annotations: Any,
    expected_document_identities: list[str],
) -> dict[str, Any]:
    """Audit whether ordered concept exposures respect declared prerequisites."""

    taxonomy = validate_taxonomy_payload(taxonomy_payload)
    if (
        not isinstance(annotations, list)
        or not annotations
        or len(annotations) != len(expected_document_identities)
    ):
        raise PrerequisiteError("annotation population differs")
    expected = [
        _sha256(identity, "expected document identity")
        for identity in expected_document_identities
    ]
    if len(expected) != len(set(expected)):
        raise PrerequisiteError("expected document identity is duplicated")

    concepts = {item["concept_id"]: item for item in taxonomy["concepts"]}
    minimum_confidence = taxonomy["minimum_annotation_confidence_ppm"]
    prior_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    first_exposure: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    previous_phase = -1

    for index, (raw_annotation, expected_identity) in enumerate(
        zip(annotations, expected, strict=True)
    ):
        annotation = _exact_keys(raw_annotation, _ANNOTATION_KEYS, "annotation row")
        if annotation["schema"] != ANNOTATION_SCHEMA:
            raise PrerequisiteError("annotation row schema differs")
        identity = _sha256(
            annotation["document_identity_sha256"], "annotation document identity"
        )
        if identity != expected_identity:
            raise PrerequisiteError("annotation document order differs")
        phase = annotation["phase"]
        if phase not in PHASES:
            raise PrerequisiteError("annotation phase differs")
        phase_index = PHASES.index(phase)
        if phase_index < previous_phase:
            raise PrerequisiteError("annotation phases are not monotonic")
        previous_phase = phase_index
        phase_counts[phase] += 1
        raw_evidence = annotation["concepts"]
        if not isinstance(raw_evidence, list):
            raise PrerequisiteError("annotation concepts differ")
        seen: set[str] = set()
        confident: list[str] = []
        for raw_item in raw_evidence:
            item = _exact_keys(raw_item, _EVIDENCE_KEYS, "concept evidence")
            concept_id = item["concept_id"]
            if concept_id not in concepts or concept_id in seen:
                raise PrerequisiteError("annotation concept differs or is duplicated")
            seen.add(concept_id)
            confidence = _ppm(item["confidence_ppm"], "annotation confidence")
            _sha256(item["evidence_sha256"], "annotation evidence")
            if confidence < minimum_confidence:
                continue
            confident.append(concept_id)
            concept_counts[concept_id] += 1
            first_exposure.setdefault(concept_id, index)
            concept = concepts[concept_id]
            minimum = concept["minimum_prior_documents"]
            for prerequisite in concept["prerequisites"]:
                observed = prior_counts[prerequisite]
                if observed < minimum:
                    violations.append(
                        {
                            "document_index": index,
                            "document_identity_sha256": identity,
                            "phase": phase,
                            "concept_id": concept_id,
                            "prerequisite": prerequisite,
                            "required_prior_documents": minimum,
                            "observed_prior_documents": observed,
                        }
                    )
        prior_counts.update(confident)

    if set(phase_counts) != set(PHASES):
        raise PrerequisiteError("annotation population does not cover every phase")
    missing_concepts = sorted(set(concepts) - set(concept_counts))
    progression_qualified = not violations and not missing_concepts
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "qualified" if progression_qualified else "not_qualified",
        "taxonomy_receipt_sha256": taxonomy["receipt_sha256"],
        "documents": len(annotations),
        "ordered_document_identity_sha256": canonical_sha256(expected),
        "annotations_sha256": canonical_sha256(annotations),
        "phase_documents": {phase: phase_counts[phase] for phase in PHASES},
        "concepts": {
            concept_id: {
                "confident_documents": concept_counts[concept_id],
                "first_document_index": first_exposure.get(concept_id),
            }
            for concept_id in sorted(concepts)
        },
        "missing_concepts": missing_concepts,
        "violations": violations,
        "progression_qualified": progression_qualified,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    report["receipt_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy", type=Path)
    args = parser.parse_args()
    payload = _read_taxonomy(args.taxonomy)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": "validated_prospective",
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
