"""Validate concept prerequisites and audit an ordered curriculum annotation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.annotation_policy import AnnotationPolicyError, validate_policy
from sai.data.curriculum import PHASES, validate_curriculum
from sai.data.prerequisite_review import (
    PrerequisiteReviewError,
    validate_review_payload,
)
from sai.data.token_stream import (
    ALLOWED_DOMAINS,
    canonical_sha256,
    normalize_document,
)

TAXONOMY_SCHEMA = "sai-semantic-prerequisite-taxonomy-v3"
CONCEPT_LIST_SCHEMA = "sai-semantic-prerequisite-concept-list-v1"
ANNOTATION_SCHEMA = "sai-prerequisite-document-annotation-v1"
REPORT_SCHEMA = "sai-semantic-prerequisite-progression-report-v3"
ANNOTATION_METHODS = {"deterministic", "model", "hybrid", "human"}
_CONCEPT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{1,95}")
_MAX_TAXONOMY_BYTES = 8 << 20
_TAXONOMY_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "minimum_annotation_confidence_ppm",
    "minimum_evidence_codepoints_per_positive_label",
    "maximum_new_concepts_per_document",
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
    "minimum_phase_documents",
}
_ANNOTATION_KEYS = {"schema", "document_identity_sha256", "phase", "concepts"}
_EVIDENCE_KEYS = {"concept_id", "confidence_ppm", "evidence_spans"}
_SPAN_KEYS = {"start", "end", "text_sha256"}


class PrerequisiteError(RuntimeError):
    """A taxonomy, annotation, document identity, or prerequisite order differs."""


def _read_small_regular(path: Path, label: str) -> bytes:
    descriptor, before = _open_regular(path, label)
    try:
        if before.st_size <= 0 or before.st_size > _MAX_TAXONOMY_BYTES:
            raise PrerequisiteError(f"{label} size differs")
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
    if len(encoded) != before.st_size or (
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
        raise PrerequisiteError(f"{label} changed while reading")
    return encoded


def build_taxonomy(
    concepts_path: Path,
    annotator_identity_path: Path,
    annotation_policy_path: Path,
    audit_sample_receipt_path: Path,
    output_path: Path,
    *,
    annotation_method: str,
    minimum_annotation_confidence_ppm: int,
    maximum_new_concepts_per_document: int,
) -> dict[str, Any]:
    """Create one immutable prospective taxonomy from real evidence artifacts."""

    try:
        concept_bytes = _read_small_regular(concepts_path, "concept list")
        concept_source = json.loads(concept_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteError("concept list JSON differs") from error
    concept_source = _exact_keys(
        concept_source,
        {"schema", "status", "concepts"},
        "concept list",
    )
    if (
        concept_source["schema"] != CONCEPT_LIST_SCHEMA
        or concept_source["status"] != "candidate"
        or not isinstance(concept_source["concepts"], list)
        or not concept_source["concepts"]
    ):
        raise PrerequisiteError("concept list differs")
    if annotation_method not in ANNOTATION_METHODS:
        raise PrerequisiteError("annotation method differs")
    _ppm(
        minimum_annotation_confidence_ppm,
        "minimum annotation confidence",
        positive=True,
    )
    maximum_new_concepts_per_document = _nonnegative_int(
        maximum_new_concepts_per_document,
        "maximum new concepts per document",
    )
    if not 1 <= maximum_new_concepts_per_document <= 64:
        raise PrerequisiteError("maximum new concepts per document differs")
    annotator_hash = hashlib.sha256(
        _read_small_regular(annotator_identity_path, "annotator identity")
    ).hexdigest()
    policy_hash = hashlib.sha256(
        _read_small_regular(annotation_policy_path, "annotation policy")
    ).hexdigest()
    try:
        policy = validate_policy(
            annotation_policy_path,
            expected_concept_list_sha256=hashlib.sha256(concept_bytes).hexdigest(),
        )
    except AnnotationPolicyError as error:
        raise PrerequisiteError("annotation policy differs") from error
    audit_bytes = _read_small_regular(audit_sample_receipt_path, "audit sample receipt")
    try:
        audit_payload = json.loads(audit_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteError("audit sample receipt JSON differs") from error
    try:
        audit = validate_review_payload(
            audit_payload,
            expected_annotator_identity_sha256=annotator_hash,
            expected_annotation_policy_sha256=policy_hash,
            expected_concept_list_sha256=hashlib.sha256(concept_bytes).hexdigest(),
        )
    except PrerequisiteReviewError as error:
        raise PrerequisiteError("audit sample qualification differs") from error
    if audit["status"] != "passed" or audit["audit_qualified"] is not True:
        raise PrerequisiteError("audit sample qualification differs")
    evidence_hashes = [
        annotator_hash,
        policy_hash,
        hashlib.sha256(audit_bytes).hexdigest(),
    ]
    if len(set(evidence_hashes)) != len(evidence_hashes):
        raise PrerequisiteError("taxonomy evidence artifacts are not distinct")
    payload: dict[str, Any] = {
        "schema": TAXONOMY_SCHEMA,
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "minimum_annotation_confidence_ppm": minimum_annotation_confidence_ppm,
        "minimum_evidence_codepoints_per_positive_label": policy[
            "evidence_span_contract"
        ]["minimum_codepoints_per_positive_label"],
        "maximum_new_concepts_per_document": maximum_new_concepts_per_document,
        "annotation_method": {
            "method": annotation_method,
            "annotator_identity_sha256": evidence_hashes[0],
            "policy_sha256": evidence_hashes[1],
            "audit_sample_receipt_sha256": evidence_hashes[2],
        },
        "concepts": concept_source["concepts"],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    validate_taxonomy_payload(payload)
    _atomic_write_report(output_path, payload)
    return payload


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
    maximum_new_concepts_per_document = _nonnegative_int(
        taxonomy["maximum_new_concepts_per_document"],
        "maximum new concepts per document",
    )
    if not 1 <= maximum_new_concepts_per_document <= 64:
        raise PrerequisiteError("maximum new concepts per document differs")
    minimum_evidence_codepoints = _nonnegative_int(
        taxonomy["minimum_evidence_codepoints_per_positive_label"],
        "minimum evidence codepoints per positive label",
    )
    if not 1 <= minimum_evidence_codepoints <= 10_000:
        raise PrerequisiteError("minimum evidence codepoints differ")

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
        minimum_phase_documents = _exact_keys(
            concept["minimum_phase_documents"],
            set(PHASES),
            "concept phase minimums",
        )
        for phase in PHASES:
            _nonnegative_int(
                minimum_phase_documents[phase],
                f"{phase} minimum documents",
            )
        if not any(minimum_phase_documents.values()):
            raise PrerequisiteError("concept has no required curriculum exposure")
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
    expected_document_texts: list[str],
) -> dict[str, Any]:
    """Audit whether ordered concept exposures respect declared prerequisites."""

    taxonomy = validate_taxonomy_payload(taxonomy_payload)
    if (
        not isinstance(annotations, list)
        or not annotations
        or len(annotations) != len(expected_document_identities)
        or len(annotations) != len(expected_document_texts)
    ):
        raise PrerequisiteError("annotation population differs")
    expected = [
        _sha256(identity, "expected document identity")
        for identity in expected_document_identities
    ]
    if len(expected) != len(set(expected)):
        raise PrerequisiteError("expected document identity is duplicated")

    state = _ProgressionState(taxonomy)
    for index, (raw_annotation, expected_identity, expected_text) in enumerate(
        zip(annotations, expected, expected_document_texts, strict=True)
    ):
        if not isinstance(expected_text, str) or not expected_text:
            raise PrerequisiteError("expected document text differs")
        state.add(index, raw_annotation, expected_identity, expected_text=expected_text)
    return state.report(
        documents=len(annotations),
        ordered_document_identity_sha256=canonical_sha256(expected),
        annotations_sha256=canonical_sha256(annotations),
    )


class _ProgressionState:
    def __init__(self, taxonomy: dict[str, Any]) -> None:
        self.taxonomy = taxonomy
        self.concepts = {item["concept_id"]: item for item in taxonomy["concepts"]}
        self.minimum_confidence = taxonomy["minimum_annotation_confidence_ppm"]
        self.minimum_evidence_codepoints = taxonomy[
            "minimum_evidence_codepoints_per_positive_label"
        ]
        self.maximum_new_concepts_per_document = taxonomy[
            "maximum_new_concepts_per_document"
        ]
        self.prior_counts: Counter[str] = Counter()
        self.phase_counts: Counter[str] = Counter()
        self.concept_counts: Counter[str] = Counter()
        self.concept_phase_counts: Counter[tuple[str, str]] = Counter()
        self.first_exposure: dict[str, int] = {}
        self.violations: list[dict[str, Any]] = []
        self.premature_exposure_violations: list[dict[str, Any]] = []
        self.concept_density_violations: list[dict[str, Any]] = []
        self.previous_phase = -1

    def add(
        self,
        index: int,
        raw_annotation: Any,
        expected_identity: str,
        *,
        expected_phase: str | None = None,
        expected_text: str,
    ) -> None:
        annotation = _exact_keys(raw_annotation, _ANNOTATION_KEYS, "annotation row")
        if annotation["schema"] != ANNOTATION_SCHEMA:
            raise PrerequisiteError("annotation row schema differs")
        identity = _sha256(
            annotation["document_identity_sha256"], "annotation document identity"
        )
        if identity != expected_identity:
            raise PrerequisiteError("annotation document order differs")
        phase = annotation["phase"]
        if phase not in PHASES or (
            expected_phase is not None and phase != expected_phase
        ):
            raise PrerequisiteError("annotation phase differs")
        phase_index = PHASES.index(phase)
        if phase_index < self.previous_phase:
            raise PrerequisiteError("annotation phases are not monotonic")
        self.previous_phase = phase_index
        self.phase_counts[phase] += 1
        raw_evidence = annotation["concepts"]
        if not isinstance(raw_evidence, list):
            raise PrerequisiteError("annotation concepts differ")
        seen: set[str] = set()
        confident: list[str] = []
        new_concepts: list[str] = []
        for raw_item in raw_evidence:
            item = _exact_keys(raw_item, _EVIDENCE_KEYS, "concept evidence")
            concept_id = item["concept_id"]
            if concept_id not in self.concepts or concept_id in seen:
                raise PrerequisiteError("annotation concept differs or is duplicated")
            seen.add(concept_id)
            confidence = _ppm(item["confidence_ppm"], "annotation confidence")
            spans = item["evidence_spans"]
            if not isinstance(spans, list) or not spans:
                raise PrerequisiteError("annotation evidence spans differ")
            previous_end = -1
            evidence_codepoints = 0
            for raw_span in spans:
                span = _exact_keys(raw_span, _SPAN_KEYS, "annotation evidence span")
                start = _nonnegative_int(span["start"], "evidence span start")
                end = _nonnegative_int(span["end"], "evidence span end")
                if not previous_end <= start < end <= len(expected_text):
                    raise PrerequisiteError("annotation evidence span differs")
                expected_span_hash = hashlib.sha256(
                    expected_text[start:end].encode("utf-8")
                ).hexdigest()
                if _sha256(span["text_sha256"], "annotation evidence span") != (
                    expected_span_hash
                ):
                    raise PrerequisiteError("annotation evidence text differs")
                evidence_codepoints += end - start
                previous_end = end
            if evidence_codepoints < self.minimum_evidence_codepoints:
                raise PrerequisiteError("annotation evidence is too short")
            if confidence < self.minimum_confidence:
                continue
            confident.append(concept_id)
            if concept_id not in self.first_exposure:
                new_concepts.append(concept_id)
            self.concept_counts[concept_id] += 1
            self.concept_phase_counts[(concept_id, phase)] += 1
            self.first_exposure.setdefault(concept_id, index)
            concept = self.concepts[concept_id]
            earliest_phase = next(
                candidate
                for candidate in PHASES
                if concept["minimum_phase_documents"][candidate] > 0
            )
            if phase_index < PHASES.index(earliest_phase):
                self.premature_exposure_violations.append(
                    {
                        "document_index": index,
                        "document_identity_sha256": identity,
                        "concept_id": concept_id,
                        "observed_phase": phase,
                        "earliest_permitted_phase": earliest_phase,
                    }
                )
            minimum = concept["minimum_prior_documents"]
            for prerequisite in concept["prerequisites"]:
                observed = self.prior_counts[prerequisite]
                if observed < minimum:
                    self.violations.append(
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
        if len(new_concepts) > self.maximum_new_concepts_per_document:
            self.concept_density_violations.append(
                {
                    "document_index": index,
                    "document_identity_sha256": identity,
                    "phase": phase,
                    "new_concepts": sorted(new_concepts),
                    "observed_new_concepts": len(new_concepts),
                    "maximum_new_concepts": self.maximum_new_concepts_per_document,
                }
            )
        self.prior_counts.update(confident)

    def report(
        self,
        *,
        documents: int,
        ordered_document_identity_sha256: str,
        annotations_sha256: str,
        lineage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if set(self.phase_counts) != set(PHASES):
            raise PrerequisiteError("annotation population does not cover every phase")
        missing_concepts = sorted(set(self.concepts) - set(self.concept_counts))
        phase_coverage_violations = []
        for concept_id, concept in sorted(self.concepts.items()):
            for phase in PHASES:
                required = concept["minimum_phase_documents"][phase]
                observed = self.concept_phase_counts[(concept_id, phase)]
                if observed < required:
                    phase_coverage_violations.append(
                        {
                            "concept_id": concept_id,
                            "phase": phase,
                            "required_documents": required,
                            "observed_documents": observed,
                        }
                    )
        progression_qualified = (
            not self.violations
            and not self.premature_exposure_violations
            and not self.concept_density_violations
            and not missing_concepts
            and not phase_coverage_violations
        )
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "status": "qualified" if progression_qualified else "not_qualified",
            "taxonomy_receipt_sha256": self.taxonomy["receipt_sha256"],
            "documents": documents,
            "ordered_document_identity_sha256": ordered_document_identity_sha256,
            "annotations_sha256": annotations_sha256,
            "phase_documents": {phase: self.phase_counts[phase] for phase in PHASES},
            "concepts": {
                concept_id: {
                    "confident_documents": self.concept_counts[concept_id],
                    "first_document_index": self.first_exposure.get(concept_id),
                    "earliest_permitted_phase": next(
                        phase
                        for phase in PHASES
                        if self.concepts[concept_id]["minimum_phase_documents"][phase]
                        > 0
                    ),
                    "phase_documents": {
                        phase: self.concept_phase_counts[(concept_id, phase)]
                        for phase in PHASES
                    },
                }
                for concept_id in sorted(self.concepts)
            },
            "missing_concepts": missing_concepts,
            "violations": self.violations,
            "premature_exposure_violations": self.premature_exposure_violations,
            "concept_density_violations": self.concept_density_violations,
            "phase_coverage_violations": phase_coverage_violations,
            "progression_qualified": progression_qualified,
            "training_authorized": False,
            "four_b_training_authorized": False,
        }
        if lineage is not None:
            report["curriculum_lineage"] = lineage
        report["receipt_sha256"] = canonical_sha256(report)
        return report


class _CanonicalListHasher:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.digest.update(b"[")
        self.count = 0

    def add(self, value: Any) -> None:
        if self.count:
            self.digest.update(b",")
        self.digest.update(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        self.count += 1

    def hexdigest(self) -> str:
        copy = self.digest.copy()
        copy.update(b"]")
        return copy.hexdigest()


def _open_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise PrerequisiteError(f"{label} is missing or unsafe") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise PrerequisiteError(f"{label} is missing or unsafe")
    return descriptor, metadata


def _atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise PrerequisiteError("progression output path is unsafe or already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise PrerequisiteError("progression output parent is missing or unsafe")
    encoded = (json.dumps(report, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise PrerequisiteError("progression output already exists") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def replay_curriculum_annotation_files(
    taxonomy_path: Path,
    curriculum_receipt_path: Path,
    annotations_path: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Replay a qualified curriculum and its semantic annotations without writes."""

    taxonomy = _read_taxonomy(taxonomy_path)
    curriculum = validate_curriculum(curriculum_receipt_path, workers=workers)
    curriculum_output = Path(curriculum["output"]["path"])
    curriculum_descriptor, curriculum_before = _open_regular(
        curriculum_output, "curriculum output"
    )
    annotations_descriptor, annotations_before = _open_regular(
        annotations_path, "curriculum annotations"
    )
    state = _ProgressionState(taxonomy)
    identities = _CanonicalListHasher()
    annotation_population = _CanonicalListHasher()
    curriculum_file_hash = hashlib.sha256()
    annotation_file_hash = hashlib.sha256()
    curriculum_bytes = 0
    annotation_bytes = 0
    document_index = 0
    try:
        with (
            os.fdopen(curriculum_descriptor, "rb", closefd=False) as curriculum_handle,
            os.fdopen(annotations_descriptor, "rb", closefd=False) as annotation_handle,
        ):
            for phase in PHASES:
                expected_documents = curriculum["phases"][phase]["documents"]
                for _ in range(expected_documents):
                    curriculum_line = curriculum_handle.readline()
                    annotation_line = annotation_handle.readline()
                    if not curriculum_line or not annotation_line:
                        raise PrerequisiteError(
                            "curriculum or annotation population ended early"
                        )
                    curriculum_bytes += len(curriculum_line)
                    annotation_bytes += len(annotation_line)
                    curriculum_file_hash.update(curriculum_line)
                    annotation_file_hash.update(annotation_line)
                    try:
                        row = normalize_document(json.loads(curriculum_line))
                        annotation = json.loads(annotation_line)
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        RuntimeError,
                    ) as error:
                        raise PrerequisiteError(
                            "curriculum or annotation row differs"
                        ) from error
                    identity = row["identity_sha256"]
                    state.add(
                        document_index,
                        annotation,
                        identity,
                        expected_phase=phase,
                        expected_text=row["text"],
                    )
                    identities.add(identity)
                    annotation_population.add(annotation)
                    document_index += 1
            if curriculum_handle.readline() or annotation_handle.readline():
                raise PrerequisiteError(
                    "curriculum or annotation population has undeclared rows"
                )
        curriculum_after = os.fstat(curriculum_descriptor)
        annotations_after = os.fstat(annotations_descriptor)
    finally:
        os.close(curriculum_descriptor)
        os.close(annotations_descriptor)

    if (
        curriculum_before.st_dev,
        curriculum_before.st_ino,
        curriculum_before.st_nlink,
        curriculum_before.st_size,
        curriculum_before.st_mtime_ns,
    ) != (
        curriculum_after.st_dev,
        curriculum_after.st_ino,
        curriculum_after.st_nlink,
        curriculum_after.st_size,
        curriculum_after.st_mtime_ns,
    ):
        raise PrerequisiteError("curriculum output changed while reading")
    if (
        annotations_before.st_dev,
        annotations_before.st_ino,
        annotations_before.st_nlink,
        annotations_before.st_size,
        annotations_before.st_mtime_ns,
    ) != (
        annotations_after.st_dev,
        annotations_after.st_ino,
        annotations_after.st_nlink,
        annotations_after.st_size,
        annotations_after.st_mtime_ns,
    ):
        raise PrerequisiteError("curriculum annotations changed while reading")
    if annotation_bytes != annotations_before.st_size:
        raise PrerequisiteError("curriculum annotation size differs")
    if (
        curriculum_bytes != curriculum["output"]["bytes"]
        or curriculum_file_hash.hexdigest() != curriculum["output"]["sha256"]
    ):
        raise PrerequisiteError("curriculum output lineage differs")
    report = state.report(
        documents=document_index,
        ordered_document_identity_sha256=identities.hexdigest(),
        annotations_sha256=annotation_population.hexdigest(),
        lineage={
            "curriculum_receipt_sha256": curriculum["receipt_sha256"],
            "curriculum_output_bytes": curriculum_bytes,
            "curriculum_output_sha256": curriculum_file_hash.hexdigest(),
            "annotations_path": str(annotations_path.resolve()),
            "annotations_bytes": annotation_bytes,
            "annotations_file_sha256": annotation_file_hash.hexdigest(),
        },
    )
    return report


def analyze_curriculum_annotation_files(
    taxonomy_path: Path,
    curriculum_receipt_path: Path,
    annotations_path: Path,
    output_path: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Replay a qualified curriculum and atomically publish its semantic report."""

    if (
        not output_path.is_absolute()
        or output_path.exists()
        or output_path.is_symlink()
    ):
        raise PrerequisiteError("progression output path is unsafe or already exists")
    report = replay_curriculum_annotation_files(
        taxonomy_path,
        curriculum_receipt_path,
        annotations_path,
        workers=workers,
    )
    _atomic_write_report(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-taxonomy")
    build.add_argument("--concepts", type=Path, required=True)
    build.add_argument("--annotator-identity", type=Path, required=True)
    build.add_argument("--annotation-policy", type=Path, required=True)
    build.add_argument("--audit-sample-receipt", type=Path, required=True)
    build.add_argument(
        "--annotation-method", choices=sorted(ANNOTATION_METHODS), required=True
    )
    build.add_argument("--minimum-confidence-ppm", type=int, required=True)
    build.add_argument("--maximum-new-concepts-per-document", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate-taxonomy")
    validate.add_argument("--taxonomy", type=Path, required=True)
    audit = subparsers.add_parser("audit-curriculum")
    audit.add_argument("--taxonomy", type=Path, required=True)
    audit.add_argument("--curriculum-receipt", type=Path, required=True)
    audit.add_argument("--annotations", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build-taxonomy":
        payload = build_taxonomy(
            args.concepts,
            args.annotator_identity,
            args.annotation_policy,
            args.audit_sample_receipt,
            args.output,
            annotation_method=args.annotation_method,
            minimum_annotation_confidence_ppm=args.minimum_confidence_ppm,
            maximum_new_concepts_per_document=(args.maximum_new_concepts_per_document),
        )
        result = {
            "schema": payload["schema"],
            "status": "created_prospective",
            "receipt_sha256": payload["receipt_sha256"],
            "training_authorized": False,
            "four_b_training_authorized": False,
        }
    elif args.command == "validate-taxonomy":
        payload = _read_taxonomy(args.taxonomy)
        result = {
            "schema": payload["schema"],
            "status": "validated_prospective",
            "receipt_sha256": payload["receipt_sha256"],
            "training_authorized": False,
            "four_b_training_authorized": False,
        }
    else:
        payload = analyze_curriculum_annotation_files(
            args.taxonomy,
            args.curriculum_receipt,
            args.annotations,
            args.output,
            workers=args.workers,
        )
        result = {
            "schema": payload["schema"],
            "status": payload["status"],
            "receipt_sha256": payload["receipt_sha256"],
            "progression_qualified": payload["progression_qualified"],
            "training_authorized": False,
            "four_b_training_authorized": False,
        }
    print(
        json.dumps(
            result,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
