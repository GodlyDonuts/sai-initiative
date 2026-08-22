"""Audit semantic prerequisite annotations against an independent review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from sai.data.annotation_policy import validate_policy
from sai.data.prerequisite_sample import validate_audit_population
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-semantic-prerequisite-audit-sample-v2"
ANNOTATION_SCHEMA = "sai-prerequisite-document-annotation-v1"
CONCEPT_LIST_SCHEMA = "sai-semantic-prerequisite-concept-list-v1"
MINIMUM_REVIEWED_DOCUMENTS = 100
MAXIMUM_ALLOWED_DISAGREEMENT_PPM = 50_000
_MAX_FILE_BYTES = 64 << 20
_TOP_KEYS = {
    "schema",
    "status",
    "audit_qualified",
    "training_authorized",
    "four_b_training_authorized",
    "annotator_identity_sha256",
    "annotation_policy_sha256",
    "reviewer_identity",
    "concept_list",
    "sample_population_sha256",
    "population",
    "annotator_annotations",
    "reviewer_annotations",
    "reviewed_documents",
    "disagreement_documents",
    "observed_disagreement_ppm",
    "maximum_disagreement_ppm",
    "receipt_sha256",
}
_POPULATION_KEYS = {
    "receipt_path",
    "receipt_file_sha256",
    "receipt_sha256",
    "output_path",
    "output_bytes",
    "output_sha256",
    "ordered_population_sha256",
}
_FILE_KEYS = {"path", "bytes", "sha256"}
_ANNOTATION_FILE_KEYS = _FILE_KEYS | {"ordered_annotations_sha256"}
_ANNOTATION_KEYS = {"schema", "document_identity_sha256", "phase", "concepts"}
_EVIDENCE_KEYS = {"concept_id", "confidence_ppm", "evidence_spans"}
_SPAN_KEYS = {"start", "end", "text_sha256"}


class PrerequisiteReviewError(RuntimeError):
    """The population, annotations, evidence, or review receipt differs."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PrerequisiteReviewError(f"{label} fields differ")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PrerequisiteReviewError(f"{label} differs")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise PrerequisiteReviewError(f"{label} differs") from error
    if not any(raw):
        raise PrerequisiteReviewError(f"{label} is a placeholder")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrerequisiteReviewError(f"{label} differs")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise PrerequisiteReviewError(f"{label} is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_FILE_BYTES
        ):
            raise PrerequisiteReviewError(f"{label} is missing or unsafe")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
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
        raise PrerequisiteReviewError(f"{label} changed while reading")
    return encoded


def _json_file(path: Path, label: str) -> tuple[Any, bytes]:
    encoded = _read_regular(path, label)
    try:
        return json.loads(encoded.decode("utf-8")), encoded
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteReviewError(f"{label} JSON differs") from error


def _jsonl(path: Path, label: str) -> tuple[list[Any], bytes]:
    encoded = _read_regular(path, label)
    try:
        lines = encoded.decode("utf-8").splitlines()
        if not lines:
            raise ValueError
        return [json.loads(line) for line in lines], encoded
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PrerequisiteReviewError(f"{label} JSONL differs") from error


def _file_descriptor(path: Path, encoded: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _load_concepts(path: Path) -> tuple[set[str], bytes]:
    payload, encoded = _json_file(path, "concept list")
    payload = _exact(payload, {"schema", "status", "concepts"}, "concept list")
    concepts = payload["concepts"]
    if (
        payload["schema"] != CONCEPT_LIST_SCHEMA
        or payload["status"] != "candidate"
        or not isinstance(concepts, list)
        or not concepts
    ):
        raise PrerequisiteReviewError("concept list differs")
    identities = []
    for concept in concepts:
        if not isinstance(concept, dict) or not isinstance(
            concept.get("concept_id"), str
        ):
            raise PrerequisiteReviewError("concept list differs")
        identities.append(concept["concept_id"])
    if len(identities) != len(set(identities)):
        raise PrerequisiteReviewError("concept identities are duplicated")
    return set(identities), encoded


def _validate_annotations(
    annotations: list[Any], population: list[Any], concepts: set[str], label: str
) -> list[tuple[str, ...]]:
    if len(annotations) != len(population):
        raise PrerequisiteReviewError(f"{label} population differs")
    concept_sets: list[tuple[str, ...]] = []
    for annotation, document in zip(annotations, population, strict=True):
        row = _exact(annotation, _ANNOTATION_KEYS, f"{label} row")
        if (
            row["schema"] != ANNOTATION_SCHEMA
            or row["document_identity_sha256"]
            != document.get("document_identity_sha256")
            or row["phase"] != document.get("phase")
            or not isinstance(row["concepts"], list)
        ):
            raise PrerequisiteReviewError(f"{label} row differs")
        seen: list[str] = []
        for raw_evidence in row["concepts"]:
            evidence = _exact(raw_evidence, _EVIDENCE_KEYS, f"{label} evidence")
            concept_id = evidence["concept_id"]
            confidence = _integer(evidence["confidence_ppm"], f"{label} confidence")
            spans = evidence["evidence_spans"]
            if (
                concept_id not in concepts
                or concept_id in seen
                or confidence == 0
                or confidence > 1_000_000
                or not isinstance(spans, list)
                or not spans
            ):
                raise PrerequisiteReviewError(f"{label} evidence differs")
            seen.append(concept_id)
            text = document.get("text")
            if not isinstance(text, str):
                raise PrerequisiteReviewError("audit population text differs")
            for raw_span in spans:
                span = _exact(raw_span, _SPAN_KEYS, f"{label} evidence span")
                start = _integer(span["start"], f"{label} span start")
                end = _integer(span["end"], f"{label} span end")
                if start >= end or end > len(text):
                    raise PrerequisiteReviewError(f"{label} evidence span differs")
                expected = hashlib.sha256(text[start:end].encode("utf-8")).hexdigest()
                if span["text_sha256"] != expected:
                    raise PrerequisiteReviewError(f"{label} evidence text differs")
        if seen != sorted(seen):
            raise PrerequisiteReviewError(f"{label} concepts are not canonical")
        concept_sets.append(tuple(seen))
    return concept_sets


def _population_descriptor(
    population_receipt: Path, payload: dict[str, Any], receipt_bytes: bytes
) -> tuple[dict[str, Any], list[Any]]:
    selection = payload.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("per_stratum") != 8
        or selection.get("strata") != 16
        or selection.get("selected_documents") != 128
    ):
        raise PrerequisiteReviewError("audit population geometry differs")
    output = Path(payload["output"]["path"])
    population, output_bytes = _jsonl(output, "audit population")
    descriptor = {
        "receipt_path": str(population_receipt.resolve()),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "receipt_sha256": payload["receipt_sha256"],
        "output_path": str(output.resolve()),
        "output_bytes": len(output_bytes),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "ordered_population_sha256": canonical_sha256(population),
    }
    if (
        descriptor["output_bytes"] != payload["output"]["bytes"]
        or descriptor["output_sha256"] != payload["output"]["sha256"]
        or descriptor["ordered_population_sha256"]
        != payload["output"]["ordered_population_sha256"]
    ):
        raise PrerequisiteReviewError("audit population descriptor differs")
    return descriptor, population


def build_review_receipt(
    population_receipt: Path,
    concept_list: Path,
    annotator_identity: Path,
    annotation_policy: Path,
    annotator_annotations: Path,
    reviewer_identity: Path,
    reviewer_annotations: Path,
    output: Path,
    *,
    maximum_disagreement_ppm: int = MAXIMUM_ALLOWED_DISAGREEMENT_PPM,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Create an evidence-replayable annotation-quality receipt."""

    if output.exists() or output.is_symlink():
        raise PrerequisiteReviewError("review receipt already exists")
    if (
        isinstance(maximum_disagreement_ppm, bool)
        or not isinstance(maximum_disagreement_ppm, int)
        or maximum_disagreement_ppm < 0
        or maximum_disagreement_ppm > MAXIMUM_ALLOWED_DISAGREEMENT_PPM
    ):
        raise PrerequisiteReviewError("maximum disagreement differs")
    try:
        population_payload = validate_audit_population(
            population_receipt, curriculum_workers=curriculum_workers
        )
    except Exception as error:
        raise PrerequisiteReviewError("audit population validation failed") from error
    population_receipt_bytes = _read_regular(
        population_receipt, "audit population receipt"
    )
    population_descriptor, population = _population_descriptor(
        population_receipt, population_payload, population_receipt_bytes
    )
    concepts, concept_bytes = _load_concepts(concept_list)
    validate_policy(
        annotation_policy,
        expected_concept_list_sha256=hashlib.sha256(concept_bytes).hexdigest(),
    )
    annotator_identity_bytes = _read_regular(annotator_identity, "annotator identity")
    policy_bytes = _read_regular(annotation_policy, "annotation policy")
    reviewer_identity_bytes = _read_regular(reviewer_identity, "reviewer identity")
    proposed, proposed_bytes = _jsonl(annotator_annotations, "annotator annotations")
    reviewed, reviewed_bytes = _jsonl(reviewer_annotations, "reviewer annotations")
    proposed_sets = _validate_annotations(proposed, population, concepts, "annotator")
    reviewed_sets = _validate_annotations(reviewed, population, concepts, "reviewer")
    disagreements = sum(
        a != b for a, b in zip(proposed_sets, reviewed_sets, strict=True)
    )
    documents = len(population)
    observed = disagreements * 1_000_000 // documents
    qualified = (
        documents >= MINIMUM_REVIEWED_DOCUMENTS and observed <= maximum_disagreement_ppm
    )
    reviewer_descriptor = _file_descriptor(reviewer_identity, reviewer_identity_bytes)
    proposed_descriptor = _file_descriptor(annotator_annotations, proposed_bytes)
    proposed_descriptor["ordered_annotations_sha256"] = canonical_sha256(proposed)
    reviewed_descriptor = _file_descriptor(reviewer_annotations, reviewed_bytes)
    reviewed_descriptor["ordered_annotations_sha256"] = canonical_sha256(reviewed)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if qualified else "failed",
        "audit_qualified": qualified,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "annotator_identity_sha256": hashlib.sha256(
            annotator_identity_bytes
        ).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "reviewer_identity": reviewer_descriptor,
        "concept_list": _file_descriptor(concept_list, concept_bytes),
        "sample_population_sha256": population_descriptor["ordered_population_sha256"],
        "population": population_descriptor,
        "annotator_annotations": proposed_descriptor,
        "reviewer_annotations": reviewed_descriptor,
        "reviewed_documents": documents,
        "disagreement_documents": disagreements,
        "observed_disagreement_ppm": observed,
        "maximum_disagreement_ppm": maximum_disagreement_ppm,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    validate_review_payload(
        payload,
        expected_annotator_identity_sha256=payload["annotator_identity_sha256"],
        expected_annotation_policy_sha256=payload["annotation_policy_sha256"],
        expected_concept_list_sha256=payload["concept_list"]["sha256"],
        curriculum_workers=curriculum_workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    try:
        with stage.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, output)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    return payload


def _check_file_descriptor(raw: Any, keys: set[str], label: str) -> tuple[Path, bytes]:
    descriptor = _exact(raw, keys, label)
    path = Path(descriptor["path"])
    encoded = _read_regular(path, label)
    if descriptor != {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        **(
            {"ordered_annotations_sha256": descriptor["ordered_annotations_sha256"]}
            if "ordered_annotations_sha256" in descriptor
            else {}
        ),
    }:
        raise PrerequisiteReviewError(f"{label} descriptor differs")
    return path, encoded


def validate_review_payload(
    payload: Any,
    *,
    expected_annotator_identity_sha256: str,
    expected_annotation_policy_sha256: str,
    expected_concept_list_sha256: str,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Replay every artifact behind an annotation-quality receipt."""

    receipt = _exact(payload, _TOP_KEYS, "review receipt")
    receipt_hash = _sha256(receipt["receipt_sha256"], "review receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt_hash != canonical_sha256(unsigned):
        raise PrerequisiteReviewError("review receipt hash differs")
    if (
        receipt["schema"] != SCHEMA
        or receipt["training_authorized"] is not False
        or receipt["four_b_training_authorized"] is not False
        or receipt["annotator_identity_sha256"] != expected_annotator_identity_sha256
        or receipt["annotation_policy_sha256"] != expected_annotation_policy_sha256
    ):
        raise PrerequisiteReviewError("review identity boundary differs")
    concept_path, concept_bytes = _check_file_descriptor(
        receipt["concept_list"], _FILE_KEYS, "concept list"
    )
    if hashlib.sha256(concept_bytes).hexdigest() != expected_concept_list_sha256:
        raise PrerequisiteReviewError("review concept-list identity differs")
    concepts, reopened_concept_bytes = _load_concepts(concept_path)
    if reopened_concept_bytes != concept_bytes:
        raise PrerequisiteReviewError("concept list changed while reading")
    population_row = _exact(receipt["population"], _POPULATION_KEYS, "population")
    population_receipt = Path(population_row["receipt_path"])
    try:
        population_payload = validate_audit_population(
            population_receipt, curriculum_workers=curriculum_workers
        )
    except Exception as error:
        raise PrerequisiteReviewError("audit population validation failed") from error
    population_receipt_bytes = _read_regular(
        population_receipt, "audit population receipt"
    )
    expected_population, population = _population_descriptor(
        population_receipt, population_payload, population_receipt_bytes
    )
    if population_row != expected_population:
        raise PrerequisiteReviewError("audit population lineage differs")
    reviewer_identity_path, reviewer_identity_bytes = _check_file_descriptor(
        receipt["reviewer_identity"], _FILE_KEYS, "reviewer identity"
    )
    del reviewer_identity_path
    if (
        len(
            {
                receipt["annotator_identity_sha256"],
                receipt["annotation_policy_sha256"],
                hashlib.sha256(reviewer_identity_bytes).hexdigest(),
            }
        )
        != 3
    ):
        raise PrerequisiteReviewError("review identities are not independent")
    proposed_path, proposed_bytes = _check_file_descriptor(
        receipt["annotator_annotations"],
        _ANNOTATION_FILE_KEYS,
        "annotator annotations",
    )
    reviewed_path, reviewed_bytes = _check_file_descriptor(
        receipt["reviewer_annotations"],
        _ANNOTATION_FILE_KEYS,
        "reviewer annotations",
    )
    if proposed_path.resolve() == reviewed_path.resolve():
        raise PrerequisiteReviewError("annotation reviews are not independent")
    proposed, _ = _jsonl(proposed_path, "annotator annotations")
    reviewed, _ = _jsonl(reviewed_path, "reviewer annotations")
    if (
        canonical_sha256(proposed)
        != receipt["annotator_annotations"]["ordered_annotations_sha256"]
        or canonical_sha256(reviewed)
        != receipt["reviewer_annotations"]["ordered_annotations_sha256"]
        or hashlib.sha256(proposed_bytes).hexdigest()
        != receipt["annotator_annotations"]["sha256"]
        or hashlib.sha256(reviewed_bytes).hexdigest()
        != receipt["reviewer_annotations"]["sha256"]
    ):
        raise PrerequisiteReviewError("annotation population hash differs")
    proposed_sets = _validate_annotations(proposed, population, concepts, "annotator")
    reviewed_sets = _validate_annotations(reviewed, population, concepts, "reviewer")
    disagreements = sum(
        a != b for a, b in zip(proposed_sets, reviewed_sets, strict=True)
    )
    documents = len(population)
    observed = disagreements * 1_000_000 // documents
    reviewed_documents = _integer(receipt["reviewed_documents"], "reviewed documents")
    disagreement_documents = _integer(
        receipt["disagreement_documents"], "disagreement documents"
    )
    observed_receipt = _integer(
        receipt["observed_disagreement_ppm"], "observed disagreement"
    )
    maximum = _integer(receipt["maximum_disagreement_ppm"], "maximum disagreement")
    if maximum > MAXIMUM_ALLOWED_DISAGREEMENT_PPM:
        raise PrerequisiteReviewError("maximum disagreement differs")
    qualified = documents >= MINIMUM_REVIEWED_DOCUMENTS and observed <= maximum
    if (
        receipt["sample_population_sha256"]
        != population_row["ordered_population_sha256"]
        or reviewed_documents != documents
        or disagreement_documents != disagreements
        or observed_receipt != observed
        or receipt["audit_qualified"] is not qualified
        or receipt["status"] != ("passed" if qualified else "failed")
    ):
        raise PrerequisiteReviewError("review qualification arithmetic differs")
    return receipt


def validate_review_receipt(
    receipt_path: Path,
    *,
    expected_annotator_identity_sha256: str,
    expected_annotation_policy_sha256: str,
    expected_concept_list_sha256: str,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    payload, _ = _json_file(receipt_path, "review receipt")
    return validate_review_payload(
        payload,
        expected_annotator_identity_sha256=expected_annotator_identity_sha256,
        expected_annotation_policy_sha256=expected_annotation_policy_sha256,
        expected_concept_list_sha256=expected_concept_list_sha256,
        curriculum_workers=curriculum_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--population-receipt", type=Path, required=True)
    build.add_argument("--concept-list", type=Path, required=True)
    build.add_argument("--annotator-identity", type=Path, required=True)
    build.add_argument("--annotation-policy", type=Path, required=True)
    build.add_argument("--annotator-annotations", type=Path, required=True)
    build.add_argument("--reviewer-identity", type=Path, required=True)
    build.add_argument("--reviewer-annotations", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--maximum-disagreement-ppm",
        type=int,
        default=MAXIMUM_ALLOWED_DISAGREEMENT_PPM,
    )
    build.add_argument("--curriculum-workers", type=int, default=1)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--annotator-identity", type=Path, required=True)
    validate.add_argument("--annotation-policy", type=Path, required=True)
    validate.add_argument("--concept-list", type=Path, required=True)
    validate.add_argument("--curriculum-workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_review_receipt(
            args.population_receipt,
            args.concept_list,
            args.annotator_identity,
            args.annotation_policy,
            args.annotator_annotations,
            args.reviewer_identity,
            args.reviewer_annotations,
            args.output,
            maximum_disagreement_ppm=args.maximum_disagreement_ppm,
            curriculum_workers=args.curriculum_workers,
        )
    else:
        validate_policy(
            args.annotation_policy,
            expected_concept_list_sha256=sha256_file(args.concept_list),
        )
        payload = validate_review_receipt(
            args.receipt,
            expected_annotator_identity_sha256=sha256_file(args.annotator_identity),
            expected_annotation_policy_sha256=sha256_file(args.annotation_policy),
            expected_concept_list_sha256=sha256_file(args.concept_list),
            curriculum_workers=args.curriculum_workers,
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "audit_qualified": payload["audit_qualified"],
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
