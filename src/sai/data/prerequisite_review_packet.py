"""Freeze a phase-blinded semantic-prerequisite review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sai.data.prerequisite_review import (
    ANNOTATION_SCHEMA,
    PrerequisiteReviewError,
    _load_concepts,
    _population_descriptor,
    _read_regular,
    _validate_annotations,
    _validate_population,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-semantic-prerequisite-blind-review-packet-v1"
REVIEW_SCHEMA = "sai-semantic-prerequisite-blind-review-row-v1"
KEY_SCHEMA = "sai-semantic-prerequisite-blind-review-key-row-v1"
BLIND_ANNOTATION_SCHEMA = "sai-semantic-prerequisite-blind-annotation-row-v1"
COMPILE_SCHEMA = "sai-semantic-prerequisite-blind-annotation-compilation-v1"
SELECTION_SALT = b"sai-semantic-prerequisite-blind-review-packet-v1"
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "population",
    "selection",
    "review_output",
    "key_output",
    "limitations",
    "receipt_sha256",
}
_REVIEW_KEYS = {
    "schema",
    "review_identity_sha256",
    "text_sha256",
    "text",
    "requested_annotation",
}
_KEY_KEYS = {
    "schema",
    "review_identity_sha256",
    "document_identity_sha256",
    "document_index",
    "phase",
    "surface_band",
    "source",
    "selection_rank_sha256",
    "text_sha256",
}
_DESCRIPTOR_KEYS = {"path", "bytes", "rows", "sha256", "ordered_sha256"}
_BLIND_ANNOTATION_KEYS = {"schema", "review_identity_sha256", "concepts"}
_COMPILE_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "packet",
    "blind_annotations",
    "concept_list",
    "compiled_annotations",
    "limitations",
    "receipt_sha256",
}
_REQUESTED_ANNOTATION = {
    "concepts": [],
    "instructions": (
        "Select only concepts explicitly taught or assumed by this text; cite exact "
        "character spans and confidence for every selected concept."
    ),
}


class PrerequisiteReviewPacketError(RuntimeError):
    """The population, blinded packet, or hidden review key differs."""


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _write_create_only(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PrerequisiteReviewPacketError("review output write failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _descriptor(
    path: Path, encoded: bytes, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "rows": len(rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "ordered_sha256": canonical_sha256(rows),
    }


def _load_population(
    population_receipt: Path, *, curriculum_workers: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = _validate_population(
            population_receipt, curriculum_workers=curriculum_workers
        )
        receipt_bytes = _read_regular(population_receipt, "audit population receipt")
        descriptor, raw_rows = _population_descriptor(
            population_receipt, payload, receipt_bytes
        )
    except PrerequisiteReviewError as error:
        raise PrerequisiteReviewPacketError(
            "audit population validation failed"
        ) from error
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise PrerequisiteReviewPacketError("audit population row differs")
        rows.append(raw)
    return descriptor, rows


def _derive_rows(
    population: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviews: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    for row in population:
        required = {
            "document_identity_sha256",
            "document_index",
            "phase",
            "surface_band",
            "source",
            "selection_rank_sha256",
            "text",
        }
        if not required.issubset(row):
            raise PrerequisiteReviewPacketError("audit population row differs")
        identity = row["document_identity_sha256"]
        text = row["text"]
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or not isinstance(text, str)
        ):
            raise PrerequisiteReviewPacketError("audit population row differs")
        try:
            review_identity = hashlib.sha256(
                SELECTION_SALT + bytes.fromhex(identity)
            ).hexdigest()
        except ValueError as error:
            raise PrerequisiteReviewPacketError(
                "audit population identity differs"
            ) from error
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        reviews.append(
            {
                "schema": REVIEW_SCHEMA,
                "review_identity_sha256": review_identity,
                "text_sha256": text_sha256,
                "text": text,
                "requested_annotation": _REQUESTED_ANNOTATION,
            }
        )
        keys.append(
            {
                "schema": KEY_SCHEMA,
                "review_identity_sha256": review_identity,
                "document_identity_sha256": identity,
                "document_index": row["document_index"],
                "phase": row["phase"],
                "surface_band": row["surface_band"],
                "source": row["source"],
                "selection_rank_sha256": row["selection_rank_sha256"],
                "text_sha256": text_sha256,
            }
        )
    reviews.sort(key=lambda row: row["review_identity_sha256"])
    keys.sort(key=lambda row: row["review_identity_sha256"])
    identities = [row["review_identity_sha256"] for row in reviews]
    if (
        len(reviews) != 120
        or len(keys) != 120
        or len(set(identities)) != 120
        or identities != [row["review_identity_sha256"] for row in keys]
    ):
        raise PrerequisiteReviewPacketError("blind review population differs")
    return reviews, keys


def build_packet(
    population_receipt: Path,
    review_output: Path,
    key_output: Path,
    receipt_output: Path,
    *,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Publish one phase-blinded packet and a separate hidden key."""

    outputs = [review_output.resolve(), key_output.resolve(), receipt_output.resolve()]
    if len(set(outputs)) != 3 or any(
        path.exists() or path.is_symlink() for path in outputs
    ):
        raise PrerequisiteReviewPacketError("review output boundary differs")
    population_descriptor, population = _load_population(
        population_receipt, curriculum_workers=curriculum_workers
    )
    reviews, keys = _derive_rows(population)
    review_encoded = _jsonl_bytes(reviews)
    key_encoded = _jsonl_bytes(keys)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "awaiting_blind_independent_review",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "population": population_descriptor,
        "selection": {
            "rows": 120,
            "ordering": "ascending_sha256_of_salt_plus_document_identity",
            "review_hides_phase_surface_band_source_and_document_order": True,
            "hidden_key_is_separate": True,
        },
        "review_output": _descriptor(review_output, review_encoded, reviews),
        "key_output": _descriptor(key_output, key_encoded, keys),
        "limitations": [
            "packet_contains_no_completed_annotations",
            "packet_does_not_qualify_the_semantic_taxonomy",
            "packet_authorizes_no_training_or_architecture_promotion",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = (
        json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    created: list[Path] = []
    try:
        for path, encoded in (
            (review_output, review_encoded),
            (key_output, key_encoded),
            (receipt_output, receipt_encoded),
        ):
            _write_create_only(path, encoded)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return validate_packet(
        population_receipt,
        review_output,
        key_output,
        receipt_output,
        curriculum_workers=curriculum_workers,
    )


def validate_packet(
    population_receipt: Path,
    review_output: Path,
    key_output: Path,
    receipt_output: Path,
    *,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Replay the population, blinded packet, and separate hidden key."""

    population_descriptor, population = _load_population(
        population_receipt, curriculum_workers=curriculum_workers
    )
    expected_reviews, expected_keys = _derive_rows(population)
    try:
        review_encoded = _read_regular(review_output, "blind review packet")
        key_encoded = _read_regular(key_output, "blind review key")
        receipt_encoded = _read_regular(receipt_output, "blind review receipt")
        reviews = [
            json.loads(line) for line in review_encoded.decode("utf-8").splitlines()
        ]
        keys = [json.loads(line) for line in key_encoded.decode("utf-8").splitlines()]
        payload = json.loads(receipt_encoded.decode("utf-8"))
    except (PrerequisiteReviewError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteReviewPacketError("review packet is unreadable") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_KEYS
        or payload["schema"] != SCHEMA
        or payload["status"] != "awaiting_blind_independent_review"
        or payload["training_authorized"] is not False
        or payload["four_b_training_authorized"] is not False
        or payload["population"] != population_descriptor
        or payload["receipt_sha256"]
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "receipt_sha256"}
        )
        or reviews != expected_reviews
        or keys != expected_keys
    ):
        raise PrerequisiteReviewPacketError("review receipt differs")
    for descriptor, path, encoded, rows in (
        (payload["review_output"], review_output, review_encoded, reviews),
        (payload["key_output"], key_output, key_encoded, keys),
    ):
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != _DESCRIPTOR_KEYS
            or descriptor != _descriptor(path, encoded, rows)
        ):
            raise PrerequisiteReviewPacketError("review output differs")
    if any(set(row) != _REVIEW_KEYS for row in reviews) or any(
        set(row) != _KEY_KEYS for row in keys
    ):
        raise PrerequisiteReviewPacketError("review row fields differ")
    if set(payload["selection"]) != {
        "rows",
        "ordering",
        "review_hides_phase_surface_band_source_and_document_order",
        "hidden_key_is_separate",
    } or payload["selection"] != {
        "rows": 120,
        "ordering": "ascending_sha256_of_salt_plus_document_identity",
        "review_hides_phase_surface_band_source_and_document_order": True,
        "hidden_key_is_separate": True,
    }:
        raise PrerequisiteReviewPacketError("blind selection differs")
    if payload["limitations"] != [
        "packet_contains_no_completed_annotations",
        "packet_does_not_qualify_the_semantic_taxonomy",
        "packet_authorizes_no_training_or_architecture_promotion",
    ]:
        raise PrerequisiteReviewPacketError("review limitations differ")
    return payload


def _read_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        encoded = _read_regular(path, label)
        rows = [json.loads(line) for line in encoded.decode("utf-8").splitlines()]
    except (PrerequisiteReviewError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteReviewPacketError(f"{label} is unreadable") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise PrerequisiteReviewPacketError(f"{label} differs")
    return rows, encoded


def _compile_rows(
    *,
    population: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
    concepts: set[str],
) -> list[dict[str, Any]]:
    review_identities = [row["review_identity_sha256"] for row in review_rows]
    if (
        len(blind_rows) != 120
        or any(set(row) != _BLIND_ANNOTATION_KEYS for row in blind_rows)
        or [row["review_identity_sha256"] for row in blind_rows] != review_identities
        or any(row["schema"] != BLIND_ANNOTATION_SCHEMA for row in blind_rows)
        or any(not isinstance(row["concepts"], list) for row in blind_rows)
    ):
        raise PrerequisiteReviewPacketError("blind annotations differ")
    blind_by_identity = {
        row["review_identity_sha256"]: row["concepts"] for row in blind_rows
    }
    if len(blind_by_identity) != 120:
        raise PrerequisiteReviewPacketError("blind annotation identities differ")
    compiled: list[dict[str, Any]] = []
    for document in population:
        review_identity = hashlib.sha256(
            SELECTION_SALT + bytes.fromhex(document["document_identity_sha256"])
        ).hexdigest()
        if review_identity not in blind_by_identity:
            raise PrerequisiteReviewPacketError("blind annotation identity differs")
        compiled.append(
            {
                "schema": ANNOTATION_SCHEMA,
                "document_identity_sha256": document["document_identity_sha256"],
                "phase": document["phase"],
                "concepts": blind_by_identity[review_identity],
            }
        )
    try:
        _validate_annotations(compiled, population, concepts, "compiled")
    except PrerequisiteReviewError as error:
        raise PrerequisiteReviewPacketError(
            "compiled annotation evidence differs"
        ) from error
    return compiled


def _compilation_inputs(
    *,
    population_receipt: Path,
    review_output: Path,
    key_output: Path,
    packet_receipt: Path,
    blind_annotations: Path,
    concept_list: Path,
    curriculum_workers: int,
) -> tuple[
    dict[str, Any],
    bytes,
    list[dict[str, Any]],
    bytes,
    bytes,
    list[dict[str, Any]],
]:
    packet_payload = validate_packet(
        population_receipt,
        review_output,
        key_output,
        packet_receipt,
        curriculum_workers=curriculum_workers,
    )
    _population_descriptor_value, population = _load_population(
        population_receipt, curriculum_workers=curriculum_workers
    )
    review_rows, _review_encoded = _read_jsonl(review_output, "blind review packet")
    blind_rows, blind_encoded = _read_jsonl(blind_annotations, "blind annotations")
    try:
        concepts, concept_encoded = _load_concepts(concept_list)
        packet_receipt_encoded = _read_regular(packet_receipt, "blind review receipt")
    except PrerequisiteReviewError as error:
        raise PrerequisiteReviewPacketError("compilation input differs") from error
    compiled = _compile_rows(
        population=population,
        review_rows=review_rows,
        blind_rows=blind_rows,
        concepts=concepts,
    )
    return (
        packet_payload,
        packet_receipt_encoded,
        blind_rows,
        blind_encoded,
        concept_encoded,
        compiled,
    )


def compile_annotations(
    *,
    population_receipt: Path,
    review_output: Path,
    key_output: Path,
    packet_receipt: Path,
    blind_annotations: Path,
    concept_list: Path,
    output: Path,
    compilation_receipt: Path,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Restore hidden identities only after one blind annotation file is frozen."""

    outputs = [output.resolve(), compilation_receipt.resolve()]
    if len(set(outputs)) != 2 or any(
        path.exists() or path.is_symlink() for path in outputs
    ):
        raise PrerequisiteReviewPacketError("compilation output boundary differs")
    (
        packet_payload,
        packet_receipt_encoded,
        blind_rows,
        blind_encoded,
        concept_encoded,
        compiled,
    ) = _compilation_inputs(
        population_receipt=population_receipt,
        review_output=review_output,
        key_output=key_output,
        packet_receipt=packet_receipt,
        blind_annotations=blind_annotations,
        concept_list=concept_list,
        curriculum_workers=curriculum_workers,
    )
    compiled_encoded = _jsonl_bytes(compiled)
    payload: dict[str, Any] = {
        "schema": COMPILE_SCHEMA,
        "status": "compiled_blind_annotations_unreviewed",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "packet": {
            "path": str(packet_receipt.resolve()),
            "bytes": len(packet_receipt_encoded),
            "sha256": hashlib.sha256(packet_receipt_encoded).hexdigest(),
            "receipt_sha256": packet_payload["receipt_sha256"],
        },
        "blind_annotations": _descriptor(blind_annotations, blind_encoded, blind_rows),
        "concept_list": {
            "path": str(concept_list.resolve()),
            "bytes": len(concept_encoded),
            "sha256": hashlib.sha256(concept_encoded).hexdigest(),
        },
        "compiled_annotations": _descriptor(output, compiled_encoded, compiled),
        "limitations": [
            "one_compilation_is_not_independent_review",
            "compilation_does_not_qualify_the_semantic_taxonomy",
            "compilation_authorizes_no_training_or_architecture_promotion",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = (
        json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    created: list[Path] = []
    try:
        for path, encoded in (
            (output, compiled_encoded),
            (compilation_receipt, receipt_encoded),
        ):
            _write_create_only(path, encoded)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return validate_compilation(
        population_receipt=population_receipt,
        review_output=review_output,
        key_output=key_output,
        packet_receipt=packet_receipt,
        blind_annotations=blind_annotations,
        concept_list=concept_list,
        output=output,
        compilation_receipt=compilation_receipt,
        curriculum_workers=curriculum_workers,
    )


def validate_compilation(
    *,
    population_receipt: Path,
    review_output: Path,
    key_output: Path,
    packet_receipt: Path,
    blind_annotations: Path,
    concept_list: Path,
    output: Path,
    compilation_receipt: Path,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Replay one blind-to-canonical annotation compilation."""

    (
        packet_payload,
        packet_receipt_encoded,
        blind_rows,
        blind_encoded,
        concept_encoded,
        compiled,
    ) = _compilation_inputs(
        population_receipt=population_receipt,
        review_output=review_output,
        key_output=key_output,
        packet_receipt=packet_receipt,
        blind_annotations=blind_annotations,
        concept_list=concept_list,
        curriculum_workers=curriculum_workers,
    )
    compiled_encoded = _jsonl_bytes(compiled)
    try:
        actual_output = _read_regular(output, "compiled annotations")
        receipt_encoded = _read_regular(
            compilation_receipt, "annotation compilation receipt"
        )
        payload = json.loads(receipt_encoded.decode("utf-8"))
    except (PrerequisiteReviewError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteReviewPacketError(
            "annotation compilation is unreadable"
        ) from error
    expected: dict[str, Any] = {
        "schema": COMPILE_SCHEMA,
        "status": "compiled_blind_annotations_unreviewed",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "packet": {
            "path": str(packet_receipt.resolve()),
            "bytes": len(packet_receipt_encoded),
            "sha256": hashlib.sha256(packet_receipt_encoded).hexdigest(),
            "receipt_sha256": packet_payload["receipt_sha256"],
        },
        "blind_annotations": _descriptor(blind_annotations, blind_encoded, blind_rows),
        "concept_list": {
            "path": str(concept_list.resolve()),
            "bytes": len(concept_encoded),
            "sha256": hashlib.sha256(concept_encoded).hexdigest(),
        },
        "compiled_annotations": _descriptor(output, compiled_encoded, compiled),
        "limitations": [
            "one_compilation_is_not_independent_review",
            "compilation_does_not_qualify_the_semantic_taxonomy",
            "compilation_authorizes_no_training_or_architecture_promotion",
        ],
    }
    expected["receipt_sha256"] = canonical_sha256(expected)
    if (
        not isinstance(payload, dict)
        or set(payload) != _COMPILE_TOP_KEYS
        or payload != expected
        or actual_output != compiled_encoded
    ):
        raise PrerequisiteReviewPacketError("annotation compilation differs")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_packet_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--population-receipt", type=Path, required=True)
        target.add_argument("--review-output", type=Path, required=True)
        target.add_argument("--key-output", type=Path, required=True)
        target.add_argument("--receipt-output", type=Path, required=True)
        target.add_argument("--curriculum-workers", type=int, default=1)

    add_packet_arguments(subparsers.add_parser("build"))
    add_packet_arguments(subparsers.add_parser("validate"))
    for name in ("compile", "validate-compile"):
        target = subparsers.add_parser(name)
        target.add_argument("--population-receipt", type=Path, required=True)
        target.add_argument("--review-output", type=Path, required=True)
        target.add_argument("--key-output", type=Path, required=True)
        target.add_argument("--packet-receipt", type=Path, required=True)
        target.add_argument("--blind-annotations", type=Path, required=True)
        target.add_argument("--concept-list", type=Path, required=True)
        target.add_argument("--output", type=Path, required=True)
        target.add_argument("--compilation-receipt", type=Path, required=True)
        target.add_argument("--curriculum-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command in {"build", "validate"}:
        function = build_packet if args.command == "build" else validate_packet
        payload = function(
            args.population_receipt,
            args.review_output,
            args.key_output,
            args.receipt_output,
            curriculum_workers=args.curriculum_workers,
        )
        rows = payload["selection"]["rows"]
    else:
        function = (
            compile_annotations if args.command == "compile" else validate_compilation
        )
        payload = function(
            population_receipt=args.population_receipt,
            review_output=args.review_output,
            key_output=args.key_output,
            packet_receipt=args.packet_receipt,
            blind_annotations=args.blind_annotations,
            concept_list=args.concept_list,
            output=args.output,
            compilation_receipt=args.compilation_receipt,
            curriculum_workers=args.curriculum_workers,
        )
        rows = payload["compiled_annotations"]["rows"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": rows,
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
