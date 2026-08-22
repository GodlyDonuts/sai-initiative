"""Freeze a phase-blinded semantic-prerequisite review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sai.data.prerequisite_review import (
    PrerequisiteReviewError,
    _population_descriptor,
    _read_regular,
    _validate_population,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-semantic-prerequisite-blind-review-packet-v1"
REVIEW_SCHEMA = "sai-semantic-prerequisite-blind-review-row-v1"
KEY_SCHEMA = "sai-semantic-prerequisite-blind-review-key-row-v1"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--population-receipt", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--curriculum-workers", type=int, default=1)
    args = parser.parse_args(argv)
    function = build_packet if args.command == "build" else validate_packet
    payload = function(
        args.population_receipt,
        args.review_output,
        args.key_output,
        args.receipt_output,
        curriculum_workers=args.curriculum_workers,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["selection"]["rows"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
