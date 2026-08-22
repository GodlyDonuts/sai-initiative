"""Freeze a blinded semantic-review population from an authored curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import (
    AuthoredCurriculumError,
    _read_regular_bytes,
    _write_create_only,
    validate,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-review-packet-receipt-v1"
REVIEW_SCHEMA = "sai-authored-curriculum-blind-review-row-v1"
KEY_SCHEMA = "sai-authored-curriculum-review-key-row-v1"
SELECTION_SALT = b"sai-authored-curriculum-review-packet-v1"
_REVIEW_KEYS = {
    "schema",
    "review_identity_sha256",
    "title",
    "text_sha256",
    "text",
    "requested_review",
}
_KEY_KEYS = {
    "schema",
    "review_identity_sha256",
    "candidate_identity_sha256",
    "source_name",
    "source_revision",
    "source_path",
    "source_order_index",
    "candidate_stage",
    "required_prior_concepts",
    "text_sha256",
}


class AuthoredReviewPacketError(RuntimeError):
    """The candidate, blind packet, or separate review key differs."""


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
        for row in rows
    )


def build(
    *,
    candidate: Path,
    candidate_receipt: Path,
    review_output: Path,
    key_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Publish all candidate chapters in blinded deterministic order."""

    outputs = [review_output.resolve(), key_output.resolve(), receipt_output.resolve()]
    if len(set(outputs)) != 3 or any(path.exists() for path in outputs):
        raise AuthoredReviewPacketError("review output boundary differs")
    try:
        candidate_payload = validate(candidate, candidate_receipt)
        candidate_bytes = _read_regular_bytes(candidate, maximum_bytes=1 << 30)
        receipt_bytes = _read_regular_bytes(candidate_receipt, maximum_bytes=1 << 20)
    except AuthoredCurriculumError as error:
        raise AuthoredReviewPacketError("authored candidate differs") from error
    try:
        candidate_rows = [
            json.loads(line) for line in candidate_bytes.decode().splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewPacketError("authored candidate rows differ") from error
    review_rows = []
    key_rows = []
    for row in candidate_rows:
        identity = row["identity_sha256"]
        review_identity = hashlib.sha256(
            SELECTION_SALT + bytes.fromhex(identity)
        ).hexdigest()
        review_rows.append(
            {
                "schema": REVIEW_SCHEMA,
                "review_identity_sha256": review_identity,
                "title": row["title"],
                "text_sha256": row["source_sha256"],
                "text": row["text"],
                "requested_review": {
                    "instructional_quality": "unlabeled",
                    "assumed_prior_concepts": [],
                    "taught_concepts_with_evidence_spans": [],
                    "extraction_or_factual_defects": [],
                    "admission_recommendation": "unlabeled",
                },
            }
        )
        key_rows.append(
            {
                "schema": KEY_SCHEMA,
                "review_identity_sha256": review_identity,
                "candidate_identity_sha256": identity,
                "source_name": row["source_name"],
                "source_revision": row["source_revision"],
                "source_path": row["source_path"],
                "source_order_index": row["source_order_index"],
                "candidate_stage": row["candidate_stage"],
                "required_prior_concepts": row["required_prior_concepts"],
                "text_sha256": row["source_sha256"],
            }
        )
    review_rows.sort(key=lambda row: row["review_identity_sha256"])
    key_rows.sort(key=lambda row: row["review_identity_sha256"])
    if (
        len(review_rows) != 127
        or len({row["review_identity_sha256"] for row in review_rows}) != 127
    ):
        raise AuthoredReviewPacketError("review population geometry differs")
    review_encoded = _jsonl_bytes(review_rows)
    key_encoded = _jsonl_bytes(key_rows)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "awaiting_independent_review",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "candidate": {
            "path": str(candidate.resolve()),
            "bytes": len(candidate_bytes),
            "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "receipt_path": str(candidate_receipt.resolve()),
            "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "receipt_sha256": candidate_payload["receipt_sha256"],
        },
        "selection": {
            "population": "all_authored_candidate_rows",
            "rows": len(review_rows),
            "ordering": "ascending_sha256_of_salt_plus_candidate_identity",
            "review_hides_source_path_order_stage_and_declared_prerequisites": True,
            "review_key_is_separate": True,
        },
        "review_output": {
            "path": str(review_output.resolve()),
            "bytes": len(review_encoded),
            "rows": len(review_rows),
            "sha256": hashlib.sha256(review_encoded).hexdigest(),
        },
        "key_output": {
            "path": str(key_output.resolve()),
            "bytes": len(key_encoded),
            "rows": len(key_rows),
            "sha256": hashlib.sha256(key_encoded).hexdigest(),
        },
        "review_contract": {
            "annotator_and_reviewer_must_be_independently_identified": True,
            "evidence_spans_must_match_exact_source_text": True,
            "concept_set_disagreement_maximum_ppm": 50_000,
            "adjudication_cannot_rewrite_observed_disagreement": True,
            "candidate_stage_cannot_be_revealed_before_labels_are_frozen": True,
        },
        "limitations": [
            "packet_contains_no_completed_labels",
            "packet_does_not_qualify_the_candidate_curriculum",
            "packet_authorizes_no_training_or_architecture_promotion",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
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
    return payload


def validate_packet(
    *,
    candidate: Path,
    candidate_receipt: Path,
    review_output: Path,
    key_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Replay the blind population and its separate hidden key."""

    try:
        candidate_payload = validate(candidate, candidate_receipt)
        candidate_encoded = _read_regular_bytes(candidate, maximum_bytes=1 << 30)
        candidate_receipt_encoded = _read_regular_bytes(
            candidate_receipt, maximum_bytes=1 << 20
        )
        review_encoded = _read_regular_bytes(review_output, maximum_bytes=1 << 30)
        key_encoded = _read_regular_bytes(key_output, maximum_bytes=1 << 20)
        receipt_encoded = _read_regular_bytes(receipt_output, maximum_bytes=1 << 20)
        payload = json.loads(receipt_encoded)
        candidates = [
            json.loads(line) for line in candidate_encoded.decode().splitlines()
        ]
        reviews = [json.loads(line) for line in review_encoded.decode().splitlines()]
        keys = [json.loads(line) for line in key_encoded.decode().splitlines()]
    except (AuthoredCurriculumError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewPacketError("review packet is unreadable") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "awaiting_independent_review"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("candidate", {}).get("sha256")
        != hashlib.sha256(candidate_encoded).hexdigest()
        or payload.get("candidate", {}).get("receipt_file_sha256")
        != hashlib.sha256(candidate_receipt_encoded).hexdigest()
        or payload.get("candidate", {}).get("receipt_sha256")
        != candidate_payload["receipt_sha256"]
    ):
        raise AuthoredReviewPacketError("review receipt differs")
    for descriptor, encoded, rows in (
        (payload.get("review_output", {}), review_encoded, reviews),
        (payload.get("key_output", {}), key_encoded, keys),
    ):
        if (
            descriptor.get("bytes") != len(encoded)
            or descriptor.get("sha256") != hashlib.sha256(encoded).hexdigest()
            or descriptor.get("rows") != len(rows)
        ):
            raise AuthoredReviewPacketError("review output differs")
    candidates_by_identity = {row["identity_sha256"]: row for row in candidates}
    if (
        len(candidates_by_identity)
        != len(candidates)
        == len(reviews)
        == len(keys)
        == 127
        or [row.get("review_identity_sha256") for row in reviews]
        != sorted(row.get("review_identity_sha256") for row in reviews)
        or [row.get("review_identity_sha256") for row in reviews]
        != [row.get("review_identity_sha256") for row in keys]
    ):
        raise AuthoredReviewPacketError("review population differs")
    expected_review_fields = {
        "instructional_quality": "unlabeled",
        "assumed_prior_concepts": [],
        "taught_concepts_with_evidence_spans": [],
        "extraction_or_factual_defects": [],
        "admission_recommendation": "unlabeled",
    }
    for review, key in zip(reviews, keys, strict=True):
        candidate_row = candidates_by_identity.get(key.get("candidate_identity_sha256"))
        expected_review_identity = (
            hashlib.sha256(
                SELECTION_SALT + bytes.fromhex(candidate_row["identity_sha256"])
            ).hexdigest()
            if candidate_row is not None
            else None
        )
        if (
            set(review) != _REVIEW_KEYS
            or set(key) != _KEY_KEYS
            or review["schema"] != REVIEW_SCHEMA
            or key["schema"] != KEY_SCHEMA
            or candidate_row is None
            or review["review_identity_sha256"] != expected_review_identity
            or key["review_identity_sha256"] != expected_review_identity
            or review["title"] != candidate_row["title"]
            or review["text"] != candidate_row["text"]
            or review["text_sha256"] != candidate_row["source_sha256"]
            or review["requested_review"] != expected_review_fields
        ):
            raise AuthoredReviewPacketError("blind review row differs")
        expected_key = {
            "schema": KEY_SCHEMA,
            "review_identity_sha256": expected_review_identity,
            "candidate_identity_sha256": candidate_row["identity_sha256"],
            "source_name": candidate_row["source_name"],
            "source_revision": candidate_row["source_revision"],
            "source_path": candidate_row["source_path"],
            "source_order_index": candidate_row["source_order_index"],
            "candidate_stage": candidate_row["candidate_stage"],
            "required_prior_concepts": candidate_row["required_prior_concepts"],
            "text_sha256": candidate_row["source_sha256"],
        }
        if key != expected_key:
            raise AuthoredReviewPacketError("review key row differs")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    function = build if args.command == "build" else validate_packet
    payload = function(
        candidate=args.candidate,
        candidate_receipt=args.candidate_receipt,
        review_output=args.review_output,
        key_output=args.key_output,
        receipt_output=args.receipt_output,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
