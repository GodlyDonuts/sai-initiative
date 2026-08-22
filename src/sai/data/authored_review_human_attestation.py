"""Bind one completed blind human review to an exact identity attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_adjudication import (
    IDENTITY_SCHEMA,
    _validate_reviews,
)
from sai.data.authored_review_model import _blind_inputs
from sai.data.token_stream import canonical_sha256


class AuthoredReviewHumanAttestationError(RuntimeError):
    """The human-review labels or bound identity attestation differs."""


def _payload(
    *,
    role: str,
    identity_attestation_sha256: str,
    review_packet: Path,
    review_packet_receipt: Path,
    expected_review_packet_sha256: str,
    expected_review_packet_receipt_sha256: str,
    concept_list: Path,
    annotation_policy: Path,
    completed_reviews: Path,
) -> dict[str, Any]:
    if (
        role not in {"annotator", "reviewer"}
        or len(identity_attestation_sha256) != 64
        or identity_attestation_sha256 == "0" * 64
        or any(
            character not in "0123456789abcdef"
            for character in identity_attestation_sha256
        )
    ):
        raise AuthoredReviewHumanAttestationError("human reviewer identity differs")
    try:
        inputs = _blind_inputs(
            review_packet=review_packet,
            review_packet_receipt=review_packet_receipt,
            expected_review_packet_sha256=expected_review_packet_sha256,
            expected_review_packet_receipt_sha256=(
                expected_review_packet_receipt_sha256
            ),
            concept_list=concept_list,
            annotation_policy=annotation_policy,
        )
        encoded = _read_regular_bytes(completed_reviews, maximum_bytes=1 << 30)
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
        concept_ids = {
            concept["concept_id"] for concept in inputs.concept_payload["concepts"]
        }
        _validate_reviews(rows, inputs.packet, concept_ids, role)
    except Exception as error:
        raise AuthoredReviewHumanAttestationError(
            "completed human review differs"
        ) from error
    payload: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "status": "complete",
        "role": role,
        "identity_attestation_sha256": identity_attestation_sha256,
        "blind_review_packet_sha256": hashlib.sha256(inputs.packet_encoded).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(inputs.policy_encoded).hexdigest(),
        "completed_reviews_sha256": hashlib.sha256(encoded).hexdigest(),
        "reviewed_documents": len(rows),
        "human_review_completed": True,
        "model_generated_labels": False,
        "hidden_review_key_accessed_before_label_freeze": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def run(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    payload = _payload(**kwargs)
    _write_create_only(
        output, json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    )
    return payload


def validate(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    try:
        actual = json.loads(_read_regular_bytes(output, maximum_bytes=1 << 20))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewHumanAttestationError(
            "human review attestation differs"
        ) from error
    expected = _payload(**kwargs)
    if actual != expected:
        raise AuthoredReviewHumanAttestationError("human review attestation differs")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("attest", "validate"))
    parser.add_argument("--role", choices=("annotator", "reviewer"), required=True)
    parser.add_argument("--identity-attestation-sha256", required=True)
    for name in (
        "review-packet",
        "review-packet-receipt",
        "concept-list",
        "annotation-policy",
        "completed-reviews",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-review-packet-sha256", required=True)
    parser.add_argument("--expected-review-packet-receipt-sha256", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (run if command == "attest" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "role": payload["role"],
                "reviewed_documents": payload["reviewed_documents"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
