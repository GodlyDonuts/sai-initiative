"""Validate Sai's prospective semantic-annotation policy."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-semantic-annotation-policy-v1"
_MAX_BYTES = 1 << 20
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "concept_list_sha256",
    "annotation_unit",
    "positive_label_rule",
    "negative_label_rule",
    "evidence_span_contract",
    "confidence_contract",
    "prerequisite_contract",
    "review_contract",
    "receipt_sha256",
}
_SPAN_KEYS = {
    "coordinate_system",
    "minimum_spans_per_positive_label",
    "source_hash_required",
    "exact_text_match_required",
}
_CONFIDENCE_KEYS = {
    "minimum_confidence_ppm",
    "confidence_is_probability_of_policy_compliance",
    "below_threshold_action",
}
_PREREQUISITE_KEYS = {
    "same_document_exposure_counts_as_prior",
    "phase_source",
    "unmet_prerequisite_action",
}
_REVIEW_KEYS = {
    "blind_independent_review",
    "disagreement_unit",
    "minimum_reviewed_documents",
    "maximum_disagreement_ppm",
    "adjudication_may_not_change_measured_disagreement",
}


class AnnotationPolicyError(RuntimeError):
    """The semantic annotation policy is missing, unsafe, or underspecified."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AnnotationPolicyError(f"{label} fields differ")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AnnotationPolicyError(f"{label} differs")
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise AnnotationPolicyError(f"{label} differs") from error
    if not any(raw):
        raise AnnotationPolicyError(f"{label} is a placeholder")
    return value


def _read_regular(path: Path) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise AnnotationPolicyError("annotation policy is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_BYTES
        ):
            raise AnnotationPolicyError("annotation policy is missing or unsafe")
        chunks = []
        remaining = before.st_size
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
        raise AnnotationPolicyError("annotation policy changed while reading")
    return encoded


def validate_policy_payload(
    payload: Any, *, expected_concept_list_sha256: str
) -> dict[str, Any]:
    """Validate the exact prospective rules used to assign semantic labels."""

    policy = _exact(payload, _TOP_KEYS, "annotation policy")
    unsigned = {key: value for key, value in policy.items() if key != "receipt_sha256"}
    if (
        policy["schema"] != SCHEMA
        or policy["status"] != "prospective"
        or policy["training_authorized"] is not False
        or policy["four_b_training_authorized"] is not False
        or _sha256(policy["receipt_sha256"], "annotation policy receipt")
        != canonical_sha256(unsigned)
        or _sha256(policy["concept_list_sha256"], "concept list")
        != expected_concept_list_sha256
        or policy["annotation_unit"] != "document_concept_presence"
        or policy["positive_label_rule"]
        != "explicit_instruction_or_demonstrated_use_with_verifiable_source_span"
        or policy["negative_label_rule"]
        != "omit_when_direct_source_evidence_is_absent_or_ambiguous"
    ):
        raise AnnotationPolicyError("annotation policy boundary differs")
    span = _exact(policy["evidence_span_contract"], _SPAN_KEYS, "evidence span")
    if span != {
        "coordinate_system": "unicode_codepoint_half_open",
        "minimum_spans_per_positive_label": 1,
        "source_hash_required": True,
        "exact_text_match_required": True,
    }:
        raise AnnotationPolicyError("evidence span contract differs")
    confidence = _exact(policy["confidence_contract"], _CONFIDENCE_KEYS, "confidence")
    if confidence != {
        "minimum_confidence_ppm": 800_000,
        "confidence_is_probability_of_policy_compliance": True,
        "below_threshold_action": "omit_and_flag_for_review",
    }:
        raise AnnotationPolicyError("confidence contract differs")
    prerequisite = _exact(
        policy["prerequisite_contract"], _PREREQUISITE_KEYS, "prerequisite"
    )
    if prerequisite != {
        "same_document_exposure_counts_as_prior": False,
        "phase_source": "bound_curriculum_receipt_only",
        "unmet_prerequisite_action": "record_violation_and_reject_progression",
    }:
        raise AnnotationPolicyError("prerequisite contract differs")
    review = _exact(policy["review_contract"], _REVIEW_KEYS, "review")
    if review != {
        "blind_independent_review": True,
        "disagreement_unit": "unordered_unique_concept_identity_set_per_document",
        "minimum_reviewed_documents": 100,
        "maximum_disagreement_ppm": 50_000,
        "adjudication_may_not_change_measured_disagreement": True,
    }:
        raise AnnotationPolicyError("review contract differs")
    return policy


def validate_policy(path: Path, *, expected_concept_list_sha256: str) -> dict[str, Any]:
    encoded = _read_regular(path)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnnotationPolicyError("annotation policy JSON differs") from error
    return validate_policy_payload(
        payload, expected_concept_list_sha256=expected_concept_list_sha256
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--concept-list", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_policy(
        args.policy, expected_concept_list_sha256=sha256_file(args.concept_list)
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
