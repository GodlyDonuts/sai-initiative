"""Select Stack-Edu safety candidates without claiming source admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.stack_edu_content import _sealed_regular
from sai.data.stack_edu_safety import (
    FINDING_ROW_SCHEMA,
    StackEduSafetyError,
    validate_safety_receipt,
)
from sai.data.stack_v2_alignment import _publish_pair
from sai.data.token_stream import canonical_sha256, sha256_file

REVIEW_ROW_SCHEMA = "sai-stack-edu-content-safety-review-v1"
SELECTED_ROW_SCHEMA = "sai-stack-edu-safety-selected-candidate-v1"
RECEIPT_SCHEMA = "sai-stack-edu-safety-selection-v1"
REVIEW_DISPOSITIONS = {"exclude_candidate", "retain_candidate"}
REVIEW_RATIONALES = {
    "confirmed_public_non_sensitive_content",
    "documented_example_or_placeholder_only",
    "generated_but_useful_and_source_valid",
    "personal_or_sensitive_information",
    "secret_or_credential_ambiguity",
    "generated_or_low_value_content",
    "obsolete_or_incompatible_source",
    "other_documented_exclusion",
}
_HEX = re.compile(r"[0-9a-f]{64}")
_FINDING_KEYS = {
    "schema",
    "ordinal",
    "repo_name",
    "path",
    "blob_id",
    "content_sha256",
    "decision",
    "reject_reasons",
    "review_reasons",
    "signals",
    "policy_sha256",
}
_REVIEW_KEYS = {
    "schema",
    "ordinal",
    "content_sha256",
    "disposition",
    "rationale_codes",
    "reviewer_identity_sha256",
    "reviewed_at_utc",
    "review_sha256",
}
_SELECTED_KEYS = {
    "schema",
    "ordinal",
    "repo_name",
    "path",
    "blob_id",
    "content_sha256",
    "safety_decision",
    "review_disposition",
    "policy_sha256",
}
_RECEIPT_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "safety_scan",
    "reviews",
    "policy",
    "summary",
    "selected_candidates",
    "limitations",
    "receipt_sha256",
}
_LIMITATIONS = [
    "selection_resolves_only_bounded_scanner_findings",
    "retained_candidate_is_not_source_admission_or_quality_proof",
    "reviewer_adjudication_does_not_replace_independent_quality_sampling",
    "global_deduplication_benchmark_decontamination_and_semantic_placement_pending",
    "matched_source_addition_evidence_pending",
    "selection_authorizes_no_training_or_four_b_scale",
]


class StackEduSafetySelectionError(RuntimeError):
    """The scan, review population, selection, or replay differs."""


def _read_jsonl(
    path: Path, label: str, *, maximum_bytes: int = 1 << 30
) -> tuple[list[Any], bytes, os.stat_result]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise StackEduSafetySelectionError(f"{label} differs") from error
    try:
        metadata = os.fstat(descriptor)
        encoded = os.read(descriptor, maximum_bytes + 1)
        if os.read(descriptor, 1):
            raise StackEduSafetySelectionError(f"{label} differs")
    except OSError as error:
        raise StackEduSafetySelectionError(f"{label} differs") from error
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o222
        or metadata.st_size > maximum_bytes
        or len(encoded) != metadata.st_size
    ):
        raise StackEduSafetySelectionError(f"{label} differs")
    try:
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduSafetySelectionError(f"{label} differs") from error
    return rows, encoded, metadata


def _findings(scan: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(scan["findings"]["path"])
    rows, _, _ = _read_jsonl(path, "Stack-Edu safety findings")
    if len(rows) != scan["findings"]["rows"]:
        raise StackEduSafetySelectionError("Stack-Edu safety findings differ")
    for ordinal, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != _FINDING_KEYS
            or row.get("schema") != FINDING_ROW_SCHEMA
            or row.get("ordinal") != ordinal
            or row.get("policy_sha256") != scan["policy_sha256"]
            or row.get("decision")
            not in {
                "candidate_clean_by_bounded_scanner",
                "manual_review_required",
                "rejected_high_confidence_sensitive_or_invalid",
            }
            or not isinstance(row.get("repo_name"), str)
            or not row["repo_name"]
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or not isinstance(row.get("blob_id"), str)
            or not row["blob_id"]
            or not isinstance(row.get("content_sha256"), str)
            or _HEX.fullmatch(row["content_sha256"]) is None
        ):
            raise StackEduSafetySelectionError("Stack-Edu safety finding differs")
    return rows


def _reviews(path: Path, findings: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows, _, _ = _read_jsonl(path, "Stack-Edu safety reviews", maximum_bytes=64 << 20)
    manual = {
        row["ordinal"]: row
        for row in findings
        if row["decision"] == "manual_review_required"
    }
    validated: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _REVIEW_KEYS:
            raise StackEduSafetySelectionError("Stack-Edu safety review differs")
        unsigned = {key: value for key, value in row.items() if key != "review_sha256"}
        ordinal = row.get("ordinal")
        rationale = row.get("rationale_codes")
        if (
            row.get("schema") != REVIEW_ROW_SCHEMA
            or isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal not in manual
            or ordinal in validated
            or row.get("content_sha256") != manual[ordinal]["content_sha256"]
            or row.get("disposition") not in REVIEW_DISPOSITIONS
            or not isinstance(rationale, list)
            or not rationale
            or rationale != sorted(rationale)
            or len(rationale) != len(set(rationale))
            or any(value not in REVIEW_RATIONALES for value in rationale)
            or _HEX.fullmatch(str(row.get("reviewer_identity_sha256"))) is None
            or not isinstance(row.get("reviewed_at_utc"), str)
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
                row["reviewed_at_utc"],
            )
            is None
            or row.get("review_sha256") != canonical_sha256(unsigned)
        ):
            raise StackEduSafetySelectionError("Stack-Edu safety review differs")
        validated[ordinal] = row
    if set(validated) != set(manual):
        raise StackEduSafetySelectionError("Stack-Edu safety review population differs")
    return validated


def _selection_payload(
    safety_receipt_path: Path, reviews_path: Path, selected_output: Path
) -> tuple[dict[str, Any], bytes]:
    safety_file_sha256 = sha256_file(safety_receipt_path)
    try:
        scan = validate_safety_receipt(safety_receipt_path)
    except StackEduSafetyError as error:
        raise StackEduSafetySelectionError("Stack-Edu safety scan differs") from error
    findings = _findings(scan)
    review_file_sha256 = sha256_file(reviews_path)
    reviews = _reviews(reviews_path, findings)
    selected = []
    decisions: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    for finding in findings:
        decision = finding["decision"]
        decisions[decision] += 1
        review = reviews.get(finding["ordinal"])
        retain = decision == "candidate_clean_by_bounded_scanner" or (
            review is not None and review["disposition"] == "retain_candidate"
        )
        if not retain:
            excluded[
                (
                    "high_confidence_reject"
                    if decision == "rejected_high_confidence_sensitive_or_invalid"
                    else "review_exclusion"
                )
            ] += 1
            continue
        row = {
            "schema": SELECTED_ROW_SCHEMA,
            "ordinal": finding["ordinal"],
            "repo_name": finding["repo_name"],
            "path": finding["path"],
            "blob_id": finding["blob_id"],
            "content_sha256": finding["content_sha256"],
            "safety_decision": decision,
            "review_disposition": None if review is None else review["disposition"],
            "policy_sha256": scan["policy_sha256"],
        }
        if set(row) != _SELECTED_KEYS:
            raise StackEduSafetySelectionError("Stack-Edu selected row differs")
        selected.append(row)
    encoded = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in selected
    )
    policy = {
        "high_confidence_rejects_are_never_overridable": True,
        "bounded_clean_candidates_are_retained": True,
        "every_manual_review_row_requires_one_hashed_adjudication": True,
        "manual_review_disposition_controls_candidate_retention": True,
        "selection_preserves_original_candidate_order": True,
        "selected_rows_remain_candidates_not_training_sources": True,
    }
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "bounded_safety_selection_complete_candidate_only",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "safety_scan": {
            "path": str(safety_receipt_path.resolve()),
            "file_sha256": safety_file_sha256,
            "receipt_sha256": scan["receipt_sha256"],
            "rows": len(findings),
        },
        "reviews": {
            "path": str(reviews_path.resolve()),
            "file_sha256": review_file_sha256,
            "rows": len(reviews),
            "ordered_review_sha256": canonical_sha256(
                [reviews[index] for index in sorted(reviews)]
            ),
        },
        "policy": policy,
        "summary": {
            "input_rows": len(findings),
            "selected_rows": len(selected),
            "excluded_rows": len(findings) - len(selected),
            "safety_decision_counts": dict(sorted(decisions.items())),
            "exclusion_counts": dict(sorted(excluded.items())),
            "complete_population_decided": True,
        },
        "selected_candidates": {
            "path": str(selected_output.resolve()),
            "rows": len(selected),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_sha256": canonical_sha256(selected),
        },
        "limitations": _LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    if sha256_file(safety_receipt_path) != safety_file_sha256:
        raise StackEduSafetySelectionError(
            "Stack-Edu safety scan changed while selecting"
        )
    if sha256_file(reviews_path) != review_file_sha256:
        raise StackEduSafetySelectionError(
            "Stack-Edu safety reviews changed while selecting"
        )
    return payload, encoded


def select_candidates(
    safety_receipt: Path,
    reviews: Path,
    selected_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Freeze a candidate-only selection after complete bounded adjudication."""

    payload, encoded = _selection_payload(
        Path(safety_receipt), Path(reviews), Path(selected_output)
    )
    receipt_encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    try:
        _publish_pair(
            Path(selected_output), encoded, Path(receipt_output), receipt_encoded
        )
    except Exception as error:
        raise StackEduSafetySelectionError(
            "Stack-Edu safety selection output differs"
        ) from error
    os.chmod(selected_output, 0o444)
    os.chmod(receipt_output, 0o444)
    return validate_selection_receipt(Path(receipt_output))


def validate_selection_receipt(receipt: Path) -> dict[str, Any]:
    """Reopen and replay a complete bounded safety selection."""

    _sealed_regular(Path(receipt), "Stack-Edu safety selection receipt")
    try:
        payload = json.loads(Path(receipt).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduSafetySelectionError(
            "Stack-Edu safety selection receipt differs"
        ) from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or set(payload) != _RECEIPT_KEYS
        or payload.get("schema") != RECEIPT_SCHEMA
        or payload.get("status") != "bounded_safety_selection_complete_candidate_only"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("limitations") != _LIMITATIONS
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise StackEduSafetySelectionError("Stack-Edu safety selection receipt differs")
    scan = payload.get("safety_scan")
    reviews = payload.get("reviews")
    selected = payload.get("selected_candidates")
    if (
        not isinstance(scan, dict)
        or set(scan) != {"path", "file_sha256", "receipt_sha256", "rows"}
        or not isinstance(reviews, dict)
        or set(reviews) != {"path", "file_sha256", "rows", "ordered_review_sha256"}
        or not isinstance(selected, dict)
        or set(selected) != {"path", "rows", "bytes", "sha256", "ordered_sha256"}
    ):
        raise StackEduSafetySelectionError("Stack-Edu safety selection receipt differs")
    selected_path = Path(selected["path"])
    _, selected_encoded, metadata = _read_jsonl(
        selected_path, "Stack-Edu safety selected candidates"
    )
    expected, encoded = _selection_payload(
        Path(scan["path"]), Path(reviews["path"]), selected_path
    )
    if (
        expected != payload
        or metadata.st_size != len(encoded)
        or selected_encoded != encoded
    ):
        raise StackEduSafetySelectionError("Stack-Edu safety selection replay differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--safety-receipt", type=Path, required=True)
    select.add_argument("--reviews", type=Path, required=True)
    select.add_argument("--selected-output", type=Path, required=True)
    select.add_argument("--receipt-output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        payload = select_candidates(
            args.safety_receipt,
            args.reviews,
            args.selected_output,
            args.receipt_output,
        )
    else:
        payload = validate_selection_receipt(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
