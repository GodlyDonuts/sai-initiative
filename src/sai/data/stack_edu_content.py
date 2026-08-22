"""Verify exact Stack-Edu source bytes after authorized content acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from sai.data.stack_v2_alignment import (
    StackV2AlignmentError,
    _validate_access_evidence,
    validate_alignment,
    validate_current_snapshot,
)
from sai.data.token_stream import canonical_sha256, sha256_file

INDEX_ROW_SCHEMA = "sai-stack-edu-content-index-row-v1"
RECEIPT_SCHEMA = "sai-stack-edu-content-verification-v1"
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_INDEX_KEYS = {
    "schema",
    "ordinal",
    "repo_name",
    "path",
    "blob_id",
    "offset",
    "length_bytes",
    "sha256",
    "s3_bucket",
    "s3_key",
    "s3_etag",
}
_RECEIPT_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "alignment",
    "bulk_access_evidence",
    "bundle",
    "index",
    "verification",
    "limitations",
    "receipt_sha256",
}
_LIMITATIONS = [
    "content_identity_and_encoding_verified_quality_not_admitted",
    "license_attribution_secret_pii_and_malware_review_pending",
    "exact_and_near_content_deduplication_across_full_mixture_pending",
    "benchmark_decontamination_pending",
    "semantic_prerequisite_annotation_and_curriculum_placement_pending",
    "content_verification_authorizes_no_training_or_source_retention",
]


class StackEduContentError(RuntimeError):
    """The aligned content bundle, byte identity, or acquisition evidence differs."""


def _sealed_regular(path: Path, label: str) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise StackEduContentError(f"{label} is missing or unsafe") from error
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_size <= 0
        or stat.S_IMODE(value.st_mode) & 0o222
    ):
        raise StackEduContentError(f"{label} is missing, unsafe, or writable")
    return value


def _json(path: Path, label: str, maximum: int = 64 << 20) -> dict[str, Any]:
    metadata = _sealed_regular(path, label)
    if metadata.st_size > maximum:
        raise StackEduContentError(f"{label} differs")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduContentError(f"{label} differs") from error
    if not isinstance(payload, dict):
        raise StackEduContentError(f"{label} differs")
    return payload


def _read_alignment_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(payload["aligned_candidates"]["path"])
    _sealed_regular(path, "aligned candidate population")
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or set(row) != {"schema", "candidate", "current"}
                    or row.get("schema") != "sai-stack-edu-current-alignment-row-v1"
                    or not isinstance(row.get("candidate"), dict)
                    or not isinstance(row.get("current"), dict)
                ):
                    raise StackEduContentError("aligned candidate row differs")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduContentError("aligned candidate population differs") from error
    if len(rows) != payload["aligned_candidates"]["rows"]:
        raise StackEduContentError("aligned candidate population differs")
    return rows


def _bulk_access(alignment: dict[str, Any]) -> dict[str, Any]:
    snapshot_path = Path(alignment["current_snapshot"]["path"])
    _sealed_regular(snapshot_path, "current Stack v2 snapshot receipt")
    try:
        snapshot = validate_current_snapshot(snapshot_path)
    except StackV2AlignmentError as error:
        raise StackEduContentError("current Stack v2 snapshot differs") from error
    access_path = Path(snapshot["access_evidence"]["path"])
    _sealed_regular(access_path, "Stack v2 bulk access evidence")
    try:
        _validate_access_evidence(access_path, snapshot["dataset_card"]["sha256"])
    except StackV2AlignmentError as error:
        raise StackEduContentError("Stack v2 bulk access evidence differs") from error
    try:
        access = json.loads(access_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduContentError("Stack v2 bulk access evidence differs") from error
    if access.get("bulk_content_access_authorized") is not True:
        raise StackEduContentError("Stack v2 bulk content access is not authorized")
    return {
        "path": str(access_path.resolve()),
        "file_sha256": sha256_file(access_path),
        "receipt_sha256": access["receipt_sha256"],
        "accepted_by": access["accepted_by"],
        "accepted_at_utc": access["accepted_at_utc"],
        "dataset_card_sha256": access["dataset_card_sha256"],
    }


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _validate_index_row(
    row: Any,
    *,
    ordinal: int,
    offset: int,
    aligned: dict[str, Any],
    bundle: BinaryIO,
) -> tuple[int, str]:
    candidate = aligned["candidate"]
    if not isinstance(row, dict) or set(row) != _INDEX_KEYS:
        raise StackEduContentError("content index row differs")
    if (
        row.get("schema") != INDEX_ROW_SCHEMA
        or row.get("ordinal") != ordinal
        or row.get("repo_name") != candidate["repo_name"]
        or row.get("path") != candidate["path"]
        or row.get("blob_id") != candidate["blob_id"]
        or row.get("offset") != offset
        or row.get("length_bytes") != candidate["length_bytes"]
        or not isinstance(row.get("sha256"), str)
        or _SHA64.fullmatch(row["sha256"]) is None
        or row.get("s3_bucket") != "softwareheritage"
        or row.get("s3_key") != f"content/{candidate['blob_id']}"
        or not isinstance(row.get("s3_etag"), str)
        or not row["s3_etag"]
        or _SHA40.fullmatch(candidate["blob_id"]) is None
    ):
        raise StackEduContentError("content index row differs")
    content = bundle.read(row["length_bytes"])
    if (
        len(content) != row["length_bytes"]
        or hashlib.sha256(content).hexdigest() != row["sha256"]
        or _git_blob_sha1(content) != candidate["blob_id"]
    ):
        raise StackEduContentError("Stack-Edu content identity differs")
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StackEduContentError("Stack-Edu content is not UTF-8") from error
    if decoded.encode("utf-8") != content:
        raise StackEduContentError("Stack-Edu UTF-8 round trip differs")
    return offset + len(content), hashlib.sha256(content).hexdigest()


def _verification_payload(
    alignment_receipt: Path,
    bundle_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    _sealed_regular(alignment_receipt, "Stack-Edu current alignment receipt")
    alignment_file_sha256 = sha256_file(alignment_receipt)
    try:
        alignment = validate_alignment(alignment_receipt)
    except StackV2AlignmentError as error:
        raise StackEduContentError("Stack-Edu current alignment differs") from error
    if sha256_file(alignment_receipt) != alignment_file_sha256:
        raise StackEduContentError("Stack-Edu alignment changed while reading")
    access = _bulk_access(alignment)
    bundle_metadata = _sealed_regular(bundle_path, "Stack-Edu content bundle")
    index_metadata = _sealed_regular(index_path, "Stack-Edu content index")
    bundle_sha256 = sha256_file(bundle_path)
    index_sha256 = sha256_file(index_path)
    aligned_rows = _read_alignment_rows(alignment)
    content_hashes = []
    offset = 0
    rows = 0
    try:
        with (
            bundle_path.open("rb") as bundle,
            index_path.open("r", encoding="utf-8") as index,
        ):
            for line in index:
                if rows >= len(aligned_rows):
                    raise StackEduContentError("content index has extra rows")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StackEduContentError("content index row differs") from error
                offset, content_sha256 = _validate_index_row(
                    row,
                    ordinal=rows,
                    offset=offset,
                    aligned=aligned_rows[rows],
                    bundle=bundle,
                )
                content_hashes.append(content_sha256)
                rows += 1
            if bundle.read(1):
                raise StackEduContentError("content bundle has trailing bytes")
    except (OSError, UnicodeDecodeError) as error:
        raise StackEduContentError("content bundle or index differs") from error
    if rows != len(aligned_rows) or offset != bundle_metadata.st_size:
        raise StackEduContentError("content bundle population is incomplete")
    if (
        sha256_file(bundle_path) != bundle_sha256
        or sha256_file(index_path) != index_sha256
        or bundle_path.stat(follow_symlinks=False).st_size != bundle_metadata.st_size
        or index_path.stat(follow_symlinks=False).st_size != index_metadata.st_size
    ):
        raise StackEduContentError("content bundle changed while reading")
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "content_bytes_verified_quality_and_curriculum_pending",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "alignment": {
            "path": str(alignment_receipt.resolve()),
            "file_sha256": alignment_file_sha256,
            "receipt_sha256": alignment["receipt_sha256"],
            "rows": alignment["aligned_candidates"]["rows"],
        },
        "bulk_access_evidence": access,
        "bundle": {
            "path": str(bundle_path.resolve()),
            "bytes": bundle_metadata.st_size,
            "sha256": bundle_sha256,
        },
        "index": {
            "path": str(index_path.resolve()),
            "rows": rows,
            "bytes": index_metadata.st_size,
            "sha256": index_sha256,
        },
        "verification": {
            "ordered_content_sha256": canonical_sha256(content_hashes),
            "git_blob_sha1_verified_rows": rows,
            "independent_sha256_verified_rows": rows,
            "candidate_length_verified_rows": rows,
            "utf8_round_trip_verified_rows": rows,
            "gap_overlap_or_trailing_bytes": 0,
            "complete": True,
        },
        "limitations": _LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def verify_content_bundle(
    alignment_receipt: Path,
    bundle_path: Path,
    index_path: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Verify and freeze exact bytes for every aligned candidate."""

    payload = _verification_payload(
        Path(alignment_receipt), Path(bundle_path), Path(index_path)
    )
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    _publish_receipt(Path(receipt_output), encoded)
    return validate_content_receipt(Path(receipt_output))


def _publish_receipt(output: Path, encoded: bytes) -> None:
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise StackEduContentError("content receipt output boundary differs")
    stage = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    linked = False
    try:
        descriptor = os.open(
            stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, output)
        linked = True
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        if linked:
            output.unlink(missing_ok=True)
        raise StackEduContentError("content receipt output boundary differs") from error
    except BaseException:
        if linked:
            output.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)


def validate_content_receipt(receipt: Path) -> dict[str, Any]:
    """Reopen and replay every aligned source byte."""

    payload = _json(Path(receipt), "Stack-Edu content receipt")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload) != _RECEIPT_KEYS
        or payload.get("schema") != RECEIPT_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("status")
        != "content_bytes_verified_quality_and_curriculum_pending"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("limitations") != _LIMITATIONS
    ):
        raise StackEduContentError("Stack-Edu content receipt differs")
    alignment = payload.get("alignment")
    bundle = payload.get("bundle")
    index = payload.get("index")
    if (
        not isinstance(alignment, dict)
        or set(alignment) != {"path", "file_sha256", "receipt_sha256", "rows"}
        or not isinstance(bundle, dict)
        or set(bundle) != {"path", "bytes", "sha256"}
        or not isinstance(index, dict)
        or set(index) != {"path", "rows", "bytes", "sha256"}
    ):
        raise StackEduContentError("Stack-Edu content receipt differs")
    alignment_path = Path(alignment["path"])
    bundle_path = Path(bundle["path"])
    index_path = Path(index["path"])
    if (
        sha256_file(alignment_path) != alignment["file_sha256"]
        or sha256_file(bundle_path) != bundle["sha256"]
        or sha256_file(index_path) != index["sha256"]
    ):
        raise StackEduContentError("Stack-Edu content receipt parent differs")
    expected = _verification_payload(alignment_path, bundle_path, index_path)
    if expected != payload:
        raise StackEduContentError("Stack-Edu content receipt replay differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--alignment-receipt", type=Path, required=True)
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--receipt-output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        payload = verify_content_bundle(
            args.alignment_receipt,
            args.bundle,
            args.index,
            args.receipt_output,
        )
    else:
        payload = validate_content_receipt(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
