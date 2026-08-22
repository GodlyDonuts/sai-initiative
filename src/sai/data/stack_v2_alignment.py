"""Align Stack-Edu candidates with the current opt-out-enacted Stack v2 release.

This module deliberately operates on metadata only.  It proves that an older
Stack-Edu candidate identity is still present in the exact current Stack v2
snapshot and remains permissively licensed, non-vendor, and non-generated.
It does not download source content and never authorizes training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.stack_edu_audit import ALLOWED_LICENSES
from sai.data.stack_edu_candidates import (
    StackEduCandidateError,
    _read_candidate_rows,
    _safe_regular,
    validate_candidate_aggregate,
)
from sai.data.token_stream import canonical_sha256, sha256_file

CURRENT_DATASET = "bigcode/the-stack-v2"
CURRENT_REVISION = "e565caa3a78c2423bd374333a472b049eb090e47"
CURRENT_RELEASE = "v2.2.0"
CURRENT_OPT_OUT_CUTOFF = "2026-07-29"

SNAPSHOT_SCHEMA = "sai-stack-v2-current-python-snapshot-v1"
ACCESS_SCHEMA = "sai-stack-v2-metadata-access-evidence-v1"
ALIGNMENT_ROW_SCHEMA = "sai-stack-edu-current-alignment-row-v1"
ALIGNMENT_SCHEMA = "sai-stack-edu-current-alignment-v1"
_MEMBER = re.compile(
    r"data/Python/train-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.parquet"
)
_CURRENT_COLUMNS = {
    "blob_id",
    "content_id",
    "detected_licenses",
    "is_generated",
    "is_vendor",
    "language",
    "license_type",
    "path",
    "repo_name",
    "revision_id",
    "src_encoding",
}
_SNAPSHOT_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "dataset",
    "revision",
    "release",
    "opt_out_cutoff",
    "language",
    "dataset_card",
    "access_evidence",
    "files",
    "summary",
    "limitations",
    "receipt_sha256",
}
_ALIGNMENT_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "candidate_aggregate",
    "current_snapshot",
    "summary",
    "aligned_candidates",
    "limitations",
    "receipt_sha256",
}
_ACCESS_KEYS = {
    "schema",
    "status",
    "dataset",
    "revision",
    "release",
    "accepted_by",
    "accepted_at_utc",
    "dataset_card_sha256",
    "metadata_access_authorized",
    "bulk_content_access_authorized",
    "training_authorized",
    "four_b_training_authorized",
    "receipt_sha256",
}
_SNAPSHOT_LIMITATIONS = [
    "metadata_snapshot_only_source_content_not_acquired",
    "snapshot_must_be_replaced_when_a_newer_usable_stack_v2_release_exists",
    "current_membership_does_not_replace_license_attribution_or_content_review",
    "snapshot_authorizes_no_training_or_source_retention",
]
_ALIGNMENT_LIMITATIONS = [
    "current_stack_v2_membership_is_required_but_not_content_admission",
    "source_content_sha1_sha256_and_length_verification_pending",
    "license_attribution_secret_pii_malware_and_quality_review_pending",
    "exact_and_near_content_deduplication_pending",
    "benchmark_decontamination_and_semantic_curriculum_review_pending",
    "alignment_authorizes_no_training_or_source_retention",
]


class StackV2AlignmentError(RuntimeError):
    """The current Stack v2 snapshot or candidate alignment differs."""


def _json(path: Path, label: str, maximum: int = 64 << 20) -> dict[str, Any]:
    metadata = _safe_regular(path, label)
    if metadata.st_size > maximum:
        raise StackV2AlignmentError(f"{label} differs")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackV2AlignmentError(f"{label} differs") from error
    if not isinstance(payload, dict):
        raise StackV2AlignmentError(f"{label} differs")
    return payload


def _artifact(path: Path, label: str) -> dict[str, Any]:
    metadata = _safe_regular(path, label)
    return {
        "path": str(path.resolve()),
        "bytes": metadata.st_size,
        "sha256": sha256_file(path),
    }


def _validate_dataset_card(path: Path) -> dict[str, Any]:
    descriptor = _artifact(path, "Stack v2 dataset card")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise StackV2AlignmentError("Stack v2 dataset card differs") from error
    required = (
        CURRENT_DATASET,
        CURRENT_RELEASE,
        CURRENT_OPT_OUT_CUTOFF,
        "SoftwareHeritage",
        "update your own version of The Stack v2",
    )
    if not all(value in text for value in required):
        raise StackV2AlignmentError("Stack v2 dataset card differs")
    return descriptor


def _validate_access_evidence(path: Path, card_sha256: str) -> dict[str, Any]:
    descriptor = _artifact(path, "Stack v2 access evidence")
    payload = _json(path, "Stack v2 access evidence")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    accepted_at = payload.get("accepted_at_utc")
    try:
        parsed = dt.datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise StackV2AlignmentError("Stack v2 access evidence differs") from error
    if (
        set(payload) != _ACCESS_KEYS
        or payload.get("schema") != ACCESS_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("status") != "metadata_access_terms_accepted"
        or payload.get("dataset") != CURRENT_DATASET
        or payload.get("revision") != CURRENT_REVISION
        or payload.get("release") != CURRENT_RELEASE
        or not isinstance(payload.get("accepted_by"), str)
        or len(payload["accepted_by"].strip()) < 3
        or parsed.tzinfo is None
        or payload.get("dataset_card_sha256") != card_sha256
        or payload.get("metadata_access_authorized") is not True
        or not isinstance(payload.get("bulk_content_access_authorized"), bool)
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
    ):
        raise StackV2AlignmentError("Stack v2 access evidence differs")
    return descriptor


def _validate_artifact(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise StackV2AlignmentError(f"{label} differs")
    path = Path(value.get("path", ""))
    metadata = _safe_regular(path, label)
    if (
        isinstance(value.get("bytes"), bool)
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] <= 0
        or metadata.st_size != value["bytes"]
        or sha256_file(path) != value.get("sha256")
    ):
        raise StackV2AlignmentError(f"{label} differs")
    return path


def _source_member(value: Any) -> tuple[int, int]:
    if not isinstance(value, str):
        raise StackV2AlignmentError("current Stack v2 source member differs")
    match = _MEMBER.fullmatch(value)
    if match is None:
        raise StackV2AlignmentError("current Stack v2 source member differs")
    index = int(match.group("index"))
    count = int(match.group("count"))
    if count <= 0 or index >= count:
        raise StackV2AlignmentError("current Stack v2 source member differs")
    return index, count


def _import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise StackV2AlignmentError(
            "PyArrow is required for Stack v2 alignment"
        ) from error
    return pq


def _publish_one(output: Path, encoded: bytes) -> None:
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise StackV2AlignmentError("Stack v2 output boundary differs")
    stage = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    linked = False
    try:
        descriptor = os.open(
            stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
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
        raise StackV2AlignmentError("Stack v2 output boundary differs") from error
    except BaseException:
        if linked:
            output.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)


def _publish_pair(
    first: Path, first_encoded: bytes, second: Path, second_encoded: bytes
) -> None:
    if first.parent != second.parent:
        raise StackV2AlignmentError("Stack v2 output boundary differs")
    token = uuid.uuid4().hex
    stages = [
        first.with_name(f".{first.name}.{token}.partial"),
        second.with_name(f".{second.name}.{token}.partial"),
    ]
    outputs = [first, second]
    if not first.parent.is_dir() or any(
        path.exists() or path.is_symlink() for path in outputs
    ):
        raise StackV2AlignmentError("Stack v2 output boundary differs")
    linked: list[Path] = []
    try:
        for stage, encoded in zip(stages, (first_encoded, second_encoded), strict=True):
            descriptor = os.open(
                stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        for stage, output in zip(stages, outputs, strict=True):
            os.link(stage, output)
            linked.append(output)
        directory = os.open(first.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        for output in reversed(linked):
            output.unlink(missing_ok=True)
        raise StackV2AlignmentError("Stack v2 output boundary differs") from error
    except BaseException:
        for output in reversed(linked):
            output.unlink(missing_ok=True)
        raise
    finally:
        for stage in stages:
            stage.unlink(missing_ok=True)


def freeze_current_snapshot(
    *,
    dataset_card: Path,
    access_evidence: Path,
    sources: list[tuple[str, Path]],
    receipt_output: Path,
) -> dict[str, Any]:
    """Freeze exact current-release Python metadata files after authorized download."""

    if not sources:
        raise StackV2AlignmentError("current Stack v2 source set is empty")
    card = _validate_dataset_card(Path(dataset_card))
    access = _validate_access_evidence(Path(access_evidence), card["sha256"])
    pq = _import_parquet()
    members = []
    inodes: set[tuple[int, int]] = set()
    counts: set[int] = set()
    indices: set[int] = set()
    for source_file, raw_path in sources:
        index, count = _source_member(source_file)
        path = Path(raw_path)
        metadata = _safe_regular(path, "current Stack v2 source")
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes or index in indices:
            raise StackV2AlignmentError("current Stack v2 source is duplicated")
        inodes.add(inode)
        indices.add(index)
        counts.add(count)
        parquet = pq.ParquetFile(path)
        if not _CURRENT_COLUMNS.issubset(parquet.schema_arrow.names):
            raise StackV2AlignmentError("current Stack v2 columns differ")
        members.append(
            {
                "index": index,
                "source_file": source_file,
                "path": str(path.resolve()),
                "bytes": metadata.st_size,
                "sha256": sha256_file(path),
                "rows": parquet.metadata.num_rows,
            }
        )
    if len(counts) != 1:
        raise StackV2AlignmentError("current Stack v2 shard geometry differs")
    count = counts.pop()
    if len(members) != count or indices != set(range(count)):
        raise StackV2AlignmentError("current Stack v2 shard set is incomplete")
    members.sort(key=lambda row: row["index"])
    payload: dict[str, Any] = {
        "schema": SNAPSHOT_SCHEMA,
        "status": "current_opt_out_enacted_python_metadata_snapshot_frozen",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "dataset": CURRENT_DATASET,
        "revision": CURRENT_REVISION,
        "release": CURRENT_RELEASE,
        "opt_out_cutoff": CURRENT_OPT_OUT_CUTOFF,
        "language": "Python",
        "dataset_card": card,
        "access_evidence": access,
        "files": members,
        "summary": {
            "files": len(members),
            "rows": sum(row["rows"] for row in members),
            "bytes": sum(row["bytes"] for row in members),
            "complete_shard_set": True,
        },
        "limitations": _SNAPSHOT_LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    _publish_one(Path(receipt_output), encoded)
    return validate_current_snapshot(Path(receipt_output))


def validate_current_snapshot(receipt: Path) -> dict[str, Any]:
    """Reopen the exact current-release Python metadata snapshot."""

    payload = _json(Path(receipt), "current Stack v2 snapshot receipt")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload) != _SNAPSHOT_KEYS
        or payload.get("schema") != SNAPSHOT_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("status")
        != "current_opt_out_enacted_python_metadata_snapshot_frozen"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("dataset") != CURRENT_DATASET
        or payload.get("revision") != CURRENT_REVISION
        or payload.get("release") != CURRENT_RELEASE
        or payload.get("opt_out_cutoff") != CURRENT_OPT_OUT_CUTOFF
        or payload.get("language") != "Python"
        or payload.get("limitations") != _SNAPSHOT_LIMITATIONS
    ):
        raise StackV2AlignmentError("current Stack v2 snapshot receipt differs")
    card_path = _validate_artifact(payload.get("dataset_card"), "Stack v2 dataset card")
    card = _validate_dataset_card(card_path)
    access_path = _validate_artifact(
        payload.get("access_evidence"), "Stack v2 access evidence"
    )
    _validate_access_evidence(access_path, card["sha256"])
    files = payload.get("files")
    summary = payload.get("summary")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(summary, dict)
        or set(summary) != {"files", "rows", "bytes", "complete_shard_set"}
    ):
        raise StackV2AlignmentError("current Stack v2 snapshot receipt differs")
    pq = _import_parquet()
    counts: set[int] = set()
    indices: set[int] = set()
    inodes: set[tuple[int, int]] = set()
    rows = 0
    size = 0
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "index",
            "source_file",
            "path",
            "bytes",
            "sha256",
            "rows",
        }:
            raise StackV2AlignmentError("current Stack v2 snapshot member differs")
        index, count = _source_member(row.get("source_file"))
        if row.get("index") != index or index in indices:
            raise StackV2AlignmentError("current Stack v2 snapshot member differs")
        path = Path(row.get("path", ""))
        metadata = _safe_regular(path, "current Stack v2 source")
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes:
            raise StackV2AlignmentError("current Stack v2 source is duplicated")
        parquet = pq.ParquetFile(path)
        if (
            not _CURRENT_COLUMNS.issubset(parquet.schema_arrow.names)
            or metadata.st_size != row.get("bytes")
            or sha256_file(path) != row.get("sha256")
            or parquet.metadata.num_rows != row.get("rows")
        ):
            raise StackV2AlignmentError("current Stack v2 snapshot member differs")
        counts.add(count)
        indices.add(index)
        inodes.add(inode)
        rows += parquet.metadata.num_rows
        size += metadata.st_size
    if (
        len(counts) != 1
        or len(files) != next(iter(counts))
        or indices != set(range(len(files)))
        or summary
        != {
            "files": len(files),
            "rows": rows,
            "bytes": size,
            "complete_shard_set": True,
        }
    ):
        raise StackV2AlignmentError("current Stack v2 shard set is incomplete")
    return payload


def _current_index(
    snapshot: dict[str, Any], wanted: set[tuple[str, str, str]]
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    pq = _import_parquet()
    present: set[tuple[str, str, str]] = set()
    eligible: dict[tuple[str, str, str], dict[str, Any]] = {}
    geometry: dict[tuple[str, str, str], tuple[str, str]] = {}
    occurrences: Counter[tuple[str, str, str]] = Counter()
    disqualified: dict[tuple[str, str, str], set[str]] = {}
    scanned = 0
    for member in sorted(snapshot["files"], key=lambda row: row["index"]):
        path = Path(member["path"])
        before = _safe_regular(path, "current Stack v2 source")
        columns = sorted(_CURRENT_COLUMNS)
        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=8_192, columns=columns):
            values = batch.to_pydict()
            for raw in zip(*(values[column] for column in columns), strict=True):
                row = dict(zip(columns, raw, strict=True))
                triple = (row["repo_name"], row["path"], row["blob_id"])
                if triple in wanted:
                    if (
                        not all(isinstance(value, str) and value for value in triple)
                        or row["language"] != "Python"
                        or not isinstance(row["src_encoding"], str)
                        or not row["src_encoding"]
                        or not isinstance(row["content_id"], str)
                        or not row["content_id"]
                        or not isinstance(row["revision_id"], str)
                        or not row["revision_id"]
                        or not isinstance(row["detected_licenses"], list)
                        or not all(
                            isinstance(value, str) and value
                            for value in row["detected_licenses"]
                        )
                        or len(row["detected_licenses"])
                        != len(set(row["detected_licenses"]))
                        or not isinstance(row["is_vendor"], bool)
                        or not isinstance(row["is_generated"], bool)
                    ):
                        raise StackV2AlignmentError(
                            "current Stack v2 matching row differs"
                        )
                    present.add(triple)
                    occurrences[triple] += 1
                    current_geometry = (row["content_id"], row["src_encoding"])
                    previous = geometry.setdefault(triple, current_geometry)
                    if previous != current_geometry:
                        raise StackV2AlignmentError(
                            "current Stack v2 identity geometry differs"
                        )
                    reasons = set()
                    if (
                        row["license_type"] != "permissive"
                        or not row["detected_licenses"]
                        or not set(row["detected_licenses"]).issubset(ALLOWED_LICENSES)
                    ):
                        reasons.add("current_nonpermissive_or_unattributed")
                    if row["is_vendor"]:
                        reasons.add("current_vendor")
                    if row["is_generated"]:
                        reasons.add("current_generated")
                    if reasons:
                        disqualified.setdefault(triple, set()).update(reasons)
                    elif triple not in eligible:
                        eligible[triple] = {
                            "source_file": member["source_file"],
                            "source_row_index": row_index,
                            "content_id": row["content_id"],
                            "revision_id": row["revision_id"],
                            "src_encoding": row["src_encoding"],
                            "detected_licenses": row["detected_licenses"],
                            "license_type": row["license_type"],
                            "is_vendor": False,
                            "is_generated": False,
                        }
                row_index += 1
        scanned += row_index
        after = path.stat(follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or sha256_file(path) != member["sha256"]:
            raise StackV2AlignmentError("current Stack v2 source changed while reading")
    return eligible, {
        "current_rows_scanned": scanned,
        "current_matching_occurrences": sum(occurrences.values()),
        "current_duplicate_matching_occurrences": sum(
            count - 1 for count in occurrences.values()
        ),
        "present": present,
        "disqualified": disqualified,
    }


def _alignment_payload(
    candidate_receipt: Path,
    snapshot_receipt: Path,
    aligned_output: Path,
) -> tuple[dict[str, Any], bytes]:
    candidate_file_sha256 = sha256_file(candidate_receipt)
    snapshot_file_sha256 = sha256_file(snapshot_receipt)
    try:
        candidate = validate_candidate_aggregate(candidate_receipt)
    except StackEduCandidateError as error:
        raise StackV2AlignmentError("Stack-Edu candidate aggregate differs") from error
    snapshot = validate_current_snapshot(snapshot_receipt)
    if (
        sha256_file(candidate_receipt) != candidate_file_sha256
        or sha256_file(snapshot_receipt) != snapshot_file_sha256
    ):
        raise StackV2AlignmentError("alignment parent changed while reading")
    candidates = _read_candidate_rows(candidate)
    wanted = {(row["repo_name"], row["path"], row["blob_id"]) for row in candidates}
    if len(wanted) != len(candidates):
        raise StackV2AlignmentError("Stack-Edu candidate identity is duplicated")
    current, scan = _current_index(snapshot, wanted)
    aligned = []
    removed = Counter()
    for candidate_row in candidates:
        triple = (
            candidate_row["repo_name"],
            candidate_row["path"],
            candidate_row["blob_id"],
        )
        evidence = current.get(triple)
        if evidence is None:
            if triple not in scan["present"]:
                removed["absent_from_current_opt_out_enacted_snapshot"] += 1
            else:
                reasons = scan["disqualified"].get(triple, set())
                if not reasons:
                    raise StackV2AlignmentError("current Stack v2 eligibility differs")
                for reason in sorted(reasons):
                    removed[reason] += 1
            continue
        aligned.append(
            {
                "schema": ALIGNMENT_ROW_SCHEMA,
                "candidate": candidate_row,
                "current": evidence,
            }
        )
    encoded = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in aligned
    )
    summary = {
        "input_candidate_rows": len(candidates),
        "aligned_rows": len(aligned),
        "removed_rows": len(candidates) - len(aligned),
        "removed_by_reason": dict(sorted(removed.items())),
        "current_rows_scanned": scan["current_rows_scanned"],
        "current_matching_occurrences": scan["current_matching_occurrences"],
        "current_duplicate_matching_occurrences": scan[
            "current_duplicate_matching_occurrences"
        ],
        "membership_key": "exact_repo_name_path_blob_id",
        "current_release_is_opt_out_authority": True,
    }
    payload: dict[str, Any] = {
        "schema": ALIGNMENT_SCHEMA,
        "status": "current_stack_v2_membership_aligned_content_not_acquired",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "candidate_aggregate": {
            "path": str(candidate_receipt.resolve()),
            "file_sha256": candidate_file_sha256,
            "receipt_sha256": candidate["receipt_sha256"],
        },
        "current_snapshot": {
            "path": str(snapshot_receipt.resolve()),
            "file_sha256": snapshot_file_sha256,
            "receipt_sha256": snapshot["receipt_sha256"],
            "dataset": snapshot["dataset"],
            "revision": snapshot["revision"],
            "release": snapshot["release"],
            "opt_out_cutoff": snapshot["opt_out_cutoff"],
        },
        "summary": summary,
        "aligned_candidates": {
            "path": str(aligned_output.resolve()),
            "rows": len(aligned),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_sha256": canonical_sha256(aligned),
        },
        "limitations": _ALIGNMENT_LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload, encoded


def align_candidates(
    candidate_receipt: Path,
    snapshot_receipt: Path,
    aligned_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Intersect candidates with current opt-out-enacted Stack v2 metadata."""

    payload, encoded = _alignment_payload(
        Path(candidate_receipt), Path(snapshot_receipt), Path(aligned_output)
    )
    receipt_encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    _publish_pair(Path(aligned_output), encoded, Path(receipt_output), receipt_encoded)
    return validate_alignment(Path(receipt_output))


def validate_alignment(receipt: Path) -> dict[str, Any]:
    """Replay the complete candidate/current-release intersection."""

    payload = _json(Path(receipt), "Stack v2 alignment receipt")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload) != _ALIGNMENT_KEYS
        or payload.get("schema") != ALIGNMENT_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("status")
        != "current_stack_v2_membership_aligned_content_not_acquired"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("limitations") != _ALIGNMENT_LIMITATIONS
    ):
        raise StackV2AlignmentError("Stack v2 alignment receipt differs")
    candidate = payload.get("candidate_aggregate")
    snapshot = payload.get("current_snapshot")
    output = payload.get("aligned_candidates")
    if (
        not isinstance(candidate, dict)
        or set(candidate) != {"path", "file_sha256", "receipt_sha256"}
        or not isinstance(snapshot, dict)
        or set(snapshot)
        != {
            "path",
            "file_sha256",
            "receipt_sha256",
            "dataset",
            "revision",
            "release",
            "opt_out_cutoff",
        }
        or not isinstance(output, dict)
        or set(output) != {"path", "rows", "bytes", "sha256", "ordered_sha256"}
    ):
        raise StackV2AlignmentError("Stack v2 alignment receipt differs")
    candidate_path = Path(candidate["path"])
    snapshot_path = Path(snapshot["path"])
    if (
        sha256_file(candidate_path) != candidate["file_sha256"]
        or sha256_file(snapshot_path) != snapshot["file_sha256"]
    ):
        raise StackV2AlignmentError("Stack v2 alignment parent differs")
    expected, encoded = _alignment_payload(
        candidate_path, snapshot_path, Path(output["path"])
    )
    aligned_path = Path(output["path"])
    metadata = _safe_regular(aligned_path, "Stack v2 aligned candidates")
    if (
        expected != payload
        or metadata.st_size != len(encoded)
        or aligned_path.read_bytes() != encoded
    ):
        raise StackV2AlignmentError("Stack v2 alignment replay differs")
    return payload


def _source_argument(value: str) -> tuple[str, Path]:
    source_file, separator, path = value.partition("=")
    if not separator or not source_file or not path:
        raise argparse.ArgumentTypeError("source must be SOURCE_FILE=ABSOLUTE_PATH")
    return source_file, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-snapshot")
    freeze.add_argument("--dataset-card", type=Path, required=True)
    freeze.add_argument("--access-evidence", type=Path, required=True)
    freeze.add_argument(
        "--source", type=_source_argument, action="append", required=True
    )
    freeze.add_argument("--receipt-output", type=Path, required=True)
    check_snapshot = commands.add_parser("validate-snapshot")
    check_snapshot.add_argument("--receipt", type=Path, required=True)
    align = commands.add_parser("align")
    align.add_argument("--candidate-receipt", type=Path, required=True)
    align.add_argument("--snapshot-receipt", type=Path, required=True)
    align.add_argument("--aligned-output", type=Path, required=True)
    align.add_argument("--receipt-output", type=Path, required=True)
    check_alignment = commands.add_parser("validate-alignment")
    check_alignment.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-snapshot":
        payload = freeze_current_snapshot(
            dataset_card=args.dataset_card,
            access_evidence=args.access_evidence,
            sources=args.source,
            receipt_output=args.receipt_output,
        )
    elif args.command == "validate-snapshot":
        payload = validate_current_snapshot(args.receipt)
    elif args.command == "align":
        payload = align_candidates(
            args.candidate_receipt,
            args.snapshot_receipt,
            args.aligned_output,
            args.receipt_output,
        )
    else:
        payload = validate_alignment(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
