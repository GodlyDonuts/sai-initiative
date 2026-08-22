"""Freeze and deduplicate complete Stack-Edu candidate identity manifests."""

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

from sai.data.stack_edu_aggregate import (
    StackEduAggregateError,
)
from sai.data.stack_edu_aggregate import (
    validate_aggregate as validate_metadata_aggregate,
)
from sai.data.stack_edu_audit import (
    EXPECTED_COLUMNS,
    StackEduAuditError,
    _import_parquet,
    _selected_by_policy,
    validate_audit,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-stack-edu-candidate-identity-row-v1"
SHARD_SCHEMA = "sai-stack-edu-candidate-identity-shard-v1"
AGGREGATE_SCHEMA = "sai-stack-edu-candidate-identity-aggregate-v1"
_MEMBER = re.compile(
    r"(?P<language>[A-Za-z]+)/train-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.parquet"
)
_ROW_KEYS = {
    "schema",
    "source_file",
    "source_row_index",
    "blob_id",
    "language",
    "repo_name",
    "path",
    "src_encoding",
    "length_bytes",
    "score",
    "int_score",
    "detected_licenses",
    "license_type",
}
_SHARD_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "source_audit",
    "source",
    "candidates",
    "limitations",
    "receipt_sha256",
}
_AGGREGATE_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "metadata_aggregate",
    "inputs",
    "summary",
    "candidates",
    "limitations",
    "receipt_sha256",
}
_SHARD_LIMITATIONS = [
    "candidate_identity_metadata_only_code_content_not_acquired",
    "current_opt_out_snapshot_not_replayed",
    "license_detection_is_not_legal_approval",
    "secret_scan_content_deduplication_and_benchmark_decontamination_pending",
    "candidate_manifest_authorizes_no_training_or_source_retention",
]
_AGGREGATE_LIMITATIONS = [
    "candidate_identity_metadata_only_code_content_not_acquired",
    "duplicate_blob_ids_removed_by_exact_upstream_sha1_identity",
    "sha1_identity_must_be_reverified_against_acquired_content_bytes",
    "repo_path_collisions_with_distinct_blob_ids_are_reported_not_removed",
    "current_opt_out_snapshot_not_replayed",
    "license_detection_is_not_legal_approval",
    "secret_scan_content_deduplication_and_benchmark_decontamination_pending",
    "candidate_aggregate_authorizes_no_training_or_source_retention",
]


class StackEduCandidateError(RuntimeError):
    """The candidate identity population or its provenance differs."""


def _safe_regular(path: Path, label: str) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise StackEduCandidateError(f"{label} is missing or unsafe") from error
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_size <= 0:
        raise StackEduCandidateError(f"{label} is missing or unsafe")
    return value


def _json(path: Path, label: str, maximum: int = 64 << 20) -> tuple[Any, bytes]:
    metadata = _safe_regular(path, label)
    if metadata.st_size > maximum:
        raise StackEduCandidateError(f"{label} differs")
    try:
        encoded = path.read_bytes()
        return json.loads(encoded), encoded
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduCandidateError(f"{label} differs") from error


def _member(value: Any) -> tuple[str, int, int]:
    if not isinstance(value, str):
        raise StackEduCandidateError("Stack-Edu source member differs")
    match = _MEMBER.fullmatch(value)
    if match is None:
        raise StackEduCandidateError("Stack-Edu source member differs")
    index = int(match.group("index"))
    count = int(match.group("count"))
    if count <= 0 or index >= count:
        raise StackEduCandidateError("Stack-Edu source member differs")
    return match.group("language"), index, count


def _candidate_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(audit["source"]["path"])
    source_file = audit["source"]["source_file"]
    language, _index, _count = _member(source_file)
    before = _safe_regular(source, "Stack-Edu source")
    source_sha256 = sha256_file(source)
    pq = _import_parquet()
    parquet = pq.ParquetFile(source)
    if set(parquet.schema_arrow.names) != EXPECTED_COLUMNS:
        raise StackEduCandidateError("Stack-Edu Parquet columns differ")
    columns = sorted(EXPECTED_COLUMNS)
    output = []
    row_index = 0
    for batch in parquet.iter_batches(batch_size=8_192, columns=columns):
        values = batch.to_pydict()
        for raw in zip(*(values[column] for column in columns), strict=True):
            row = dict(zip(columns, raw, strict=True))
            if _selected_by_policy(row):
                output.append(
                    {
                        "schema": ROW_SCHEMA,
                        "source_file": source_file,
                        "source_row_index": row_index,
                        "blob_id": row["blob_id"],
                        "language": language,
                        "repo_name": row["repo_name"],
                        "path": row["path"],
                        "src_encoding": row["src_encoding"],
                        "length_bytes": row["length_bytes"],
                        "score": row["score"],
                        "int_score": row["int_score"],
                        "detected_licenses": row["detected_licenses"],
                        "license_type": row["license_type"],
                    }
                )
            row_index += 1
    after = source.stat(follow_symlinks=False)
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
    ) or sha256_file(
        source
    ) != source_sha256:
        raise StackEduCandidateError("Stack-Edu source changed while reading")
    if (
        row_index != audit["summary"]["rows"]
        or len(output) != audit["summary"]["candidate_rows"]
        or sum(row["length_bytes"] for row in output)
        != audit["summary"]["candidate_declared_content_bytes"]
    ):
        raise StackEduCandidateError("Stack-Edu candidate replay differs")
    return output


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _publish_pair(
    first: Path, first_encoded: bytes, second: Path, second_encoded: bytes
) -> None:
    if (
        first.parent != second.parent
        or not first.parent.is_dir()
        or any(path.exists() or path.is_symlink() for path in (first, second))
    ):
        raise StackEduCandidateError("candidate output boundary differs")
    token = uuid.uuid4().hex
    stages = [
        first.with_name(f".{first.name}.{token}.partial"),
        second.with_name(f".{second.name}.{token}.partial"),
    ]
    linked: list[Path] = []
    try:
        for path, encoded in zip(stages, (first_encoded, second_encoded), strict=True):
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        for stage, target in zip(stages, (first, second), strict=True):
            os.link(stage, target)
            linked.append(target)
        directory = os.open(first.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        for target in reversed(linked):
            target.unlink(missing_ok=True)
        raise StackEduCandidateError("candidate output boundary differs") from error
    except BaseException:
        for target in reversed(linked):
            target.unlink(missing_ok=True)
        raise
    finally:
        for stage in stages:
            stage.unlink(missing_ok=True)


def extract_candidates(
    audit_receipt: Path, candidates_output: Path, receipt_output: Path
) -> dict[str, Any]:
    """Replay one metadata audit and freeze every selected identity."""

    audit_receipt = Path(audit_receipt)
    audit_file_sha256 = sha256_file(audit_receipt)
    try:
        audit = validate_audit(audit_receipt)
    except StackEduAuditError as error:
        raise StackEduCandidateError("Stack-Edu source audit differs") from error
    if sha256_file(audit_receipt) != audit_file_sha256:
        raise StackEduCandidateError("Stack-Edu source audit changed while reading")
    rows = _candidate_rows(audit)
    encoded = _jsonl(rows)
    payload: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "status": "candidate_identity_metadata_frozen_content_not_acquired",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "source_audit": {
            "path": str(audit_receipt.resolve()),
            "file_sha256": audit_file_sha256,
            "receipt_sha256": audit["receipt_sha256"],
        },
        "source": audit["source"],
        "candidates": {
            "path": str(Path(candidates_output).resolve()),
            "rows": len(rows),
            "declared_content_bytes": sum(row["length_bytes"] for row in rows),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_sha256": canonical_sha256(rows),
        },
        "limitations": _SHARD_LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    _publish_pair(
        Path(candidates_output), encoded, Path(receipt_output), receipt_encoded
    )
    return validate_shard(Path(receipt_output))


def validate_shard(receipt: Path) -> dict[str, Any]:
    """Reopen a candidate shard receipt and replay every selected row."""

    payload, _encoded = _json(Path(receipt), "candidate shard receipt")
    if not isinstance(payload, dict) or set(payload) != _SHARD_KEYS:
        raise StackEduCandidateError("candidate shard receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != SHARD_SCHEMA or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise StackEduCandidateError("candidate shard receipt hash differs")
    source_audit = payload.get("source_audit")
    if not isinstance(source_audit, dict) or set(source_audit) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise StackEduCandidateError("candidate source audit differs")
    audit_path = Path(source_audit["path"])
    if sha256_file(audit_path) != source_audit["file_sha256"]:
        raise StackEduCandidateError("candidate source audit differs")
    try:
        audit = validate_audit(audit_path)
    except StackEduAuditError as error:
        raise StackEduCandidateError("candidate source audit differs") from error
    if (
        audit["receipt_sha256"] != source_audit["receipt_sha256"]
        or payload.get("source") != audit["source"]
    ):
        raise StackEduCandidateError("candidate source audit differs")
    rows = _candidate_rows(audit)
    encoded = _jsonl(rows)
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {
        "path",
        "rows",
        "declared_content_bytes",
        "bytes",
        "sha256",
        "ordered_sha256",
    }:
        raise StackEduCandidateError("candidate population differs")
    output = Path(candidates["path"])
    metadata = _safe_regular(output, "candidate population")
    if (
        payload.get("status")
        != "candidate_identity_metadata_frozen_content_not_acquired"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or candidates["rows"] != len(rows)
        or candidates["declared_content_bytes"]
        != sum(row["length_bytes"] for row in rows)
        or candidates["bytes"] != len(encoded)
        or candidates["sha256"] != hashlib.sha256(encoded).hexdigest()
        or candidates["ordered_sha256"] != canonical_sha256(rows)
        or metadata.st_size != len(encoded)
        or output.read_bytes() != encoded
        or payload.get("limitations") != _SHARD_LIMITATIONS
    ):
        raise StackEduCandidateError("candidate population replay differs")
    return payload


def _read_candidate_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(receipt["candidates"]["path"])
    _safe_regular(path, "candidate population")
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != _ROW_KEYS:
                    raise StackEduCandidateError("candidate row differs")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduCandidateError("candidate population differs") from error
    if len(rows) != receipt["candidates"]["rows"]:
        raise StackEduCandidateError("candidate population differs")
    return rows


def _aggregate_payload(
    metadata_aggregate: Path,
    shard_receipts: list[Path],
    candidates_output: Path,
) -> tuple[dict[str, Any], bytes]:
    metadata_file_sha256 = sha256_file(metadata_aggregate)
    try:
        metadata = validate_metadata_aggregate(metadata_aggregate)
    except StackEduAggregateError as error:
        raise StackEduCandidateError("metadata aggregate differs") from error
    if sha256_file(metadata_aggregate) != metadata_file_sha256:
        raise StackEduCandidateError("metadata aggregate changed while reading")
    validated = []
    inodes: set[tuple[int, int]] = set()
    for path in shard_receipts:
        before = _safe_regular(path, "candidate shard receipt")
        inode = (before.st_dev, before.st_ino)
        if inode in inodes:
            raise StackEduCandidateError("candidate shard receipt is duplicated")
        inodes.add(inode)
        file_sha256 = sha256_file(path)
        receipt = validate_shard(path)
        language, index, count = _member(receipt["source"]["source_file"])
        validated.append((index, count, language, path, file_sha256, receipt))
    counts = {row[1] for row in validated}
    languages = {row[2] for row in validated}
    if len(counts) != 1 or len(languages) != 1:
        raise StackEduCandidateError("candidate shard geometry differs")
    count = counts.pop()
    language = languages.pop()
    if len(validated) != count or {row[0] for row in validated} != set(range(count)):
        raise StackEduCandidateError("candidate shard set is incomplete")
    if metadata["language"] != language or metadata["shard_count"] != count:
        raise StackEduCandidateError("candidate metadata aggregate differs")
    validated.sort(key=lambda row: row[0])
    metadata_inputs = {row["source_file"]: row for row in metadata["inputs"]}
    rows: list[dict[str, Any]] = []
    inputs = []
    for index, _count, _language, path, file_sha256, receipt in validated:
        source_file = receipt["source"]["source_file"]
        parent = metadata_inputs.get(source_file)
        if (
            parent is None
            or parent["receipt_path"] != receipt["source_audit"]["path"]
            or parent["receipt_file_sha256"] != receipt["source_audit"]["file_sha256"]
            or parent["receipt_sha256"] != receipt["source_audit"]["receipt_sha256"]
        ):
            raise StackEduCandidateError("candidate metadata lineage differs")
        shard_rows = _read_candidate_rows(receipt)
        if any(row["source_file"] != source_file for row in shard_rows):
            raise StackEduCandidateError("candidate source member differs")
        rows.extend(shard_rows)
        inputs.append(
            {
                "index": index,
                "source_file": source_file,
                "receipt_path": str(path.resolve()),
                "receipt_file_sha256": file_sha256,
                "receipt_sha256": receipt["receipt_sha256"],
                "candidate_rows": len(shard_rows),
                "candidate_sha256": receipt["candidates"]["sha256"],
            }
        )
    if (
        len(rows) != metadata["summary"]["candidate_rows"]
        or sum(row["length_bytes"] for row in rows)
        != metadata["summary"]["candidate_declared_content_bytes"]
    ):
        raise StackEduCandidateError("candidate aggregate population differs")
    blob_counts = Counter(row["blob_id"] for row in rows)
    path_counts = Counter((row["repo_name"], row["path"]) for row in rows)
    seen_blobs = set()
    blob_geometry: dict[str, tuple[str, str, int]] = {}
    unique_rows = []
    for row in rows:
        geometry = (row["language"], row["src_encoding"], row["length_bytes"])
        previous = blob_geometry.setdefault(row["blob_id"], geometry)
        if previous != geometry:
            raise StackEduCandidateError("duplicate blob geometry differs")
        if row["blob_id"] not in seen_blobs:
            seen_blobs.add(row["blob_id"])
            unique_rows.append(row)
    unique_encoded = _jsonl(unique_rows)
    summary = {
        "input_candidate_rows": len(rows),
        "input_declared_content_bytes": sum(row["length_bytes"] for row in rows),
        "unique_blob_rows": len(unique_rows),
        "unique_blob_declared_content_bytes": sum(
            row["length_bytes"] for row in unique_rows
        ),
        "duplicate_blob_rows": len(rows) - len(unique_rows),
        "duplicate_blob_groups": sum(value > 1 for value in blob_counts.values()),
        "unique_repo_paths": len(path_counts),
        "duplicate_repo_path_rows": len(rows) - len(path_counts),
        "duplicate_repo_path_groups": sum(value > 1 for value in path_counts.values()),
        "cross_shard_duplicate_identity_check_complete": True,
        "deduplication_key": "upstream_blob_id_sha1",
        "canonical_occurrence": "lowest_source_shard_then_source_row_index",
    }
    payload: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "status": "candidate_identity_population_deduplicated_content_not_acquired",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "metadata_aggregate": {
            "path": str(metadata_aggregate.resolve()),
            "file_sha256": metadata_file_sha256,
            "receipt_sha256": metadata["receipt_sha256"],
        },
        "inputs": inputs,
        "summary": summary,
        "candidates": {
            "path": str(candidates_output.resolve()),
            "rows": len(unique_rows),
            "bytes": len(unique_encoded),
            "sha256": hashlib.sha256(unique_encoded).hexdigest(),
            "ordered_sha256": canonical_sha256(unique_rows),
        },
        "limitations": _AGGREGATE_LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload, unique_encoded


def aggregate_candidates(
    metadata_aggregate: Path,
    shard_receipts: list[Path],
    candidates_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Bind a complete shard set and remove cross-shard blob duplicates."""

    payload, encoded = _aggregate_payload(
        Path(metadata_aggregate),
        [Path(path) for path in shard_receipts],
        Path(candidates_output),
    )
    receipt_encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    _publish_pair(
        Path(candidates_output), encoded, Path(receipt_output), receipt_encoded
    )
    return validate_candidate_aggregate(Path(receipt_output))


def validate_candidate_aggregate(receipt: Path) -> dict[str, Any]:
    """Reopen and independently replay a candidate identity aggregate."""

    payload, _encoded = _json(Path(receipt), "candidate aggregate receipt")
    if not isinstance(payload, dict) or set(payload) != _AGGREGATE_KEYS:
        raise StackEduCandidateError("candidate aggregate receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != AGGREGATE_SCHEMA or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise StackEduCandidateError("candidate aggregate receipt hash differs")
    parent = payload.get("metadata_aggregate")
    inputs = payload.get("inputs")
    candidates = payload.get("candidates")
    if (
        not isinstance(parent, dict)
        or set(parent) != {"path", "file_sha256", "receipt_sha256"}
        or not isinstance(inputs, list)
        or not inputs
        or not isinstance(candidates, dict)
        or set(candidates)
        != {
            "path",
            "rows",
            "bytes",
            "sha256",
            "ordered_sha256",
        }
    ):
        raise StackEduCandidateError("candidate aggregate receipt differs")
    metadata_path = Path(parent["path"])
    if sha256_file(metadata_path) != parent["file_sha256"]:
        raise StackEduCandidateError("candidate aggregate parent differs")
    shard_paths = []
    for row in inputs:
        if not isinstance(row, dict) or set(row) != {
            "index",
            "source_file",
            "receipt_path",
            "receipt_file_sha256",
            "receipt_sha256",
            "candidate_rows",
            "candidate_sha256",
        }:
            raise StackEduCandidateError("candidate aggregate inputs differ")
        path = Path(row["receipt_path"])
        if sha256_file(path) != row["receipt_file_sha256"]:
            raise StackEduCandidateError("candidate aggregate input differs")
        shard_paths.append(path)
    expected, encoded = _aggregate_payload(
        metadata_path, shard_paths, Path(candidates["path"])
    )
    output = Path(candidates["path"])
    metadata = _safe_regular(output, "candidate aggregate population")
    if (
        expected != payload
        or metadata.st_size != len(encoded)
        or output.read_bytes() != encoded
        or payload.get("status")
        != "candidate_identity_population_deduplicated_content_not_acquired"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("limitations") != _AGGREGATE_LIMITATIONS
    ):
        raise StackEduCandidateError("candidate aggregate replay differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--audit-receipt", type=Path, required=True)
    extract.add_argument("--candidates-output", type=Path, required=True)
    extract.add_argument("--receipt-output", type=Path, required=True)
    validate_one = subparsers.add_parser("validate-shard")
    validate_one.add_argument("--receipt", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--metadata-aggregate", type=Path, required=True)
    aggregate.add_argument("--shard-receipt", type=Path, action="append", required=True)
    aggregate.add_argument("--candidates-output", type=Path, required=True)
    aggregate.add_argument("--receipt-output", type=Path, required=True)
    validate_all = subparsers.add_parser("validate-aggregate")
    validate_all.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        payload = extract_candidates(
            args.audit_receipt, args.candidates_output, args.receipt_output
        )
    elif args.command == "validate-shard":
        payload = validate_shard(args.receipt)
    elif args.command == "aggregate":
        payload = aggregate_candidates(
            args.metadata_aggregate,
            args.shard_receipt,
            args.candidates_output,
            args.receipt_output,
        )
    else:
        payload = validate_candidate_aggregate(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
