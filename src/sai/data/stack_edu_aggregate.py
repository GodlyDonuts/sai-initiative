"""Aggregate a complete Stack-Edu language metadata population fail closed."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.stack_edu_audit import (
    DATASET,
    REVISION,
    StackEduAuditError,
    validate_audit,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-stack-edu-language-metadata-aggregate-v1"
_MEMBER = re.compile(
    r"(?P<language>[A-Za-z]+)/train-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.parquet"
)
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "dataset",
    "revision",
    "language",
    "shard_count",
    "policy_sha256",
    "inputs",
    "summary",
    "limitations",
    "receipt_sha256",
}
_SUM_FIELDS = (
    "rows",
    "declared_content_bytes",
    "permissive_rows",
    "permissive_rows_without_detected_license",
    "no_license_rows",
    "candidate_rows",
    "candidate_declared_content_bytes",
    "duplicate_blob_ids",
    "duplicate_repo_paths",
)
_COUNTER_FIELDS = (
    "integer_score_counts",
    "detected_license_counts",
    "source_encoding_counts",
)
_LIMITATIONS = [
    "metadata_only_code_content_not_downloaded_or_inspected",
    "current_opt_out_snapshot_not_replayed",
    "license_detection_is_not_legal_approval",
    "cross_shard_duplicate_identities_not_measured",
    "secret_scan_content_deduplication_and_benchmark_decontamination_pending",
    "aggregate_authorizes_no_training_or_source_retention",
]


class StackEduAggregateError(RuntimeError):
    """A Stack-Edu language shard set or aggregate receipt differs."""


def _safe_regular(path: Path, label: str) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise StackEduAggregateError(f"{label} is missing or unsafe") from error
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_size <= 0:
        raise StackEduAggregateError(f"{label} is missing or unsafe")
    return value


def _member(value: Any) -> tuple[str, int, int]:
    if not isinstance(value, str):
        raise StackEduAggregateError("Stack-Edu source member differs")
    match = _MEMBER.fullmatch(value)
    if match is None:
        raise StackEduAggregateError("Stack-Edu source member differs")
    index = int(match.group("index"))
    count = int(match.group("count"))
    if count <= 0 or index >= count:
        raise StackEduAggregateError("Stack-Edu source member differs")
    return match.group("language"), index, count


def _validated_receipt(path: Path) -> tuple[dict[str, Any], str, os.stat_result]:
    before = _safe_regular(path, "Stack-Edu shard receipt")
    file_sha256 = sha256_file(path)
    try:
        payload = validate_audit(path)
    except StackEduAuditError as error:
        raise StackEduAggregateError("Stack-Edu shard replay failed") from error
    after = path.stat(follow_symlinks=False)
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
        path
    ) != file_sha256:
        raise StackEduAggregateError("Stack-Edu shard receipt changed while reading")
    return payload, file_sha256, after


def _build_payload(receipts: list[Path]) -> dict[str, Any]:
    if not isinstance(receipts, list) or not receipts:
        raise StackEduAggregateError("Stack-Edu shard population is empty")
    rows = []
    inodes: set[tuple[int, int]] = set()
    for path in receipts:
        payload, file_sha256, metadata = _validated_receipt(Path(path))
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes:
            raise StackEduAggregateError("Stack-Edu shard receipt is duplicated")
        inodes.add(inode)
        source = payload["source"]
        language, index, count = _member(source["source_file"])
        rows.append((index, count, language, payload, file_sha256, Path(path)))
    counts = {row[1] for row in rows}
    languages = {row[2] for row in rows}
    if len(counts) != 1 or len(languages) != 1:
        raise StackEduAggregateError("Stack-Edu language shard geometry differs")
    shard_count = counts.pop()
    language = languages.pop()
    if len(rows) != shard_count or {row[0] for row in rows} != set(range(shard_count)):
        raise StackEduAggregateError("Stack-Edu language shard set is incomplete")
    rows.sort(key=lambda row: row[0])
    policy_hashes = {row[3]["policy_sha256"] for row in rows}
    if len(policy_hashes) != 1:
        raise StackEduAggregateError("Stack-Edu metadata policies differ")
    totals = {field: 0 for field in _SUM_FIELDS}
    counters = {field: Counter() for field in _COUNTER_FIELDS}
    inputs = []
    for index, _count, _language, payload, file_sha256, path in rows:
        summary = payload["summary"]
        for field in _SUM_FIELDS:
            value = summary.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StackEduAggregateError("Stack-Edu shard summary differs")
            totals[field] += value
        for field in _COUNTER_FIELDS:
            values = summary.get(field)
            if not isinstance(values, dict) or any(
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in values.items()
            ):
                raise StackEduAggregateError("Stack-Edu shard counters differ")
            counters[field].update(values)
        inputs.append(
            {
                "index": index,
                "source_file": payload["source"]["source_file"],
                "source_bytes": payload["source"]["bytes"],
                "source_sha256": payload["source"]["sha256"],
                "receipt_path": str(path.resolve()),
                "receipt_file_sha256": file_sha256,
                "receipt_sha256": payload["receipt_sha256"],
                "review_sample_sha256": payload["review_sample"]["sha256"],
            }
        )
    total_rows = totals["rows"]
    if total_rows <= 0:
        raise StackEduAggregateError("Stack-Edu language population is empty")
    summary = {
        "shards": shard_count,
        **totals,
        "candidate_fraction_ppm": totals["candidate_rows"] * 1_000_000 // total_rows,
        **{field: dict(sorted(counter.items())) for field, counter in counters.items()},
        "cross_shard_duplicate_identity_check_complete": False,
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "language_metadata_audited_content_not_acquired",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "dataset": DATASET,
        "revision": REVISION,
        "language": language,
        "shard_count": shard_count,
        "policy_sha256": policy_hashes.pop(),
        "inputs": inputs,
        "summary": summary,
        "limitations": _LIMITATIONS,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise StackEduAggregateError("Stack-Edu aggregate output is unsafe")
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, path)
        linked = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise StackEduAggregateError(
            "Stack-Edu aggregate output already exists"
        ) from error
    except BaseException:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)


def aggregate_audits(receipts: list[Path], output: Path) -> dict[str, Any]:
    """Replay every shard and publish a complete language aggregate."""

    payload = _build_payload([Path(path) for path in receipts])
    _atomic_json(Path(output), payload)
    return validate_aggregate(Path(output))


def validate_aggregate(path: Path) -> dict[str, Any]:
    """Reopen the aggregate and independently replay its full shard set."""

    _safe_regular(path, "Stack-Edu aggregate receipt")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StackEduAggregateError("Stack-Edu aggregate receipt differs") from error
    if not isinstance(payload, dict) or set(payload) != _TOP_KEYS:
        raise StackEduAggregateError("Stack-Edu aggregate receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != SCHEMA or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise StackEduAggregateError("Stack-Edu aggregate receipt hash differs")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise StackEduAggregateError("Stack-Edu aggregate inputs differ")
    receipt_paths = []
    for row in inputs:
        if not isinstance(row, dict) or not isinstance(row.get("receipt_path"), str):
            raise StackEduAggregateError("Stack-Edu aggregate inputs differ")
        receipt_path = Path(row["receipt_path"])
        if sha256_file(receipt_path) != row.get("receipt_file_sha256"):
            raise StackEduAggregateError("Stack-Edu aggregate input hash differs")
        receipt_paths.append(receipt_path)
    if _build_payload(receipt_paths) != payload:
        raise StackEduAggregateError("Stack-Edu aggregate replay differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--receipt", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "aggregate":
        payload = aggregate_audits(args.receipt, args.output)
    else:
        payload = validate_aggregate(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
