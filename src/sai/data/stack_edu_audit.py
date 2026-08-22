"""Audit exact Stack-Edu metadata shards before any code-content acquisition."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import stat
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-stack-edu-metadata-audit-v1"
SAMPLE_SCHEMA = "sai-stack-edu-metadata-review-row-v1"
DATASET = "HuggingFaceTB/stack-edu"
REVISION = "eeec5caac5cc3758a18f1d3ba4416837a9ba814c"
SELECTION_SALT = b"sai-stack-edu-metadata-review-v1"
LANGUAGES = {
    "C",
    "CSharp",
    "Cpp",
    "Go",
    "Java",
    "JavaScript",
    "Markdown",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "SQL",
    "Shell",
    "Swift",
    "TypeScript",
}
EXPECTED_COLUMNS = {
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
ALLOWED_LICENSES = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "ISC",
    "MIT",
    "Unlicense",
    "Zlib",
}
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "source",
    "policy",
    "policy_sha256",
    "summary",
    "review_sample",
    "limitations",
    "receipt_sha256",
}
_SOURCE_FILE = re.compile(
    r"(?P<language>[A-Za-z]+)/train-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.parquet"
)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class StackEduAuditError(RuntimeError):
    """The Stack-Edu metadata, policy, or replay evidence differs."""


def _safe_regular(path: Path, label: str) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise StackEduAuditError(f"{label} is missing or unsafe") from error
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_size <= 0:
        raise StackEduAuditError(f"{label} is missing or unsafe")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise StackEduAuditError(f"{label} differs")
    if not any(bytes.fromhex(value)):
        raise StackEduAuditError(f"{label} is a placeholder")
    return value


def _source_file(value: Any) -> tuple[str, str]:
    if not isinstance(value, str):
        raise StackEduAuditError("Stack-Edu source file differs")
    match = _SOURCE_FILE.fullmatch(value)
    if match is None or match.group("language") not in LANGUAGES:
        raise StackEduAuditError("Stack-Edu source file differs")
    if int(match.group("count")) <= 0 or int(match.group("index")) >= int(
        match.group("count")
    ):
        raise StackEduAuditError("Stack-Edu source file differs")
    return value, match.group("language")


def _import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise StackEduAuditError("PyArrow is required for Stack-Edu audit") from error
    return pq


def _policy() -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "exact_parquet_columns": sorted(EXPECTED_COLUMNS),
        "accepted_integer_scores": [4, 5],
        "accepted_license_type": "permissive",
        "accepted_detected_licenses": sorted(ALLOWED_LICENSES),
        "all_detected_licenses_must_be_allowlisted": True,
        "required_source_encoding": "UTF-8",
        "minimum_length_bytes": 128,
        "maximum_length_bytes": 1_000_000,
        "review_rows_per_policy_outcome": 32,
        "review_rank": "sha256_salt_plus_blob_id_plus_repo_plus_path",
    }


def _selected_by_policy(row: dict[str, Any]) -> bool:
    licenses = row["detected_licenses"]
    return (
        row["license_type"] == "permissive"
        and bool(licenses)
        and set(licenses).issubset(ALLOWED_LICENSES)
        and row["int_score"] in {4, 5}
        and row["src_encoding"] == "UTF-8"
        and 128 <= row["length_bytes"] <= 1_000_000
    )


def _push_sample(
    heap: list[tuple[int, int, dict[str, Any]]],
    *,
    rank: int,
    row_index: int,
    row: dict[str, Any],
    limit: int,
) -> None:
    item = (-rank, row_index, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _scan(
    path: Path, *, expected_language: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pq = _import_parquet()
    parquet = pq.ParquetFile(path)
    if set(parquet.schema_arrow.names) != EXPECTED_COLUMNS:
        raise StackEduAuditError("Stack-Edu Parquet columns differ")
    rows = 0
    total_bytes = 0
    candidate_rows = 0
    candidate_bytes = 0
    permissive_rows = 0
    permissive_rows_without_detected_license = 0
    no_license_rows = 0
    duplicate_blob_ids = 0
    duplicate_repo_paths = 0
    seen_blobs: set[str] = set()
    seen_paths: set[tuple[str, str]] = set()
    score_counts: Counter[int] = Counter()
    license_counts: Counter[str] = Counter()
    encoding_counts: Counter[str] = Counter()
    accepted: list[tuple[int, int, dict[str, Any]]] = []
    rejected: list[tuple[int, int, dict[str, Any]]] = []
    columns = sorted(EXPECTED_COLUMNS)
    for batch in parquet.iter_batches(batch_size=8_192, columns=columns):
        values = batch.to_pydict()
        for raw in zip(*(values[column] for column in columns), strict=True):
            row = dict(zip(columns, raw, strict=True))
            row_index = rows
            rows += 1
            blob_id = row["blob_id"]
            repo_name = row["repo_name"]
            source_path = row["path"]
            licenses = row["detected_licenses"]
            if (
                not isinstance(blob_id, str)
                or _SHA40.fullmatch(blob_id) is None
                or row["language"] != expected_language
                or not isinstance(repo_name, str)
                or not repo_name
                or repo_name.count("/") != 1
                or not isinstance(source_path, str)
                or not source_path.startswith("/")
                or not isinstance(row["src_encoding"], str)
                or not row["src_encoding"]
                or isinstance(row["length_bytes"], bool)
                or not isinstance(row["length_bytes"], int)
                or not 0 < row["length_bytes"] <= 10_000_000
                or isinstance(row["score"], bool)
                or not isinstance(row["score"], (int, float))
                or not math.isfinite(row["score"])
                or isinstance(row["int_score"], bool)
                or row["int_score"] not in {3, 4, 5}
                or not isinstance(licenses, list)
                or any(not isinstance(item, str) or not item for item in licenses)
                or len(licenses) != len(set(licenses))
                or row["license_type"] not in {"permissive", "no_license"}
            ):
                raise StackEduAuditError("Stack-Edu row fields differ")
            total_bytes += row["length_bytes"]
            score_counts[row["int_score"]] += 1
            encoding_counts[row["src_encoding"]] += 1
            for license_id in licenses:
                license_counts[license_id] += 1
            permissive_rows += row["license_type"] == "permissive"
            permissive_rows_without_detected_license += (
                row["license_type"] == "permissive" and not licenses
            )
            no_license_rows += row["license_type"] == "no_license"
            if blob_id in seen_blobs:
                duplicate_blob_ids += 1
            seen_blobs.add(blob_id)
            repo_path = (repo_name, source_path)
            if repo_path in seen_paths:
                duplicate_repo_paths += 1
            seen_paths.add(repo_path)
            selected = _selected_by_policy(row)
            candidate_rows += selected
            candidate_bytes += row["length_bytes"] if selected else 0
            rank = int.from_bytes(
                hashlib.sha256(
                    SELECTION_SALT
                    + bytes.fromhex(blob_id)
                    + repo_name.encode("utf-8")
                    + b"\0"
                    + source_path.encode("utf-8")
                ).digest(),
                "big",
            )
            sample = {
                "schema": SAMPLE_SCHEMA,
                "row_index": row_index,
                "selected_by_metadata_policy": selected,
                "blob_id": blob_id,
                "language": expected_language,
                "repo_name": repo_name,
                "path": source_path,
                "src_encoding": row["src_encoding"],
                "length_bytes": row["length_bytes"],
                "score": row["score"],
                "int_score": row["int_score"],
                "detected_licenses": licenses,
                "license_type": row["license_type"],
            }
            _push_sample(
                accepted if selected else rejected,
                rank=rank,
                row_index=row_index,
                row=sample,
                limit=32,
            )
    if not rows:
        raise StackEduAuditError("Stack-Edu shard contains no rows")
    samples = [
        item[2] for item in sorted(accepted + rejected, key=lambda item: -item[0])
    ]
    return (
        {
            "rows": rows,
            "declared_content_bytes": total_bytes,
            "permissive_rows": permissive_rows,
            "permissive_rows_without_detected_license": (
                permissive_rows_without_detected_license
            ),
            "no_license_rows": no_license_rows,
            "candidate_rows": candidate_rows,
            "candidate_declared_content_bytes": candidate_bytes,
            "candidate_fraction_ppm": candidate_rows * 1_000_000 // rows,
            "integer_score_counts": {
                str(score): score_counts[score] for score in (3, 4, 5)
            },
            "detected_license_counts": dict(sorted(license_counts.items())),
            "source_encoding_counts": dict(sorted(encoding_counts.items())),
            "unique_blob_ids": len(seen_blobs),
            "duplicate_blob_ids": duplicate_blob_ids,
            "unique_repo_paths": len(seen_paths),
            "duplicate_repo_paths": duplicate_repo_paths,
        },
        samples,
    )


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def audit_shard(
    source: Path,
    *,
    source_file: str,
    expected_bytes: int,
    expected_sha256: str,
    sample_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Audit one exact metadata shard and publish no code content."""

    source_file, language = _source_file(source_file)
    expected_sha256 = _sha256(expected_sha256, "Stack-Edu source SHA-256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise StackEduAuditError("Stack-Edu source bytes differ")
    source_stat = _safe_regular(source, "Stack-Edu source")
    if source_stat.st_size != expected_bytes or sha256_file(source) != expected_sha256:
        raise StackEduAuditError("Stack-Edu source content differs")
    if sample_output.parent != receipt_output.parent or any(
        path.exists() or path.is_symlink() for path in (sample_output, receipt_output)
    ):
        raise StackEduAuditError("Stack-Edu audit output boundary differs")
    summary, samples = _scan(source, expected_language=language)
    sample_encoded = _jsonl(samples)
    policy = _policy()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "metadata_audited_content_not_acquired",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "source": {
            "dataset": DATASET,
            "revision": REVISION,
            "source_file": source_file,
            "path": str(source.resolve()),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        },
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "summary": summary,
        "review_sample": {
            "schema": SAMPLE_SCHEMA,
            "selection": "lowest_salted_rank_balanced_by_metadata_policy_outcome",
            "rows": len(samples),
            "path": str(sample_output.resolve()),
            "bytes": len(sample_encoded),
            "sha256": hashlib.sha256(sample_encoded).hexdigest(),
            "ordered_sha256": canonical_sha256(samples),
        },
        "limitations": [
            "metadata_only_code_content_not_downloaded_or_inspected",
            "current_opt_out_snapshot_not_replayed",
            "license_detection_is_not_legal_approval",
            "secret_scan_deduplication_and_benchmark_decontamination_pending",
            "audit_authorizes_no_training_or_source_retention",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = (
        json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    sample_stage = sample_output.with_name(f".{sample_output.name}.partial.{token}")
    receipt_stage = receipt_output.with_name(f".{receipt_output.name}.partial.{token}")
    linked: list[Path] = []
    try:
        for path, encoded in (
            (sample_stage, sample_encoded),
            (receipt_stage, receipt_encoded),
        ):
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        for stage, target in (
            (sample_stage, sample_output),
            (receipt_stage, receipt_output),
        ):
            os.link(stage, target)
            linked.append(target)
        directory = os.open(sample_output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        for target in reversed(linked):
            target.unlink(missing_ok=True)
        raise StackEduAuditError("Stack-Edu audit output boundary differs") from error
    except BaseException:
        for target in reversed(linked):
            target.unlink(missing_ok=True)
        raise
    finally:
        sample_stage.unlink(missing_ok=True)
        receipt_stage.unlink(missing_ok=True)
    return validate_audit(receipt_output)


def validate_audit(receipt: Path) -> dict[str, Any]:
    """Reopen the exact shard and replay every metadata decision."""

    _safe_regular(receipt, "Stack-Edu receipt")
    try:
        payload = json.loads(receipt.read_text())
    except json.JSONDecodeError as error:
        raise StackEduAuditError("Stack-Edu receipt differs") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_KEYS
        or payload.get("schema") != SCHEMA
    ):
        raise StackEduAuditError("Stack-Edu receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise StackEduAuditError("Stack-Edu receipt hash differs")
    source_row = payload.get("source")
    if not isinstance(source_row, dict) or set(source_row) != {
        "dataset",
        "revision",
        "source_file",
        "path",
        "bytes",
        "sha256",
    }:
        raise StackEduAuditError("Stack-Edu source differs")
    source_file, language = _source_file(source_row["source_file"])
    source = Path(source_row["path"])
    source_stat = _safe_regular(source, "Stack-Edu source")
    if (
        source_row["dataset"] != DATASET
        or source_row["revision"] != REVISION
        or source_file != source_row["source_file"]
        or source_stat.st_size != source_row["bytes"]
        or sha256_file(source) != _sha256(source_row["sha256"], "source SHA-256")
    ):
        raise StackEduAuditError("Stack-Edu source differs")
    summary, samples = _scan(source, expected_language=language)
    sample_encoded = _jsonl(samples)
    review = payload.get("review_sample")
    if not isinstance(review, dict):
        raise StackEduAuditError("Stack-Edu sample differs")
    sample = Path(review.get("path", ""))
    sample_stat = _safe_regular(sample, "Stack-Edu sample")
    policy = _policy()
    if (
        payload.get("status") != "metadata_audited_content_not_acquired"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("policy") != policy
        or payload.get("policy_sha256") != canonical_sha256(policy)
        or payload.get("summary") != summary
        or review
        != {
            "schema": SAMPLE_SCHEMA,
            "selection": "lowest_salted_rank_balanced_by_metadata_policy_outcome",
            "rows": len(samples),
            "path": str(sample.resolve()),
            "bytes": len(sample_encoded),
            "sha256": hashlib.sha256(sample_encoded).hexdigest(),
            "ordered_sha256": canonical_sha256(samples),
        }
        or sample_stat.st_size != len(sample_encoded)
        or sample.read_bytes() != sample_encoded
        or payload.get("limitations")
        != [
            "metadata_only_code_content_not_downloaded_or_inspected",
            "current_opt_out_snapshot_not_replayed",
            "license_detection_is_not_legal_approval",
            "secret_scan_deduplication_and_benchmark_decontamination_pending",
            "audit_authorizes_no_training_or_source_retention",
        ]
    ):
        raise StackEduAuditError("Stack-Edu audit replay differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--source-file", required=True)
    audit.add_argument("--expected-bytes", type=int, required=True)
    audit.add_argument("--expected-sha256", required=True)
    audit.add_argument("--sample-output", type=Path, required=True)
    audit.add_argument("--receipt-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        payload = audit_shard(
            args.source,
            source_file=args.source_file,
            expected_bytes=args.expected_bytes,
            expected_sha256=args.expected_sha256,
            sample_output=args.sample_output,
            receipt_output=args.receipt_output,
        )
    else:
        payload = validate_audit(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
