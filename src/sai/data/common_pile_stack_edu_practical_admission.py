"""Exact-deduplicate and admit the practical Common Pile Stack-Edu overlay."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.common_pile_stack_edu_practical_scan import (
    EXPECTED_PARENTS,
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    SOURCE_ID,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    _schema,
    load_parents,
)
from sai.data.pleias_practical_admission import (
    _UPSERT,
    _load_quarantine_content_hashes,
    _load_signed,
    _open_database,
    _output_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-stack-edu-practical-admission-v1"
DEFAULT_MAXIMUM_TEXT_BYTES = 150_000_000_000
DEFAULT_OUTPUT_SHARDS = 32


class StackEduPracticalAdmissionError(RuntimeError):
    """Stack-Edu locator custody, deduplication, or accounting differs."""


def _valid_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_locator(row: dict[str, Any]) -> bool:
    licenses = row.get("licenses")
    return bool(
        row.get("schema") == LOCATOR_SCHEMA
        and row.get("source_id") == SOURCE_ID
        and row.get("source_repository") == SOURCE_REPOSITORY
        and row.get("source_revision") == SOURCE_REVISION
        and isinstance(row.get("source_path"), str)
        and row["source_path"].endswith(".json.gz")
        and _valid_hex(row.get("source_parent_sha256"), 64)
        and isinstance(row.get("source_row_index"), int)
        and row["source_row_index"] >= 0
        and _valid_hex(row.get("source_row_identity_sha256"), 64)
        and _valid_hex(row.get("blob_id"), 40)
        and isinstance(row.get("repo_name"), str)
        and row["repo_name"]
        and isinstance(row.get("repo_path"), str)
        and row["repo_path"].startswith("/")
        and isinstance(row.get("language"), str)
        and row["language"]
        and isinstance(licenses, list)
        and licenses
        and licenses == sorted(set(licenses))
        and all(isinstance(value, str) and value for value in licenses)
        and row.get("integer_score") in {3, 4}
        and isinstance(row.get("text_utf8_bytes"), int)
        and row["text_utf8_bytes"] > 0
        and _valid_hex(row.get("content_sha256"), 64)
    )


def build_admission(
    manifest_path: Path,
    scan_root: Path,
    quarantine_registry_root: Path,
    output_root: Path,
    *,
    expected_parents: int = EXPECTED_PARENTS,
    maximum_text_bytes: int = DEFAULT_MAXIMUM_TEXT_BYTES,
    output_shards: int = DEFAULT_OUTPUT_SHARDS,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Verify every source parent, exact-deduplicate, and cap the code overlay."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or isinstance(maximum_text_bytes, bool)
        or maximum_text_bytes <= 0
        or not 1 <= output_shards <= 95
    ):
        raise StackEduPracticalAdmissionError("Stack-Edu admission arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise StackEduPracticalAdmissionError("pyarrow is required") from error
    parents = load_parents(manifest_path, expected_parents)
    quarantine_hashes, quarantine_descriptor = _load_quarantine_content_hashes(
        quarantine_registry_root
    )
    output_root.mkdir(parents=True)
    counts: Counter[str] = Counter()
    receipt_hashes = []
    with tempfile.TemporaryDirectory(
        prefix="sai-stack-edu-practical-admission-", dir=scratch_root
    ) as temporary_directory:
        database = _open_database(Path(temporary_directory) / "exact-dedup.sqlite3")
        try:
            for shard_index, parent in enumerate(parents):
                shard_root = scan_root / "shards" / f"shard_{shard_index:05d}"
                receipt = _load_signed(shard_root / "receipt.json", SHARD_SCHEMA)
                descriptor = receipt.get("output", {})
                locator_path = shard_root / descriptor.get("path", "")
                if (
                    receipt.get("status")
                    != "complete_common_pile_stack_edu_practical_scan_shard"
                    or receipt.get("shard_index") != shard_index
                    or receipt.get("expected_parents") != expected_parents
                    or receipt.get("source", {}).get("manifest_sha256")
                    != sha256_file(manifest_path)
                    or receipt.get("source", {}).get("repository")
                    != SOURCE_REPOSITORY
                    or receipt.get("source", {}).get("revision") != SOURCE_REVISION
                    or receipt.get("source", {}).get("path") != parent["source_path"]
                    or receipt.get("source", {}).get("parent_bytes")
                    != parent["bytes"]
                    or receipt.get("source", {}).get("parent_sha256")
                    != parent["sha256"]
                    or receipt.get("full_parent_byte_identity_verified") is not True
                    or receipt.get("all_source_rows_accounted") is not True
                    or receipt.get("training_ready") is not False
                    or not locator_path.is_file()
                    or locator_path.is_symlink()
                    or locator_path.stat().st_nlink != 1
                    or locator_path.stat().st_size != descriptor.get("bytes")
                    or sha256_file(locator_path) != descriptor.get("sha256")
                ):
                    raise StackEduPracticalAdmissionError(
                        "Stack-Edu scan shard differs"
                    )
                parquet = pq.ParquetFile(locator_path)
                observed_rows = observed_bytes = 0
                for batch in parquet.iter_batches(batch_size=4096, use_threads=False):
                    values = []
                    for row in batch.to_pylist():
                        if (
                            not _valid_locator(row)
                            or row["source_path"] != parent["source_path"]
                            or row["source_parent_sha256"] != parent["sha256"]
                        ):
                            raise StackEduPracticalAdmissionError(
                                "Stack-Edu locator row differs"
                            )
                        observed_rows += 1
                        observed_bytes += row["text_utf8_bytes"]
                        if row["content_sha256"] in quarantine_hashes:
                            counts["known_quarantine_rows_excluded"] += 1
                            counts["known_quarantine_text_utf8_bytes_excluded"] += row[
                                "text_utf8_bytes"
                            ]
                            continue
                        values.append(
                            (
                                row["content_sha256"],
                                row["source_row_identity_sha256"],
                                _output_shard(row["source_path"], output_shards),
                                row["text_utf8_bytes"],
                                0,
                                json.dumps(
                                    row["licenses"], separators=(",", ":")
                                ),
                                json.dumps(
                                    row, sort_keys=True, separators=(",", ":")
                                ),
                            )
                        )
                    database.executemany(_UPSERT, values)
                selected = receipt.get("selected", {})
                if (
                    observed_rows != selected.get("rows")
                    or observed_bytes != selected.get("text_utf8_bytes")
                ):
                    raise StackEduPracticalAdmissionError(
                        "Stack-Edu scan accounting differs"
                    )
                counts["candidate_rows"] += observed_rows
                counts["candidate_text_utf8_bytes"] += observed_bytes
                counts["locator_parquet_bytes"] += descriptor["bytes"]
                receipt_hashes.append(receipt["receipt_sha256"])
                print(
                    json.dumps(
                        {
                            "event": "stack_edu_practical_admission_scan_progress",
                            "complete_scan_shards": shard_index + 1,
                            "remaining_scan_shards": expected_parents - shard_index - 1,
                            "candidate_rows": counts["candidate_rows"],
                            "candidate_text_utf8_bytes": counts[
                                "candidate_text_utf8_bytes"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            database.commit()
            unique_rows = database.execute("SELECT COUNT(*) FROM winners").fetchone()[
                0
            ]
            duplicate_rows = (
                counts["candidate_rows"]
                - counts["known_quarantine_rows_excluded"]
                - unique_rows
            )
            output_parent = output_root / "shards"
            output_parent.mkdir()
            schema = _schema()
            writers: dict[int, Any] = {}
            temporary_paths: dict[int, Path] = {}
            output_paths: dict[int, Path] = {}
            pending: dict[int, list[dict[str, Any]]] = {}
            shard_rows: Counter[int] = Counter()
            shard_bytes: Counter[int] = Counter()
            languages: Counter[str] = Counter()
            scores: Counter[int] = Counter()
            licenses: Counter[str] = Counter()
            admitted_rows = admitted_bytes = 0
            byte_cap_rows = byte_cap_bytes = 0

            def open_writer(target: int) -> None:
                root = output_parent / f"shard_{target:05d}"
                root.mkdir()
                temporary = root / f".locators.partial.{uuid.uuid4().hex}.parquet"
                writers[target] = pq.ParquetWriter(
                    temporary, schema, compression="zstd"
                )
                temporary_paths[target] = temporary
                output_paths[target] = root / "locators.parquet"
                pending[target] = []

            def flush(target: int) -> None:
                if pending[target]:
                    writers[target].write_table(
                        pa.Table.from_pylist(pending[target], schema=schema)
                    )
                    pending[target].clear()

            cursor = database.execute(
                "SELECT output_shard, text_utf8_bytes, row_json FROM winners "
                "ORDER BY content_sha256"
            )
            for target, text_bytes, row_json in cursor:
                if admitted_bytes + text_bytes > maximum_text_bytes:
                    byte_cap_rows += 1
                    byte_cap_bytes += text_bytes
                    continue
                if target not in writers:
                    open_writer(target)
                row = json.loads(row_json)
                pending[target].append(row)
                admitted_rows += 1
                admitted_bytes += text_bytes
                shard_rows[target] += 1
                shard_bytes[target] += text_bytes
                languages[row["language"]] += 1
                scores[row["integer_score"]] += 1
                licenses.update(row["licenses"])
                if len(pending[target]) >= 4096:
                    flush(target)
            descriptors = []
            for target in sorted(writers):
                flush(target)
                writers[target].close()
                os.replace(temporary_paths[target], output_paths[target])
                path = output_paths[target]
                descriptors.append(
                    {
                        "shard_index": target,
                        "path": str(path.relative_to(output_root)),
                        "rows": shard_rows[target],
                        "text_utf8_bytes": shard_bytes[target],
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        finally:
            database.close()
    if not descriptors or admitted_rows <= 0:
        raise StackEduPracticalAdmissionError("Stack-Edu practical admission is empty")
    payload = {
        "schema": SCHEMA,
        "status": "complete_common_pile_stack_edu_practical_admission",
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "source_parent_count": len(parents),
            "source_parent_bytes": sum(row["bytes"] for row in parents),
            "ordered_scan_receipts_sha256": canonical_sha256(receipt_hashes),
            "quarantine_registry": quarantine_descriptor,
        },
        "policy": {
            "exact_content_duplicate_policy": "smallest_identity_sha256_wins",
            "byte_cap_selection_policy": "canonical_content_sha256_order",
            "output_partition_policy": "canonical_source_path_sha256_modulo",
            "maximum_text_utf8_bytes": maximum_text_bytes,
            "known_quarantine_content_excluded": True,
        },
        "counts": {
            **dict(sorted(counts.items())),
            "unique_candidate_rows": unique_rows,
            "exact_duplicate_rows_excluded": duplicate_rows,
            "byte_cap_excluded_rows": byte_cap_rows,
            "byte_cap_excluded_text_utf8_bytes": byte_cap_bytes,
            "admitted_rows": admitted_rows,
            "admitted_text_utf8_bytes": admitted_bytes,
            "languages": dict(sorted(languages.items())),
            "integer_scores": {
                str(key): value for key, value in sorted(scores.items())
            },
            "licenses": dict(sorted(licenses.items())),
        },
        "outputs": {
            "shards": len(descriptors),
            "descriptors": descriptors,
            "ordered_descriptors_sha256": canonical_sha256(descriptors),
        },
        "complete_source_parent_content_scan": True,
        "global_exact_content_deduplication_complete": True,
        "known_quarantine_exclusions_applied": True,
        "global_near_deduplication_complete": False,
        "official_benchmark_decontamination_complete": False,
        "evaluation_claims_allowed": False,
        "source_text_copied": False,
        "practical_pretraining_ready": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--quarantine-registry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, default=EXPECTED_PARENTS)
    parser.add_argument(
        "--maximum-text-bytes", type=int, default=DEFAULT_MAXIMUM_TEXT_BYTES
    )
    parser.add_argument("--output-shards", type=int, default=DEFAULT_OUTPUT_SHARDS)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = build_admission(
        args.manifest,
        args.scan_root,
        args.quarantine_registry_root,
        args.output_root,
        expected_parents=args.expected_parents,
        maximum_text_bytes=args.maximum_text_bytes,
        output_shards=args.output_shards,
        scratch_root=args.scratch_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
