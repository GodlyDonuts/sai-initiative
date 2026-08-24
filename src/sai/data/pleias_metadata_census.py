"""Census every PleIAs parent without retaining its source text."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-metadata-census-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-metadata-census-aggregate-v1"
FILE_SCHEMA = "sai-pleias-metadata-census-file-v1"
SEGMENT_SCHEMA = "sai-pleias-metadata-census-segment-v1"
RECOVERY_SCHEMA = "sai-pleias-metadata-census-segment-recovery-v1"
SOURCE_ID = "pleias_common_corpus"
AXES = (
    "collection",
    "language",
    "open_type",
    "license",
    "collection_language",
    "collection_license",
)
REQUIRED_COLUMNS = frozenset(
    {
        "identifier",
        "collection",
        "open_type",
        "license",
        "language",
        "word_count",
        "token_count",
        "text",
    }
)


class PleiasMetadataCensusError(RuntimeError):
    """The manifest, Parquet identity, or census accounting differs."""


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    stage = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with stage.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the exact PleIAs rows from the materialized-lake manifest."""

    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasMetadataCensusError("PleIAs manifest is missing or unsafe")
    selected = []
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("source_id") != SOURCE_ID:
                    continue
                if (
                    not isinstance(row.get("source_path"), str)
                    or not row["source_path"]
                    or not isinstance(row.get("source_repository"), str)
                    or not row["source_repository"]
                    or not isinstance(row.get("source_revision"), str)
                    or not row["source_revision"]
                    or isinstance(row.get("bytes"), bool)
                    or not isinstance(row.get("bytes"), int)
                    or row["bytes"] <= 0
                    or not isinstance(row.get("sha256"), str)
                    or len(row["sha256"]) != 64
                    or row.get("raw_source_is_training_ready") is not False
                ):
                    raise PleiasMetadataCensusError("PleIAs manifest row differs")
                selected.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasMetadataCensusError("PleIAs manifest is invalid") from error
    selected.sort(key=lambda row: row["source_path"])
    paths = [row["source_path"] for row in selected]
    if not selected or len(paths) != len(set(paths)):
        raise PleiasMetadataCensusError("PleIAs manifest coverage differs")
    return selected


def select_shard(
    rows: list[dict[str, Any]], logical_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    """Select an exact path-ordered, identity-disjoint shard."""

    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 512
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasMetadataCensusError("PleIAs census geometry differs")
    return [
        row for index, row in enumerate(rows) if index % logical_shards == shard_index
    ]


def select_segment(
    rows: list[dict[str, Any]],
    logical_shards: int,
    shard_index: int,
    segments_per_shard: int,
    segment_index: int,
) -> list[dict[str, Any]]:
    """Select one identity-disjoint recovery segment inside a logical shard."""

    if (
        isinstance(segments_per_shard, bool)
        or not isinstance(segments_per_shard, int)
        or not 2 <= segments_per_shard <= 64
        or isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or not 0 <= segment_index < segments_per_shard
    ):
        raise PleiasMetadataCensusError("PleIAs census segment geometry differs")
    shard = select_shard(rows, logical_shards, shard_index)
    return [
        row
        for index, row in enumerate(shard)
        if index % segments_per_shard == segment_index
    ]


def _string(value: Any) -> str:
    if value is None:
        return "__null__"
    if not isinstance(value, str):
        raise PleiasMetadataCensusError("PleIAs metadata value differs")
    value = value.strip()
    return value if value else "__empty__"


def _axis_key(axis: str, collection: str, language: str, license_name: str) -> str:
    if axis == "collection":
        return collection
    if axis == "language":
        return language
    if axis == "license":
        return license_name
    if axis == "collection_language":
        return json.dumps([collection, language], separators=(",", ":"))
    if axis == "collection_license":
        return json.dumps([collection, license_name], separators=(",", ":"))
    raise PleiasMetadataCensusError("PleIAs census axis differs")


def _column_null_count(source: Any, name: str) -> int | None:
    index = source.schema_arrow.names.index(name)
    total = 0
    for group_index in range(source.metadata.num_row_groups):
        statistics = source.metadata.row_group(group_index).column(index).statistics
        if statistics is None or statistics.null_count is None:
            return None
        total += statistics.null_count
    return total


def census_local_file(path: Path, manifest_row: dict[str, Any]) -> dict[str, Any]:
    """Hash-verify one parent and summarize non-text metadata exactly."""

    try:
        import pyarrow.compute as compute
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise PleiasMetadataCensusError("pyarrow is required") from error
    if (
        not path.is_file()
        or path.stat().st_size != manifest_row["bytes"]
        or sha256_file(path) != manifest_row["sha256"]
    ):
        raise PleiasMetadataCensusError("PleIAs parent identity differs")
    source = parquet.ParquetFile(path)
    available = set(source.schema_arrow.names)
    if not REQUIRED_COLUMNS.issubset(available) or source.metadata.num_rows <= 0:
        raise PleiasMetadataCensusError("PleIAs parent schema differs")
    columns = [
        "identifier",
        "collection",
        "open_type",
        "license",
        "language",
        "word_count",
        "token_count",
    ]
    table = source.read(columns=columns, use_threads=True)
    if table.num_rows != source.metadata.num_rows:
        raise PleiasMetadataCensusError("PleIAs parent row coverage differs")
    axis_counts: dict[str, dict[str, Counter[str]]] = {
        axis: defaultdict(Counter) for axis in AXES
    }
    collections = table["collection"].to_pylist()
    languages = table["language"].to_pylist()
    open_types = table["open_type"].to_pylist()
    licenses = table["license"].to_pylist()
    words = table["word_count"].to_pylist()
    tokens = table["token_count"].to_pylist()
    structural = Counter()
    for collection, language, open_type, license_name, word_count, token_count in zip(
        collections, languages, open_types, licenses, words, tokens, strict=True
    ):
        collection = _string(collection)
        language = _string(language)
        open_type = _string(open_type)
        license_name = _string(license_name)
        word_count = word_count if isinstance(word_count, int) else 0
        token_count = token_count if isinstance(token_count, int) else 0
        if word_count <= 0:
            structural["nonpositive_word_count"] += 1
        if token_count <= 0:
            structural["nonpositive_token_count"] += 1
        values = {
            "collection": collection,
            "language": language,
            "open_type": open_type,
            "license": license_name,
            "collection_language": _axis_key(
                "collection_language", collection, language, license_name
            ),
            "collection_license": _axis_key(
                "collection_license", collection, language, license_name
            ),
        }
        for axis, value in values.items():
            counts = axis_counts[axis][value]
            counts["rows"] += 1
            counts["word_count"] += max(0, word_count)
            counts["token_count"] += max(0, token_count)
    for name in ("identifier", "text"):
        null_count = _column_null_count(source, name)
        if null_count is not None:
            structural[f"{name}_null"] = null_count
    structural["identifier_array_null"] = table["identifier"].null_count
    word_sum = compute.sum(table["word_count"]).as_py()
    token_sum = compute.sum(table["token_count"]).as_py()
    payload = {
        "schema": FILE_SCHEMA,
        "source_path": manifest_row["source_path"],
        "source_repository": manifest_row["source_repository"],
        "source_revision": manifest_row["source_revision"],
        "source_sha256": manifest_row["sha256"],
        "physical_bytes": manifest_row["bytes"],
        "rows": table.num_rows,
        "word_count": word_sum if isinstance(word_sum, int) else 0,
        "token_count": token_sum if isinstance(token_sum, int) else 0,
        "structural_counts": dict(sorted(structural.items())),
        "axes": {
            axis: {
                value: dict(sorted(counts.items()))
                for value, counts in sorted(values.items())
            }
            for axis, values in axis_counts.items()
        },
        "source_text_read": False,
        "source_text_persisted": False,
        "training_ready": False,
    }
    payload["row_sha256"] = canonical_sha256(payload)
    return payload


def _download_and_census(
    row: dict[str, Any], token: str, scratch_root: Path | None
) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise PleiasMetadataCensusError("huggingface_hub is required") from error
    parent = str(scratch_root) if scratch_root is not None else None
    with tempfile.TemporaryDirectory(prefix="sai-pleias-census-", dir=parent) as tmp:
        temporary = Path(tmp)
        downloaded = Path(
            hf_hub_download(
                repo_id=row["source_repository"],
                filename=row["source_path"],
                repo_type="dataset",
                revision=row["source_revision"],
                token=token,
                cache_dir=temporary / "cache",
                local_dir=temporary / "local",
            )
        )
        return census_local_file(downloaded, row)


def _merge_axes(
    destination: dict[str, dict[str, Counter[str]]],
    axes: dict[str, Any],
) -> None:
    if set(axes) != set(AXES):
        raise PleiasMetadataCensusError("PleIAs census axes differ")
    for axis, values in axes.items():
        if not isinstance(values, dict):
            raise PleiasMetadataCensusError("PleIAs census axis differs")
        for value, counts in values.items():
            if not isinstance(value, str) or not isinstance(counts, dict):
                raise PleiasMetadataCensusError("PleIAs census counts differ")
            destination[axis][value].update(counts)
            destination[axis][value]["files"] += 1


def _census_selected(
    selected: list[dict[str, Any]],
    token: str,
    scratch_root: Path | None,
    progress: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], Counter[str]]:
    """Census one exact parent subset and return source-text-free evidence."""

    if not selected:
        raise PleiasMetadataCensusError("PleIAs census selection is empty")
    file_rows = []
    axes: dict[str, dict[str, Counter[str]]] = {
        axis: defaultdict(Counter) for axis in AXES
    }
    totals = Counter()
    for index, row in enumerate(selected, start=1):
        result = _download_and_census(row, token, scratch_root)
        file_rows.append(result)
        totals["files"] += 1
        for field in ("physical_bytes", "rows", "word_count", "token_count"):
            totals[field] += result[field]
        _merge_axes(axes, result["axes"])
        print(
            json.dumps(
                {
                    "event": progress["event"],
                    **{key: value for key, value in progress.items() if key != "event"},
                    "complete": index,
                    "remaining": len(selected) - index,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    serialized_axes = {
        axis: {
            value: dict(sorted(counts.items()))
            for value, counts in sorted(values.items())
        }
        for axis, values in axes.items()
    }
    return file_rows, serialized_axes, totals


def run_shard(
    manifest_path: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Download and census one exact manifest shard."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise PleiasMetadataCensusError("PleIAs census output differs")
    all_rows = load_manifest(manifest_path)
    selected = select_shard(all_rows, logical_shards, shard_index)
    if not selected:
        raise PleiasMetadataCensusError("PleIAs census shard is empty")
    output_root.mkdir(parents=True)
    file_rows, axes, totals = _census_selected(
        selected,
        token,
        scratch_root,
        {
            "event": "pleias_metadata_census_progress",
            "logical_shards": logical_shards,
            "shard_index": shard_index,
        },
    )
    file_path = output_root / "files.jsonl"
    _atomic_jsonl(file_path, file_rows)
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_metadata_census_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source_manifest": {
            "path_name": manifest_path.name,
            "file_sha256": sha256_file(manifest_path),
            "pleias_files": len(all_rows),
        },
        "selected_paths_sha256": canonical_sha256(
            [row["source_path"] for row in selected]
        ),
        "file_rows": {
            "path": file_path.name,
            "rows": len(file_rows),
            "sha256": sha256_file(file_path),
            "ordered_row_sha256": canonical_sha256(
                [row["row_sha256"] for row in file_rows]
            ),
        },
        "totals": dict(sorted(totals.items())),
        "axes": axes,
        "source_text_read": False,
        "source_text_persisted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def run_segment(
    manifest_path: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    segments_per_shard: int,
    segment_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Census one bounded recovery segment without touching a healthy shard."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise PleiasMetadataCensusError("PleIAs census segment output differs")
    all_rows = load_manifest(manifest_path)
    parent_shard = select_shard(all_rows, logical_shards, shard_index)
    selected = select_segment(
        all_rows,
        logical_shards,
        shard_index,
        segments_per_shard,
        segment_index,
    )
    if not selected:
        raise PleiasMetadataCensusError("PleIAs census segment is empty")
    output_root.mkdir(parents=True)
    file_rows, axes, totals = _census_selected(
        selected,
        token,
        scratch_root,
        {
            "event": "pleias_metadata_census_segment_progress",
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "segments_per_shard": segments_per_shard,
            "segment_index": segment_index,
        },
    )
    file_path = output_root / "files.jsonl"
    _atomic_jsonl(file_path, file_rows)
    payload = {
        "schema": SEGMENT_SCHEMA,
        "status": "complete_nontraining_pleias_metadata_census_segment",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "segments_per_shard": segments_per_shard,
        "segment_index": segment_index,
        "source_manifest": {
            "path_name": manifest_path.name,
            "file_sha256": sha256_file(manifest_path),
            "pleias_files": len(all_rows),
        },
        "parent_shard_selected_paths_sha256": canonical_sha256(
            [row["source_path"] for row in parent_shard]
        ),
        "selected_paths_sha256": canonical_sha256(
            [row["source_path"] for row in selected]
        ),
        "file_rows": {
            "path": file_path.name,
            "rows": len(file_rows),
            "sha256": sha256_file(file_path),
            "ordered_row_sha256": canonical_sha256(
                [row["row_sha256"] for row in file_rows]
            ),
        },
        "totals": dict(sorted(totals.items())),
        "axes": axes,
        "source_text_read": False,
        "source_text_persisted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasMetadataCensusError("PleIAs census receipt is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasMetadataCensusError("PleIAs census receipt is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise PleiasMetadataCensusError("PleIAs census receipt differs")
    return payload


def _validated_file_rows(
    root: Path,
    receipt: dict[str, Any],
    expected_paths: set[str],
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    descriptor = receipt.get("file_rows")
    file_path = (
        root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    )
    if (
        not isinstance(descriptor, dict)
        or not file_path.is_file()
        or file_path.is_symlink()
        or file_path.stat().st_nlink != 1
        or descriptor.get("sha256") != sha256_file(file_path)
    ):
        raise PleiasMetadataCensusError("PleIAs file rows differ")
    try:
        with file_path.open() as handle:
            rows = [json.loads(line) for line in handle]
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasMetadataCensusError("PleIAs file census differs") from error
    if len(rows) != descriptor.get("rows") or canonical_sha256(
        [row.get("row_sha256") for row in rows]
    ) != descriptor.get("ordered_row_sha256"):
        raise PleiasMetadataCensusError("PleIAs file rows differ")
    totals = Counter()
    axes: dict[str, dict[str, Counter[str]]] = {
        axis: defaultdict(Counter) for axis in AXES
    }
    seen = set()
    for row in rows:
        unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
        path = row.get("source_path")
        if (
            row.get("schema") != FILE_SCHEMA
            or row.get("row_sha256") != canonical_sha256(unsigned)
            or path not in expected_paths
            or path in seen
            or row.get("source_text_read") is not False
            or row.get("source_text_persisted") is not False
        ):
            raise PleiasMetadataCensusError("PleIAs file receipt differs")
        seen.add(path)
        totals["files"] += 1
        for field in ("physical_bytes", "rows", "word_count", "token_count"):
            totals[field] += row[field]
        _merge_axes(axes, row["axes"])
    serialized_axes = {
        axis: {
            value: dict(sorted(counts.items()))
            for value, counts in sorted(values.items())
        }
        for axis, values in axes.items()
    }
    if dict(sorted(totals.items())) != receipt.get(
        "totals"
    ) or serialized_axes != receipt.get("axes"):
        raise PleiasMetadataCensusError("PleIAs census unit totals differ")
    return rows, totals, serialized_axes


def merge_segments(
    manifest_path: Path,
    segments_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    segments_per_shard: int,
) -> dict[str, Any]:
    """Merge complete recovery segments into the canonical whole-shard receipt."""

    if output_root.exists() or output_root.is_symlink():
        raise PleiasMetadataCensusError("PleIAs recovered shard output differs")
    all_rows = load_manifest(manifest_path)
    selected = select_shard(all_rows, logical_shards, shard_index)
    if not selected:
        raise PleiasMetadataCensusError("PleIAs recovered shard is empty")
    expected_order = [row["source_path"] for row in selected]
    expected_paths = set(expected_order)
    rows_by_path = {}
    segment_receipts = []
    for segment_index in range(segments_per_shard):
        root = segments_root / f"segment_{segment_index:05d}"
        receipt = _load_signed(root / "receipt.json", SEGMENT_SCHEMA)
        segment_selected = select_segment(
            all_rows,
            logical_shards,
            shard_index,
            segments_per_shard,
            segment_index,
        )
        segment_paths = [row["source_path"] for row in segment_selected]
        if (
            receipt.get("status")
            != "complete_nontraining_pleias_metadata_census_segment"
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("segments_per_shard") != segments_per_shard
            or receipt.get("segment_index") != segment_index
            or receipt.get("source_manifest", {}).get("file_sha256")
            != sha256_file(manifest_path)
            or receipt.get("parent_shard_selected_paths_sha256")
            != canonical_sha256(expected_order)
            or receipt.get("selected_paths_sha256") != canonical_sha256(segment_paths)
        ):
            raise PleiasMetadataCensusError("PleIAs census segment differs")
        rows, _totals, _axes = _validated_file_rows(root, receipt, set(segment_paths))
        for row in rows:
            path = row["source_path"]
            if path in rows_by_path:
                raise PleiasMetadataCensusError("PleIAs census segments overlap")
            rows_by_path[path] = row
        segment_receipts.append(receipt["receipt_sha256"])
    if set(rows_by_path) != expected_paths:
        raise PleiasMetadataCensusError("PleIAs census segment coverage differs")
    file_rows = [rows_by_path[path] for path in expected_order]
    axes: dict[str, dict[str, Counter[str]]] = {
        axis: defaultdict(Counter) for axis in AXES
    }
    totals = Counter()
    for row in file_rows:
        totals["files"] += 1
        for field in ("physical_bytes", "rows", "word_count", "token_count"):
            totals[field] += row[field]
        _merge_axes(axes, row["axes"])
    serialized_axes = {
        axis: {
            value: dict(sorted(counts.items()))
            for value, counts in sorted(values.items())
        }
        for axis, values in axes.items()
    }
    output_root.mkdir(parents=True)
    try:
        file_path = output_root / "files.jsonl"
        _atomic_jsonl(file_path, file_rows)
        payload = {
            "schema": SHARD_SCHEMA,
            "status": "complete_nontraining_pleias_metadata_census_shard",
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "source_manifest": {
                "path_name": manifest_path.name,
                "file_sha256": sha256_file(manifest_path),
                "pleias_files": len(all_rows),
            },
            "selected_paths_sha256": canonical_sha256(expected_order),
            "file_rows": {
                "path": file_path.name,
                "rows": len(file_rows),
                "sha256": sha256_file(file_path),
                "ordered_row_sha256": canonical_sha256(
                    [row["row_sha256"] for row in file_rows]
                ),
            },
            "totals": dict(sorted(totals.items())),
            "axes": serialized_axes,
            "source_text_read": False,
            "source_text_persisted": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "status": "complete_nontraining_segment_recovery",
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "segments_per_shard": segments_per_shard,
            "ordered_segment_receipts_sha256": canonical_sha256(segment_receipts),
            "canonical_shard_receipt_sha256": payload["receipt_sha256"],
            "source_text_persisted": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        recovery["receipt_sha256"] = canonical_sha256(recovery)
        _atomic_create(output_root / "recovery.json", recovery)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def aggregate_shards(
    manifest_path: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify complete parent coverage and combine source-safe shard counters."""

    if output.exists() or output.is_symlink():
        raise PleiasMetadataCensusError("PleIAs census aggregate exists")
    manifest_rows = load_manifest(manifest_path)
    expected_paths = {row["source_path"] for row in manifest_rows}
    seen_paths = set()
    row_hashes = []
    receipt_hashes = []
    axes: dict[str, dict[str, Counter[str]]] = {
        axis: defaultdict(Counter) for axis in AXES
    }
    totals = Counter()
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        file_path = root / "files.jsonl"
        if (
            receipt.get("status") != "complete_nontraining_pleias_metadata_census_shard"
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source_manifest", {}).get("file_sha256")
            != sha256_file(manifest_path)
            or receipt.get("file_rows", {}).get("sha256") != sha256_file(file_path)
        ):
            raise PleiasMetadataCensusError("PleIAs census shard differs")
        rows = []
        try:
            with file_path.open() as handle:
                rows = [json.loads(line) for line in handle]
        except (OSError, json.JSONDecodeError) as error:
            raise PleiasMetadataCensusError("PleIAs file census differs") from error
        if (
            len(rows) != receipt["file_rows"]["rows"]
            or canonical_sha256([row.get("row_sha256") for row in rows])
            != receipt["file_rows"]["ordered_row_sha256"]
        ):
            raise PleiasMetadataCensusError("PleIAs file rows differ")
        shard_totals = Counter()
        for row in rows:
            unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
            path = row.get("source_path")
            if (
                row.get("schema") != FILE_SCHEMA
                or row.get("row_sha256") != canonical_sha256(unsigned)
                or path not in expected_paths
                or path in seen_paths
                or row.get("source_text_read") is not False
                or row.get("source_text_persisted") is not False
            ):
                raise PleiasMetadataCensusError("PleIAs file receipt differs")
            seen_paths.add(path)
            row_hashes.append(row["row_sha256"])
            for field in ("physical_bytes", "rows", "word_count", "token_count"):
                shard_totals[field] += row[field]
                totals[field] += row[field]
            shard_totals["files"] += 1
            totals["files"] += 1
            _merge_axes(axes, row["axes"])
        if dict(sorted(shard_totals.items())) != receipt.get("totals"):
            raise PleiasMetadataCensusError("PleIAs shard totals differ")
        receipt_hashes.append(receipt["receipt_sha256"])
    if seen_paths != expected_paths:
        raise PleiasMetadataCensusError("PleIAs complete coverage differs")
    expected_bytes = sum(row["bytes"] for row in manifest_rows)
    if (
        totals["files"] != len(manifest_rows)
        or totals["physical_bytes"] != expected_bytes
    ):
        raise PleiasMetadataCensusError("PleIAs aggregate bytes differ")
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_metadata_census",
        "source_manifest": {
            "path_name": manifest_path.name,
            "file_sha256": sha256_file(manifest_path),
            "pleias_files": len(manifest_rows),
            "pleias_bytes": expected_bytes,
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipt_hashes),
        },
        "ordered_file_rows_sha256": canonical_sha256(row_hashes),
        "totals": dict(sorted(totals.items())),
        "axes": {
            axis: {
                value: dict(sorted(counts.items()))
                for value, counts in sorted(values.items())
            }
            for axis, values in axes.items()
        },
        "source_text_read": False,
        "source_text_persisted": False,
        "metadata_census_is_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    segment = subparsers.add_parser("segment")
    segment.add_argument("--manifest", type=Path, required=True)
    segment.add_argument("--output-root", type=Path, required=True)
    segment.add_argument("--logical-shards", type=int, required=True)
    segment.add_argument("--shard-index", type=int, required=True)
    segment.add_argument("--segments-per-shard", type=int, required=True)
    segment.add_argument("--segment-index", type=int, required=True)
    segment.add_argument("--token-env", default="HF_TOKEN")
    segment.add_argument("--scratch-root", type=Path)
    merge = subparsers.add_parser("merge-segments")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--segments-root", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    merge.add_argument("--logical-shards", type=int, required=True)
    merge.add_argument("--shard-index", type=int, required=True)
    merge.add_argument("--segments-per-shard", type=int, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--shards-root", type=Path, required=True)
    aggregate.add_argument("--logical-shards", type=int, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    elif args.command == "segment":
        result = run_segment(
            args.manifest,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            args.segments_per_shard,
            args.segment_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    elif args.command == "merge-segments":
        result = merge_segments(
            args.manifest,
            args.segments_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            args.segments_per_shard,
        )
    else:
        result = aggregate_shards(
            args.manifest, args.shards_root, args.logical_shards, args.output
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "totals": result["totals"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
