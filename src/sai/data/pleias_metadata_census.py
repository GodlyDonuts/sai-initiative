"""Census every PleIAs parent without retaining its source text."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-metadata-census-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-metadata-census-aggregate-v1"
FILE_SCHEMA = "sai-pleias-metadata-census-file-v1"
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
                    raise PleiasMetadataCensusError(
                        "PleIAs manifest row differs"
                    )
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
        row
        for index, row in enumerate(rows)
        if index % logical_shards == shard_index
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
                    "event": "pleias_metadata_census_progress",
                    "logical_shards": logical_shards,
                    "shard_index": shard_index,
                    "complete": index,
                    "remaining": len(selected) - index,
                },
                sort_keys=True,
            ),
            flush=True,
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
        "axes": {
            axis: {
                value: dict(sorted(counts.items()))
                for value, counts in sorted(values.items())
            }
            for axis, values in axes.items()
        },
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
            receipt.get("status")
            != "complete_nontraining_pleias_metadata_census_shard"
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
