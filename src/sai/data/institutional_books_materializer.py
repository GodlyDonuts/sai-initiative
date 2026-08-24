"""Selectively materialize strict English Institutional Books on private storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books import ENRICHED_REPOSITORY, ENRICHED_REVISION
from sai.data.institutional_books_quality_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from sai.data.institutional_books_quality_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from sai.data.source_reservoir import MANIFEST_SCHEMA, RECEIPT_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

PARENT_SCHEMA = "sai-institutional-books-materialized-parent-v1"
LINEAGE_SCHEMA = "sai-institutional-books-materialized-lineage-v1"
SHARD_SCHEMA = "sai-institutional-books-materialized-shard-v1"
AGGREGATE_SCHEMA = "sai-institutional-books-materialized-aggregate-v1"
OUTPUT_SCHEMA = "sai-institutional-books-materialized-row-v1"
SOURCE_ID = "institutional_books"
SOURCE_COLUMNS = (
    "barcode_src",
    "primary_language_gen",
    "token_count_gen",
    "char_count_gen",
    "word_count_gen",
    "tokenizability_ratio_gen",
    "processed_middlematter_gen",
)


class InstitutionalBooksMaterializerError(RuntimeError):
    """Source, selection, output, or complete coverage differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise InstitutionalBooksMaterializerError("materializer input is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise InstitutionalBooksMaterializerError(
            "materializer input is invalid"
        ) from error
    if not isinstance(value, dict):
        raise InstitutionalBooksMaterializerError("materializer input differs")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise InstitutionalBooksMaterializerError("materializer rows are unsafe")
    try:
        with path.open() as handle:
            rows = [json.loads(line) for line in handle]
    except (OSError, json.JSONDecodeError) as error:
        raise InstitutionalBooksMaterializerError(
            "materializer rows are invalid"
        ) from error
    if any(not isinstance(row, dict) for row in rows):
        raise InstitutionalBooksMaterializerError("materializer rows differ")
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksMaterializerError("materializer output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _valid_receipt(payload: dict[str, Any], schema: str) -> bool:
    unsigned = {
        key: value for key, value in payload.items() if key != "receipt_sha256"
    }
    return bool(
        payload.get("schema") == schema
        and payload.get("receipt_sha256") == canonical_sha256(unsigned)
        and payload.get("training_ready") is False
    )


def _validate_parent_files(root: Path, parent: dict[str, Any]) -> None:
    lineage_path = root / parent["lineage"]["path"]
    if (
        lineage_path.stat().st_size != parent["lineage"]["bytes"]
        or sha256_file(lineage_path) != parent["lineage"]["sha256"]
    ):
        raise InstitutionalBooksMaterializerError("parent lineage differs")
    output = parent.get("output")
    if output is not None:
        output_path = root / output["path"]
        if (
            output_path.stat().st_size != output["bytes"]
            or sha256_file(output_path) != output["sha256"]
        ):
            raise InstitutionalBooksMaterializerError("parent output differs")


def load_source_manifest(
    manifest_path: Path, receipt_path: Path
) -> list[dict[str, Any]]:
    """Validate and return all 4,916 pinned enriched-text parents."""

    receipt = _load_json(receipt_path)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    rows = _load_jsonl(manifest_path)
    selected = [row for row in rows if row.get("source_id") == SOURCE_ID]
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path)
        or receipt.get("training_ready") is not False
        or len(selected) != 4_916
        or any(
            row.get("schema") != MANIFEST_SCHEMA
            or row.get("repository") != ENRICHED_REPOSITORY
            or row.get("revision") != ENRICHED_REVISION
            or row.get("raw_source_is_training_ready") is not False
            for row in selected
        )
    ):
        raise InstitutionalBooksMaterializerError("source reservoir differs")
    return selected


def load_selection(root: Path) -> dict[str, dict[str, Any]]:
    """Validate the strict selection and return compact barcode metadata."""

    receipt = _load_json(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    path = root / "selection.jsonl"
    rows = _load_jsonl(path)
    by_barcode = {}
    for row in rows:
        barcode = row.get("barcode_src")
        unsigned_row = {key: value for key, value in row.items() if key != "row_sha256"}
        if (
            row.get("schema") != SELECTION_ROW_SCHEMA
            or not isinstance(barcode, str)
            or not barcode
            or barcode in by_barcode
            or row.get("row_sha256") != canonical_sha256(unsigned_row)
            or row.get("language_gen") != "eng"
            or row.get("training_ready") is not False
        ):
            raise InstitutionalBooksMaterializerError("selection row differs")
        by_barcode[barcode] = {
            "row_sha256": row["row_sha256"],
            "tokens": row["token_count_o200k_base_gen"],
        }
    if (
        receipt.get("schema") != SELECTION_SCHEMA
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("selection", {}).get("rows") != len(rows)
        or receipt.get("selection", {}).get("sha256") != sha256_file(path)
        or receipt.get("selection", {}).get("tokens")
        != sum(row["token_count_o200k_base_gen"] for row in rows)
        or receipt.get("training_ready") is not False
    ):
        raise InstitutionalBooksMaterializerError("selection receipt differs")
    return by_barcode


def filter_source_rows(
    rows: list[dict[str, Any]],
    selection: dict[str, dict[str, Any]],
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter one parent while retaining exact text-free lineage."""

    outputs = []
    lineage = []
    seen = set()
    for row in rows:
        barcode = row.get("barcode_src")
        if barcode not in selection:
            continue
        if barcode in seen:
            raise InstitutionalBooksMaterializerError("source barcode is duplicated")
        seen.add(barcode)
        text = row.get("processed_middlematter_gen")
        tokens = row.get("token_count_gen")
        reason = None
        if row.get("primary_language_gen") != "eng":
            reason = "enriched_primary_language_mismatch"
        elif not isinstance(text, str) or len(text.encode("utf-8")) < 200:
            reason = "enriched_text_missing_or_short"
        elif isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 2_000:
            reason = "enriched_token_count_below_floor"
        elif tokens > 2_000_000:
            reason = "enriched_token_count_above_ceiling"
        content_sha256 = (
            hashlib.sha256(text.encode()).hexdigest()
            if isinstance(text, str)
            else None
        )
        lineage_row = {
            "schema": LINEAGE_SCHEMA,
            "barcode_src": barcode,
            "selection_row_sha256": selection[barcode]["row_sha256"],
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "disposition": "materialized" if reason is None else "excluded",
            "exclusion_reason": reason,
            "source_content_sha256": content_sha256,
            "source_text_persisted": False,
            "training_ready": False,
        }
        lineage_row["row_sha256"] = canonical_sha256(lineage_row)
        lineage.append(lineage_row)
        if reason is None:
            outputs.append(
                {
                    "schema": OUTPUT_SCHEMA,
                    "barcode_src": barcode,
                    "text": text,
                    "selection_row_sha256": selection[barcode]["row_sha256"],
                    "source_content_sha256": content_sha256,
                    "metadata_token_count_o200k_base_gen": selection[barcode][
                        "tokens"
                    ],
                    "enriched_token_count_gen": tokens,
                    "enriched_char_count_gen": row.get("char_count_gen"),
                    "enriched_word_count_gen": row.get("word_count_gen"),
                    "tokenizability_ratio_gen": row.get(
                        "tokenizability_ratio_gen"
                    ),
                    "source_path": source["path"],
                    "source_sha256": source["sha256"],
                    "training_ready": False,
                }
            )
    return outputs, lineage


def _download(source: dict[str, Any], token: str, scratch: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise InstitutionalBooksMaterializerError(
            "huggingface_hub is required"
        ) from error
    path = Path(
        hf_hub_download(
            repo_id=source["repository"],
            filename=source["path"],
            repo_type="dataset",
            revision=source["revision"],
            token=token,
            local_dir=scratch,
        )
    )
    if path.stat().st_size != source["bytes"] or sha256_file(path) != source["sha256"]:
        raise InstitutionalBooksMaterializerError("downloaded parent differs")
    return path


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksMaterializerError("pyarrow is required") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_shard(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    selection_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None,
) -> dict[str, Any]:
    """Materialize one resumable, source-parent-disjoint shard."""

    if not token or not 0 <= shard_index < logical_shards:
        raise InstitutionalBooksMaterializerError("materializer shard differs")
    sources = load_source_manifest(manifest_path, reservoir_receipt_path)
    assigned = [
        (index, source)
        for index, source in enumerate(sources)
        if index % logical_shards == shard_index
    ]
    selection = load_selection(selection_root)
    root = output_root / "shards" / f"shard_{shard_index:05d}"
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        existing = _load_json(receipt_path)
        if (
            not _valid_receipt(existing, SHARD_SCHEMA)
            or existing.get("logical_shards") != logical_shards
            or existing.get("shard_index") != shard_index
            or existing.get("assigned_parents") != len(assigned)
        ):
            raise InstitutionalBooksMaterializerError("existing shard differs")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    parent_receipts = []
    totals = Counter()
    for ordinal, source in assigned:
        parent_receipt_path = root / "parents" / f"parent_{ordinal:05d}.json"
        if parent_receipt_path.exists():
            parent = _load_json(parent_receipt_path)
            if (
                not _valid_receipt(parent, PARENT_SCHEMA)
                or parent.get("ordinal") != ordinal
                or parent.get("source", {}).get("path") != source["path"]
                or parent.get("source", {}).get("sha256") != source["sha256"]
            ):
                raise InstitutionalBooksMaterializerError("parent receipt differs")
            _validate_parent_files(root, parent)
            parent_receipts.append(parent)
            totals.update(parent["counts"])
            continue
        with tempfile.TemporaryDirectory(dir=scratch_root) as temporary:
            local_path = _download(source, token, Path(temporary))
            try:
                import pyarrow.parquet as pq
            except ImportError as error:
                raise InstitutionalBooksMaterializerError(
                    "pyarrow is required"
                ) from error
            source_rows = pq.read_table(
                local_path, columns=list(SOURCE_COLUMNS)
            ).to_pylist()
            outputs, lineage = filter_source_rows(source_rows, selection, source)
        lineage_path = root / "lineage" / f"parent_{ordinal:05d}.jsonl"
        _atomic_jsonl(lineage_path, lineage)
        output_path = root / "data" / f"parent_{ordinal:05d}.parquet"
        if outputs:
            _write_parquet(output_path, outputs)
        counts = Counter(
            {
                "source_rows": len(source_rows),
                "selected_rows": len(lineage),
                "materialized_rows": len(outputs),
                "excluded_rows": len(lineage) - len(outputs),
                "metadata_tokens": sum(
                    selection[row["barcode_src"]]["tokens"] for row in lineage
                ),
                "materialized_metadata_tokens": sum(
                    row["metadata_token_count_o200k_base_gen"] for row in outputs
                ),
                "materialized_enriched_tokens": sum(
                    row["enriched_token_count_gen"] for row in outputs
                ),
            }
        )
        parent = {
            "schema": PARENT_SCHEMA,
            "ordinal": ordinal,
            "source": {
                "path": source["path"],
                "bytes": source["bytes"],
                "sha256": source["sha256"],
            },
            "counts": dict(sorted(counts.items())),
            "lineage": {
                "path": str(lineage_path.relative_to(root)),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(
                    [row["row_sha256"] for row in lineage]
                ),
            },
            "output": (
                {
                    "path": str(output_path.relative_to(root)),
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                    "rows": len(outputs),
                }
                if outputs
                else None
            ),
            "source_text_persisted_in_private_output": bool(outputs),
            "training_ready": False,
        }
        parent["receipt_sha256"] = canonical_sha256(parent)
        _atomic_create(parent_receipt_path, parent)
        parent_receipts.append(parent)
        totals.update(counts)
        print(
            json.dumps(
                {
                    "event": "institutional_books_materializer_progress",
                    "shard_index": shard_index,
                    "complete": len(parent_receipts),
                    "remaining": len(assigned) - len(parent_receipts),
                    "materialized_rows": totals["materialized_rows"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_private_book_materialization_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "assigned_parents": len(assigned),
        "ordered_parent_receipts_sha256": canonical_sha256(
            [row["receipt_sha256"] for row in parent_receipts]
        ),
        "counts": dict(sorted(totals.items())),
        "source_text_persisted_in_private_output": totals["materialized_rows"] > 0,
        "huggingface_redistribution_authorized": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(receipt_path, payload)
    return payload


def aggregate(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    selection_root: Path,
    output_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify complete source and selected-barcode coverage without publishing text."""

    if output.exists() or output.is_symlink():
        raise InstitutionalBooksMaterializerError("aggregate output exists")
    sources = load_source_manifest(manifest_path, reservoir_receipt_path)
    sources_by_path = {row["path"]: row for row in sources}
    selection = load_selection(selection_root)
    seen_sources = set()
    seen_barcodes = set()
    totals = Counter()
    shard_hashes = []
    output_bytes = 0
    output_files = 0
    for shard_index in range(logical_shards):
        root = output_root / "shards" / f"shard_{shard_index:05d}"
        shard = _load_json(root / "receipt.json")
        if (
            not _valid_receipt(shard, SHARD_SCHEMA)
            or shard.get("logical_shards") != logical_shards
            or shard.get("shard_index") != shard_index
        ):
            raise InstitutionalBooksMaterializerError("shard receipt differs")
        shard_hashes.append(shard["receipt_sha256"])
        shard_totals = Counter()
        parent_hashes = []
        parent_count = 0
        for parent_path in sorted((root / "parents").glob("parent_*.json")):
            parent = _load_json(parent_path)
            source_path = parent.get("source", {}).get("path")
            source = sources_by_path.get(source_path)
            if (
                not _valid_receipt(parent, PARENT_SCHEMA)
                or source is None
                or parent.get("source", {}).get("bytes") != source["bytes"]
                or parent.get("source", {}).get("sha256") != source["sha256"]
                or source_path in seen_sources
            ):
                raise InstitutionalBooksMaterializerError("parent source overlaps")
            seen_sources.add(source_path)
            parent_count += 1
            parent_hashes.append(parent["receipt_sha256"])
            lineage_path = root / parent["lineage"]["path"]
            if sha256_file(lineage_path) != parent["lineage"]["sha256"]:
                raise InstitutionalBooksMaterializerError("lineage hash differs")
            for row in _load_jsonl(lineage_path):
                barcode = row.get("barcode_src")
                unsigned_row = {
                    key: value for key, value in row.items() if key != "row_sha256"
                }
                if (
                    row.get("schema") != LINEAGE_SCHEMA
                    or row.get("row_sha256") != canonical_sha256(unsigned_row)
                    or row.get("source_path") != source_path
                    or barcode not in selection
                    or barcode in seen_barcodes
                ):
                    raise InstitutionalBooksMaterializerError(
                        "selected barcode coverage differs"
                    )
                seen_barcodes.add(barcode)
            output_descriptor = parent.get("output")
            if output_descriptor is not None:
                output_path = root / output_descriptor["path"]
                if (
                    output_path.stat().st_size != output_descriptor["bytes"]
                    or sha256_file(output_path) != output_descriptor["sha256"]
                ):
                    raise InstitutionalBooksMaterializerError("output hash differs")
                output_bytes += output_descriptor["bytes"]
                output_files += 1
            totals.update(parent["counts"])
            shard_totals.update(parent["counts"])
        if (
            parent_count != shard.get("assigned_parents")
            or dict(sorted(shard_totals.items())) != shard.get("counts")
            or canonical_sha256(parent_hashes)
            != shard.get("ordered_parent_receipts_sha256")
        ):
            raise InstitutionalBooksMaterializerError("shard accounting differs")
    if seen_sources != {row["path"] for row in sources} or seen_barcodes != set(
        selection
    ):
        raise InstitutionalBooksMaterializerError("complete coverage differs")
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_private_book_materialization",
        "source": {
            "repository": ENRICHED_REPOSITORY,
            "revision": ENRICHED_REVISION,
            "parents": len(sources),
            "bytes": sum(row["bytes"] for row in sources),
        },
        "selection": {
            "receipt_sha256": _load_json(selection_root / "receipt.json")[
                "receipt_sha256"
            ],
            "rows": len(selection),
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(shard_hashes),
        },
        "counts": dict(sorted(totals.items())),
        "private_outputs": {"files": output_files, "bytes": output_bytes},
        "all_selected_barcodes_accounted": True,
        "benchmark_decontamination_complete": False,
        "semantic_deduplication_complete": False,
        "source_text_persisted_in_private_output": True,
        "huggingface_redistribution_authorized": False,
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
    aggregate_parser = subparsers.add_parser("aggregate")
    for child in (shard, aggregate_parser):
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--reservoir-receipt", type=Path, required=True)
        child.add_argument("--selection-root", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.reservoir_receipt,
            args.selection_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate(
            args.manifest,
            args.reservoir_receipt,
            args.selection_root,
            args.output_root,
            args.logical_shards,
            args.output,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
