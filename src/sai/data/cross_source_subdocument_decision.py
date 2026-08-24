"""Decide exact subdocument deletions across final books and PleIAs content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from sai.data.agent_labeling import _atomic_create
from sai.data.frequency_length_subdocument_deduplication import (
    _GROUP,
    DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR,
    DEFAULT_EFFECTIVE_SHARDS_NUMERATOR,
    DEFAULT_REFERENCE_CHARACTERS,
    _merged_records,
    _records,
    _reduce_runs,
    _write_records,
    retention_budget,
)
from sai.data.institutional_books_subdocument_signature import (
    AGGREGATE_SCHEMA as BOOK_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_subdocument_signature import (
    SHARD_SCHEMA as BOOK_SHARD_SCHEMA,
)
from sai.data.pleias_final_subdocument_signature import (
    AGGREGATE_SCHEMA as PLEIAS_AGGREGATE_SCHEMA,
)
from sai.data.pleias_final_subdocument_signature import (
    SHARD_SCHEMA as PLEIAS_SHARD_SCHEMA,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_subdocument_signature import HASH_BUCKETS, SIGNATURE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-cross-source-subdocument-decision-v1"
# normalized hash, source priority, document identity, shard, row, chunk, spans,
# normalized characters, code flag
_SIGNATURE = struct.Struct(">32sB32sHQIQQQB")
# source priority, shard, row, document, chunk, spans, normalized hash,
# frequency, retention budget
DELETE_RECORD = struct.Struct(">BHQ32sIQQ32sQQ")
DEFAULT_CHUNK_RECORDS = 500_000
DEFAULT_MAXIMUM_OPEN_RUNS = 128
DEFAULT_MAXIMUM_OPEN_DELETION_FILES = 64


class CrossSourceSubdocumentDecisionError(RuntimeError):
    """Component custody, signature join, or deletion partition differs."""


@dataclass(frozen=True)
class Component:
    name: str
    priority: int
    root: Path
    logical_shards: int
    shard_schema: str
    aggregate_schema: str
    aggregate_completion_key: str


def _components(
    book_root: Path,
    pleias_root: Path,
    book_logical_shards: int,
    pleias_logical_shards: int,
) -> tuple[Component, Component]:
    if (
        book_logical_shards <= 0
        or pleias_logical_shards <= 0
        or book_root.resolve() == pleias_root.resolve()
    ):
        raise CrossSourceSubdocumentDecisionError("component geometry differs")
    return (
        Component(
            "institutional_books",
            0,
            book_root,
            book_logical_shards,
            BOOK_SHARD_SCHEMA,
            BOOK_AGGREGATE_SCHEMA,
            "complete_benchmark_disjoint_book_coverage",
        ),
        Component(
            "pleias_common_corpus",
            1,
            pleias_root,
            pleias_logical_shards,
            PLEIAS_SHARD_SCHEMA,
            PLEIAS_AGGREGATE_SCHEMA,
            "complete_final_pleias_document_coverage",
        ),
    )


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_signature(
    row: dict[str, Any], component: Component, shard_index: int, bucket_index: int
) -> None:
    unsigned = {key: value for key, value in row.items() if key != "signature_sha256"}
    if (
        row.get("schema") != SIGNATURE_SCHEMA
        or row.get("training_ready") is not False
        or row.get("component") != component.name
        or row.get("source_shard") != shard_index
        or isinstance(row.get("source_row_index"), bool)
        or not isinstance(row.get("source_row_index"), int)
        or row["source_row_index"] < 0
        or not _valid_hash(row.get("document_identity_sha256"))
        or not _valid_hash(row.get("normalized_sha256"))
        or int(row["normalized_sha256"][0], 16) != bucket_index
        or isinstance(row.get("chunk_index"), bool)
        or not isinstance(row.get("chunk_index"), int)
        or row["chunk_index"] < 0
        or not isinstance(row.get("character_start"), int)
        or not isinstance(row.get("character_end"), int)
        or not 0 <= row["character_start"] < row["character_end"]
        or not isinstance(row.get("normalized_characters"), int)
        or row["normalized_characters"] <= 0
        or not isinstance(row.get("code"), bool)
        or row.get("signature_sha256") != canonical_sha256(unsigned)
    ):
        raise CrossSourceSubdocumentDecisionError("signature row differs")


def _record(row: dict[str, Any], component: Component) -> tuple[Any, ...]:
    return (
        bytes.fromhex(row["normalized_sha256"]),
        component.priority,
        bytes.fromhex(row["document_identity_sha256"]),
        row["source_shard"],
        row["source_row_index"],
        row["chunk_index"],
        row["character_start"],
        row["character_end"],
        row["normalized_characters"],
        row["code"],
    )


def _build_runs(
    components: tuple[Component, ...],
    bucket_index: int,
    scratch: Path,
    chunk_records: int,
) -> tuple[list[Path], Counter[str], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise CrossSourceSubdocumentDecisionError("pyarrow is required") from error
    records = []
    runs = []
    counts: Counter[str] = Counter()
    bindings = []
    for component in components:
        aggregate = _load_signed(
            component.root / "aggregate.json", component.aggregate_schema
        )
        if (
            aggregate.get("shards", {}).get("logical_shards")
            != component.logical_shards
            or aggregate.get(component.aggregate_completion_key) is not True
            or aggregate.get("source_text_persisted") is not False
        ):
            raise CrossSourceSubdocumentDecisionError(
                "component signature aggregate differs"
            )
        expected = aggregate.get("totals", {}).get(
            f"bucket_{bucket_index:02x}_signatures", 0
        )
        observed = 0
        receipts = []
        for shard_index in range(component.logical_shards):
            root = component.root / "shards" / f"shard_{shard_index:05d}"
            receipt = _load_signed(root / "receipt.json", component.shard_schema)
            outputs = receipt.get("outputs")
            descriptor = (
                outputs[bucket_index]
                if isinstance(outputs, list) and len(outputs) == HASH_BUCKETS
                else {}
            )
            path = root / descriptor.get("path", "")
            if (
                receipt.get("logical_shards") != component.logical_shards
                or receipt.get("shard_index") != shard_index
                or descriptor.get("bucket") != bucket_index
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise CrossSourceSubdocumentDecisionError(
                    "component signature shard differs"
                )
            ordered = hashlib.sha256()
            rows = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
                for row in batch.to_pylist():
                    _validate_signature(row, component, shard_index, bucket_index)
                    ordered.update(bytes.fromhex(row["signature_sha256"]))
                    records.append(_record(row, component))
                    rows += 1
                    observed += 1
                    counts["signatures"] += 1
                    counts[f"component::{component.name}::signatures"] += 1
                    if len(records) >= chunk_records:
                        run = scratch / f"signature-{len(runs):08d}.bin"
                        runs.append(_write_records(records, _SIGNATURE, run))
                        records = []
            if (
                rows != descriptor.get("rows")
                or ordered.hexdigest()
                != descriptor.get("ordered_signature_digests_sha256")
            ):
                raise CrossSourceSubdocumentDecisionError(
                    "component signature coverage differs"
                )
            receipts.append(receipt["receipt_sha256"])
        if observed != expected:
            raise CrossSourceSubdocumentDecisionError(
                "component aggregate signature count differs"
            )
        bindings.append(
            {
                "component": component.name,
                "priority": component.priority,
                "logical_shards": component.logical_shards,
                "aggregate_receipt_sha256": aggregate["receipt_sha256"],
                "ordered_shard_receipts_sha256": canonical_sha256(receipts),
                "bucket_signatures": observed,
            }
        )
    if records:
        run = scratch / f"signature-{len(runs):08d}.bin"
        runs.append(_write_records(records, _SIGNATURE, run))
    if not runs:
        raise CrossSourceSubdocumentDecisionError("signature population is empty")
    return runs, counts, bindings


def _build_groups(
    runs: list[Path],
    path: Path,
    reference_characters: int,
    numerator: int,
    denominator: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    active_hash = None
    active_length = 0
    active_code = False
    frequency = 0

    def flush(handle: BinaryIO) -> None:
        if active_hash is None:
            return
        budget = retention_budget(
            frequency,
            active_length,
            reference_characters=reference_characters,
            effective_shards_numerator=numerator,
            effective_shards_denominator=denominator,
        )
        handle.write(_GROUP.pack(active_hash, frequency, active_length, budget))
        counts["groups"] += 1
        counts["duplicate_groups"] += frequency > 1
        counts["duplicate_occurrences"] += max(0, frequency - 1)

    with path.open("xb") as handle:
        for record in _merged_records(runs, _SIGNATURE):
            if record[0] != active_hash:
                flush(handle)
                active_hash = record[0]
                active_length = record[8]
                active_code = record[9]
                frequency = 1
            else:
                if record[8] != active_length or record[9] != active_code:
                    raise CrossSourceSubdocumentDecisionError(
                        "normalized signature collision differs"
                    )
                frequency += 1
        flush(handle)
    return counts


class _DeletionPool:
    def __init__(self, paths: dict[tuple[int, int], Path], maximum_open: int):
        self.paths = paths
        self.maximum_open = maximum_open
        self.handles: OrderedDict[tuple[int, int], BinaryIO] = OrderedDict()
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("xb").close()

    def write(self, key: tuple[int, int], payload: bytes) -> None:
        handle = self.handles.pop(key, None)
        if handle is None:
            if len(self.handles) >= self.maximum_open:
                _old_key, old = self.handles.popitem(last=False)
                old.close()
            handle = self.paths[key].open("ab")
        handle.write(payload)
        self.handles[key] = handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _build_deletions(
    runs: list[Path],
    group_path: Path,
    root: Path,
    components: tuple[Component, ...],
    maximum_open_files: int,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    paths = {
        (component.priority, shard): (
            root / component.name / f"shard_{shard:05d}.deletions.bin"
        )
        for component in components
        for shard in range(component.logical_shards)
    }
    row_counts: Counter[tuple[int, int]] = Counter()
    pool = _DeletionPool(paths, maximum_open_files)
    try:
        with group_path.open("rb") as groups:
            group_iterator = _records(groups, _GROUP)
            group = next(group_iterator, None)
            active_hash = None
            rank = 0
            boundary_document = None
            for record in _merged_records(runs, _SIGNATURE):
                if record[0] != active_hash:
                    if active_hash is not None:
                        group = next(group_iterator, None)
                    active_hash = record[0]
                    rank = 0
                    boundary_document = None
                if group is None or group[0] != record[0]:
                    raise CrossSourceSubdocumentDecisionError(
                        "signature group join differs"
                    )
                rank += 1
                document = record[2]
                document_key = (record[1], document)
                if rank <= group[3]:
                    boundary_document = document_key
                    counts["retained_occurrences"] += 1
                    continue
                if document_key == boundary_document:
                    counts["coherence_boundary_retained_occurrences"] += 1
                    continue
                key = (record[1], record[3])
                pool.write(
                    key,
                    DELETE_RECORD.pack(
                        record[1],
                        record[3],
                        record[4],
                        document,
                        record[5],
                        record[6],
                        record[7],
                        record[0],
                        group[1],
                        group[3],
                    ),
                )
                row_counts[key] += 1
                component = components[record[1]]
                counts["deletion_occurrences"] += 1
                counts[f"component::{component.name}::deletion_occurrences"] += 1
            if next(group_iterator, None) is not None:
                raise CrossSourceSubdocumentDecisionError(
                    "unused signature groups remain"
                )
    finally:
        pool.close()
    descriptors = []
    for component in components:
        for shard in range(component.logical_shards):
            key = (component.priority, shard)
            path = paths[key]
            descriptors.append(
                {
                    "component": component.name,
                    "component_priority": component.priority,
                    "source_shard": shard,
                    "path": str(path.relative_to(root)),
                    "rows": row_counts[key],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return counts, descriptors


def build_decision(
    book_signature_root: Path,
    pleias_signature_root: Path,
    output_root: Path,
    bucket_index: int,
    book_logical_shards: int = 64,
    pleias_logical_shards: int = 128,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    maximum_open_runs: int = DEFAULT_MAXIMUM_OPEN_RUNS,
    maximum_open_deletion_files: int = DEFAULT_MAXIMUM_OPEN_DELETION_FILES,
    reference_characters: int = DEFAULT_REFERENCE_CHARACTERS,
    effective_shards_numerator: int = DEFAULT_EFFECTIVE_SHARDS_NUMERATOR,
    effective_shards_denominator: int = DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR,
    temporary_root: Path | None = None,
) -> dict[str, Any]:
    """Join one complete hash bucket and emit component/shard deletion maps."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not 0 <= bucket_index < HASH_BUCKETS
        or chunk_records <= 0
        or maximum_open_runs < 2
        or maximum_open_deletion_files <= 0
    ):
        raise CrossSourceSubdocumentDecisionError("decision arguments differ")
    components = _components(
        book_signature_root,
        pleias_signature_root,
        book_logical_shards,
        pleias_logical_shards,
    )
    output_root.mkdir(parents=True)
    deletion_root = output_root / "deletions"
    deletion_root.mkdir()
    group_path = output_root / "groups.bin"
    with tempfile.TemporaryDirectory(
        prefix="sai-cross-source-subdocument-", dir=temporary_root
    ) as directory:
        scratch = Path(directory)
        runs, input_counts, bindings = _build_runs(
            components, bucket_index, scratch, chunk_records
        )
        runs, merge_passes = _reduce_runs(
            runs,
            scratch,
            _SIGNATURE,
            prefix="cross-source-signature",
            maximum_open_chunks=maximum_open_runs,
        )
        temporary_groups = scratch / "groups.bin"
        group_counts = _build_groups(
            runs,
            temporary_groups,
            reference_characters,
            effective_shards_numerator,
            effective_shards_denominator,
        )
        deletion_counts, deletions = _build_deletions(
            runs,
            temporary_groups,
            deletion_root,
            components,
            maximum_open_deletion_files,
        )
        os.replace(temporary_groups, group_path)
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_cross_source_subdocument_decision",
        "hash_bucket": {
            "index": bucket_index,
            "buckets": HASH_BUCKETS,
            "key": "first_normalized_sha256_hex_nibble",
        },
        "components": bindings,
        "policy": {
            "retention": "adaptive_frequency_length",
            "reference_characters": reference_characters,
            "effective_shards_numerator": effective_shards_numerator,
            "effective_shards_denominator": effective_shards_denominator,
            "boundary_document_retains_all_same_group_occurrences": True,
            "source_priority": [
                "institutional_books",
                "pleias_common_corpus",
            ],
            "representative_order": (
                "normalized_hash_then_source_priority_then_document_identity_then_locator"
            ),
        },
        "counts": dict(sorted((input_counts + group_counts + deletion_counts).items())),
        "external_sort": {
            "record_bytes": _SIGNATURE.size,
            "deletion_record_bytes": DELETE_RECORD.size,
            "chunk_records": chunk_records,
            "maximum_open_runs": maximum_open_runs,
            "merge_passes": merge_passes,
            "maximum_open_deletion_files": maximum_open_deletion_files,
        },
        "groups": {
            "path": group_path.name,
            "bytes": group_path.stat().st_size,
            "sha256": sha256_file(group_path),
            "rows": group_counts["groups"],
        },
        "deletions": deletions,
        "ordered_deletion_descriptors_sha256": canonical_sha256(deletions),
        "decision_contains_source_text": False,
        "cross_source_subdocument_decision_complete": True,
        "rewrite_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-signature-root", type=Path, required=True)
    parser.add_argument("--pleias-signature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bucket-index", type=int, required=True)
    parser.add_argument("--book-logical-shards", type=int, default=64)
    parser.add_argument("--pleias-logical-shards", type=int, default=128)
    parser.add_argument("--chunk-records", type=int, default=DEFAULT_CHUNK_RECORDS)
    parser.add_argument(
        "--maximum-open-runs", type=int, default=DEFAULT_MAXIMUM_OPEN_RUNS
    )
    parser.add_argument(
        "--maximum-open-deletion-files",
        type=int,
        default=DEFAULT_MAXIMUM_OPEN_DELETION_FILES,
    )
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    result = build_decision(
        args.book_signature_root,
        args.pleias_signature_root,
        args.output_root,
        args.bucket_index,
        args.book_logical_shards,
        args.pleias_logical_shards,
        args.chunk_records,
        args.maximum_open_runs,
        args.maximum_open_deletion_files,
        temporary_root=args.temporary_root,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
