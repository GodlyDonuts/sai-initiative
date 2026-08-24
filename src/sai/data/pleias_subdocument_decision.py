"""Build a global frequency/length-aware PleIAs subdocument deletion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

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
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_subdocument_signature import (
    AGGREGATE_SCHEMA as SIGNATURE_AGGREGATE_SCHEMA,
)
from sai.data.pleias_subdocument_signature import HASH_BUCKETS, SIGNATURE_SCHEMA
from sai.data.pleias_subdocument_signature import (
    SHARD_SCHEMA as SIGNATURE_SHARD_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-global-subdocument-decision-v1"
_SIGNATURE = struct.Struct(">32s32sHQIQQQB")
_DELETE = struct.Struct(">HQ32sIQQ32sQQ")
DEFAULT_CHUNK_RECORDS = 500_000
DEFAULT_MAXIMUM_OPEN_RUNS = 128


class PleiasSubdocumentDecisionError(RuntimeError):
    """Signature custody, external sort, group, or deletion replay differs."""


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_signature(
    row: dict[str, Any], shard_index: int, bucket_index: int
) -> None:
    unsigned = {key: value for key, value in row.items() if key != "signature_sha256"}
    if (
        row.get("schema") != SIGNATURE_SCHEMA
        or row.get("training_ready") is not False
        or row.get("component") != "pleias_common_corpus"
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
        raise PleiasSubdocumentDecisionError("signature row differs")


def _signature_record(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bytes.fromhex(row["normalized_sha256"]),
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
    signature_root: Path,
    logical_shards: int,
    bucket_index: int,
    scratch: Path,
    chunk_records: int,
) -> tuple[list[Path], Counter[str], list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasSubdocumentDecisionError("pyarrow is required") from error
    records = []
    runs = []
    counts: Counter[str] = Counter()
    receipts = []
    for shard_index in range(logical_shards):
        root = signature_root / "shards" / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SIGNATURE_SHARD_SCHEMA)
        outputs = receipt.get("outputs")
        descriptor = (
            outputs[bucket_index]
            if isinstance(outputs, list) and len(outputs) == HASH_BUCKETS
            else {}
        )
        path = root / descriptor.get("path", "")
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or descriptor.get("bucket") != bucket_index
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise PleiasSubdocumentDecisionError("signature shard differs")
        ordered = hashlib.sha256()
        rows = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
            for row in batch.to_pylist():
                _validate_signature(row, shard_index, bucket_index)
                ordered.update(bytes.fromhex(row["signature_sha256"]))
                records.append(_signature_record(row))
                rows += 1
                counts["signatures"] += 1
                if len(records) >= chunk_records:
                    run = scratch / f"signature-{len(runs):08d}.bin"
                    runs.append(_write_records(records, _SIGNATURE, run))
                    records = []
        if rows != descriptor.get("rows") or ordered.hexdigest() != descriptor.get(
            "ordered_signature_digests_sha256"
        ):
            raise PleiasSubdocumentDecisionError("signature coverage differs")
        receipts.append(receipt["receipt_sha256"])
    if records:
        run = scratch / f"signature-{len(runs):08d}.bin"
        runs.append(_write_records(records, _SIGNATURE, run))
    if not runs:
        raise PleiasSubdocumentDecisionError("signature population is empty")
    return runs, counts, receipts


def _build_groups(
    runs: list[Path],
    group_path: Path,
    reference_characters: int,
    numerator: int,
    denominator: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    active_hash = None
    active_length = 0
    active_code = 0
    frequency = 0

    def flush(handle) -> None:
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

    with group_path.open("xb") as handle:
        for record in _merged_records(runs, _SIGNATURE):
            if record[0] != active_hash:
                flush(handle)
                active_hash = record[0]
                active_length = record[7]
                active_code = record[8]
                frequency = 1
            else:
                if record[7] != active_length or record[8] != active_code:
                    raise PleiasSubdocumentDecisionError(
                        "normalized signature collision differs"
                    )
                frequency += 1
        flush(handle)
    return counts


def _build_deletions(
    runs: list[Path],
    group_path: Path,
    deletion_root: Path,
    logical_shards: int,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    row_counts = [0] * logical_shards
    with ExitStack() as stack, group_path.open("rb") as groups:
        handles = [
            stack.enter_context(
                (deletion_root / f"shard_{index:05d}.deletions.bin").open("xb")
            )
            for index in range(logical_shards)
        ]
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
                raise PleiasSubdocumentDecisionError("signature group join differs")
            rank += 1
            if rank <= group[3]:
                boundary_document = record[1]
                counts["retained_occurrences"] += 1
                continue
            if record[1] == boundary_document:
                counts["coherence_boundary_retained_occurrences"] += 1
                continue
            shard_index = record[2]
            handles[shard_index].write(
                _DELETE.pack(
                    shard_index,
                    record[3],
                    record[1],
                    record[4],
                    record[5],
                    record[6],
                    record[0],
                    group[1],
                    group[3],
                )
            )
            row_counts[shard_index] += 1
            counts["deletion_occurrences"] += 1
        if next(group_iterator, None) is not None:
            raise PleiasSubdocumentDecisionError("unused signature groups remain")
    descriptors = []
    for index, rows in enumerate(row_counts):
        path = deletion_root / f"shard_{index:05d}.deletions.bin"
        descriptors.append(
            {
                "shard_index": index,
                "path": path.name,
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return counts, descriptors


def build_decision(
    signature_root: Path,
    output_root: Path,
    logical_shards: int,
    bucket_index: int,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    maximum_open_runs: int = DEFAULT_MAXIMUM_OPEN_RUNS,
    reference_characters: int = DEFAULT_REFERENCE_CHARACTERS,
    effective_shards_numerator: int = DEFAULT_EFFECTIVE_SHARDS_NUMERATOR,
    effective_shards_denominator: int = DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR,
    temporary_root: Path | None = None,
) -> dict[str, Any]:
    """External-sort all signatures and emit shard-partitioned deletion maps."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or logical_shards <= 0
        or not 0 <= bucket_index < HASH_BUCKETS
        or chunk_records <= 0
        or maximum_open_runs < 2
    ):
        raise PleiasSubdocumentDecisionError("decision arguments differ")
    aggregate = _load_signed(
        signature_root / "aggregate.json", SIGNATURE_AGGREGATE_SCHEMA
    )
    if (
        aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or not (
            aggregate.get("complete_materialized_document_coverage") is True
            or aggregate.get("complete_virtual_document_coverage") is True
        )
        or aggregate.get("source_text_persisted") is not False
    ):
        raise PleiasSubdocumentDecisionError("signature aggregate differs")
    output_root.mkdir(parents=True)
    deletion_root = output_root / "deletions"
    deletion_root.mkdir()
    group_path = output_root / "groups.bin"
    with tempfile.TemporaryDirectory(
        prefix="sai-pleias-subdocument-decision-", dir=temporary_root
    ) as directory:
        scratch = Path(directory)
        runs, input_counts, receipts = _build_runs(
            signature_root,
            logical_shards,
            bucket_index,
            scratch,
            chunk_records,
        )
        runs, merge_passes = _reduce_runs(
            runs,
            scratch,
            _SIGNATURE,
            prefix="signature",
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
            runs, temporary_groups, deletion_root, logical_shards
        )
        os.replace(temporary_groups, group_path)
    if input_counts["signatures"] != aggregate.get("totals", {}).get(
        f"bucket_{bucket_index:02x}_signatures"
    ):
        raise PleiasSubdocumentDecisionError("aggregate signature count differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_global_subdocument_decision",
        "hash_bucket": {
            "index": bucket_index,
            "buckets": HASH_BUCKETS,
            "key": "first_normalized_sha256_hex_nibble",
        },
        "source": {
            "signature_aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "ordered_signature_shard_receipts_sha256": canonical_sha256(receipts),
        },
        "policy": {
            "retention": "adaptive_frequency_length",
            "reference_characters": reference_characters,
            "effective_shards_numerator": effective_shards_numerator,
            "effective_shards_denominator": effective_shards_denominator,
            "boundary_document_retains_all_same_group_occurrences": True,
            "representative_order": (
                "normalized_hash_then_document_identity_then_locator"
            ),
        },
        "counts": dict(sorted((input_counts + group_counts + deletion_counts).items())),
        "external_sort": {
            "record_bytes": _SIGNATURE.size,
            "chunk_records": chunk_records,
            "maximum_open_runs": maximum_open_runs,
            "merge_passes": merge_passes,
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
        "pleias_global_subdocument_decision_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
        "rewrite_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--bucket-index", type=int, required=True)
    parser.add_argument("--chunk-records", type=int, default=DEFAULT_CHUNK_RECORDS)
    parser.add_argument(
        "--maximum-open-runs", type=int, default=DEFAULT_MAXIMUM_OPEN_RUNS
    )
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    result = build_decision(
        args.signature_root,
        args.output_root,
        args.logical_shards,
        args.bucket_index,
        args.chunk_records,
        args.maximum_open_runs,
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
