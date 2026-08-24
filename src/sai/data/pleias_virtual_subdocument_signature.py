"""Stream selected PleIAs rows into benchmark-clean virtual signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import binary_boundary_index
from sai.data.pleias_bounded_mechanical_candidates import _download
from sai.data.pleias_full_candidate_decontamination import screen_text
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_production_materializer import (
    _selection_database,
    _semantic_metadata,
    replay_selected_row,
)
from sai.data.pleias_subdocument_signature import (
    AGGREGATE_SCHEMA,
    HASH_BUCKETS,
    SHARD_SCHEMA,
    signature_rows,
)
from sai.data.pleias_subdocument_signature import (
    _schema as signature_schema,
)
from sai.data.token_stream import canonical_sha256, sha256_file

LOCATOR_SCHEMA = "sai-pleias-virtual-reconstruction-locator-v1"
VIRTUAL_SHARD_STATUS = "complete_nontraining_pleias_virtual_subdocument_signature_shard"
VIRTUAL_AGGREGATE_STATUS = "complete_nontraining_pleias_virtual_subdocument_signatures"


class PleiasVirtualSubdocumentSignatureError(RuntimeError):
    """Selection replay, benchmark screen, locator, or signature differs."""


def _locator_schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasVirtualSubdocumentSignatureError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("virtual_row_index", pa.int64()),
            ("source_repository", pa.string()),
            ("source_revision", pa.string()),
            ("source_path", pa.string()),
            ("source_parent_sha256", pa.string()),
            ("source_row_index", pa.int64()),
            ("source_row_identity_sha256", pa.string()),
            ("content_sha256", pa.string()),
            ("text_utf8_bytes", pa.int64()),
            ("text_characters", pa.int64()),
            ("source_word_count", pa.int64()),
            ("source_token_count", pa.int64()),
            ("collection", pa.string()),
            ("open_type", pa.string()),
            ("license", pa.string()),
            ("language", pa.string()),
            ("semantic_stratum", pa.string()),
            ("semantic_quality_floor_milli", pa.int32()),
            ("semantic_quality_mean_milli", pa.int32()),
            ("semantic_difficulty_mean_milli", pa.int32()),
            ("semantic_prerequisite_burden_mean_milli", pa.int32()),
            ("semantic_curriculum_phase", pa.string()),
            ("semantic_domains", pa.list_(pa.string())),
            ("semantic_recurring_concepts", pa.list_(pa.string())),
            ("semantic_recurring_prerequisites", pa.list_(pa.string())),
            ("code_document", pa.bool_()),
            ("locator_sha256", pa.string()),
            ("training_ready", pa.bool_()),
        ]
    )


def locator_row(
    candidate: dict[str, Any], virtual_row_index: int, source_row_index: int
) -> dict[str, Any]:
    """Strip text while preserving enough identity to reconstruct one clean row."""

    text = candidate.get("text")
    collection = candidate.get("collection")
    if (
        not isinstance(text, str)
        or not text
        or isinstance(virtual_row_index, bool)
        or not isinstance(virtual_row_index, int)
        or virtual_row_index < 0
        or isinstance(source_row_index, bool)
        or not isinstance(source_row_index, int)
        or source_row_index < 0
        or not isinstance(collection, str)
        or not collection
    ):
        raise PleiasVirtualSubdocumentSignatureError("locator source row differs")
    row = {
        "schema": LOCATOR_SCHEMA,
        "virtual_row_index": virtual_row_index,
        "source_repository": candidate["source_repository"],
        "source_revision": candidate["source_revision"],
        "source_path": candidate["source_path"],
        "source_parent_sha256": candidate["source_parent_sha256"],
        "source_row_index": source_row_index,
        "source_row_identity_sha256": candidate["source_row_identity_sha256"],
        "content_sha256": candidate["content_sha256"],
        "text_utf8_bytes": len(text.encode()),
        "text_characters": len(text),
        "source_word_count": candidate["word_count"],
        "source_token_count": candidate["token_count"],
        "collection": collection,
        "open_type": candidate["open_type"],
        "license": candidate["license"],
        "language": candidate["language"],
        "semantic_stratum": candidate["semantic_stratum"],
        "semantic_quality_floor_milli": candidate["semantic_quality_floor_milli"],
        "semantic_quality_mean_milli": candidate["semantic_quality_mean_milli"],
        "semantic_difficulty_mean_milli": candidate["semantic_difficulty_mean_milli"],
        "semantic_prerequisite_burden_mean_milli": candidate[
            "semantic_prerequisite_burden_mean_milli"
        ],
        "semantic_curriculum_phase": candidate["semantic_curriculum_phase"],
        "semantic_domains": candidate["semantic_domains"],
        "semantic_recurring_concepts": candidate["semantic_recurring_concepts"],
        "semantic_recurring_prerequisites": candidate[
            "semantic_recurring_prerequisites"
        ],
        "code_document": "github" in collection.casefold(),
        "training_ready": False,
    }
    row["locator_sha256"] = canonical_sha256(row)
    return row


def _selected_rows(connection: sqlite3.Connection, source_path: str) -> list[Any]:
    return connection.execute(
        "SELECT source_row_index, source_row_identity_sha256, "
        "source_parent_sha256, content_sha256, stratum, text_utf8_bytes, "
        "token_count, stratum_quality_floor_milli, stratum_quality_mean_milli "
        "FROM selected WHERE source_path=? ORDER BY source_row_index",
        (source_path,),
    ).fetchall()


def run_shard(
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    boundary_roots: list[Path],
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Replay one source partition and retain only locators and signatures."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or not boundary_roots
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasVirtualSubdocumentSignatureError("virtual arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualSubdocumentSignatureError("pyarrow is required") from error
    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    selection, selection_path = _selection_database(selection_root)
    semantic_by_stratum, semantic_receipt = _semantic_metadata(semantic_decision_path)
    connection = sqlite3.connect(f"file:{selection_path.resolve()}?mode=ro", uri=True)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    word_boundary = words[0] if len(words) == 1 else _Union(words)
    code_boundary = code[0] if len(code) == 1 else _Union(code)
    temporary_name = f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary_root = output_root.parent / temporary_name
    temporary_root.mkdir(parents=True)
    output_paths = [
        temporary_root / f"bucket-{index:02x}.parquet" for index in range(HASH_BUCKETS)
    ]
    locator_path = temporary_root / "retained-locators.parquet"
    writers = [
        pq.ParquetWriter(path, signature_schema(), compression="zstd")
        for path in output_paths
    ]
    locator_writer = pq.ParquetWriter(
        locator_path, _locator_schema(), compression="zstd"
    )
    counts: Counter[str] = Counter()
    ordered = hashlib.sha256()
    ordered_by_bucket = [hashlib.sha256() for _ in range(HASH_BUCKETS)]
    ordered_selected = hashlib.sha256()
    ordered_retained = hashlib.sha256()
    ordered_contaminated = hashlib.sha256()
    ordered_locators = hashlib.sha256()
    parent_receipts = []
    virtual_row_index = 0
    try:
        for parent_number, parent in enumerate(parents, start=1):
            selected_rows = _selected_rows(connection, parent["source_path"])
            if not selected_rows:
                counts["parents_without_selected_rows"] += 1
                continue
            by_index = {row[0]: row[1:] for row in selected_rows}
            if len(by_index) != len(selected_rows):
                raise PleiasVirtualSubdocumentSignatureError(
                    "selected parent indices overlap"
                )
            with tempfile.TemporaryDirectory(
                prefix="sai-pleias-virtual-signature-", dir=scratch_root
            ) as directory:
                source_path = _download(parent, token, Path(directory))
                parquet = pq.ParquetFile(source_path)
                seen = set()
                row_offset = 0
                for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                    signature_batches = [[] for _ in range(HASH_BUCKETS)]
                    locator_batch = []
                    for relative, source_row in enumerate(batch.to_pylist()):
                        source_row_index = row_offset + relative
                        expected = by_index.get(source_row_index)
                        if expected is None:
                            continue
                        candidate = replay_selected_row(
                            source_row,
                            parent,
                            source_row_index,
                            expected,
                            semantic_by_stratum.get(expected[3], {}),
                        )
                        seen.add(source_row_index)
                        identity = candidate["source_row_identity_sha256"]
                        ordered_selected.update(bytes.fromhex(identity))
                        counts["selected_rows_replayed"] += 1
                        counts["selected_text_utf8_bytes_replayed"] += len(
                            candidate["text"].encode()
                        )
                        word_overlap, code_overlap = screen_text(
                            candidate["text"], word_boundary, code_boundary
                        )
                        counts["word_overlap_shingles"] += word_overlap
                        counts["code_overlap_shingles"] += code_overlap
                        if word_overlap or code_overlap:
                            counts["benchmark_contaminated_rows"] += 1
                            ordered_contaminated.update(bytes.fromhex(identity))
                            continue
                        locator = locator_row(
                            candidate, virtual_row_index, source_row_index
                        )
                        locator_batch.append(locator)
                        ordered_locators.update(
                            bytes.fromhex(locator["locator_sha256"])
                        )
                        ordered_retained.update(bytes.fromhex(identity))
                        rows = signature_rows(candidate, shard_index, virtual_row_index)
                        for row in rows:
                            bucket = int(row["normalized_sha256"][0], 16)
                            signature_batches[bucket].append(row)
                            ordered.update(bytes.fromhex(row["signature_sha256"]))
                            ordered_by_bucket[bucket].update(
                                bytes.fromhex(row["signature_sha256"])
                            )
                            counts[f"bucket_{bucket:02x}_signatures"] += 1
                        counts["retained_rows"] += 1
                        counts["retained_text_utf8_bytes"] += len(
                            candidate["text"].encode()
                        )
                        counts["signatures"] += len(rows)
                        counts["code_signatures"] += sum(row["code"] for row in rows)
                        virtual_row_index += 1
                    for bucket, rows in enumerate(signature_batches):
                        if rows:
                            writers[bucket].write_table(
                                pa.Table.from_pylist(rows, schema=signature_schema())
                            )
                    if locator_batch:
                        locator_writer.write_table(
                            pa.Table.from_pylist(
                                locator_batch, schema=_locator_schema()
                            )
                        )
                    row_offset += batch.num_rows
                if seen != set(by_index):
                    raise PleiasVirtualSubdocumentSignatureError(
                        "selected parent coverage differs"
                    )
            parent_receipts.append(
                {
                    "source_path": parent["source_path"],
                    "source_sha256": parent["sha256"],
                    "selected_rows": len(selected_rows),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "pleias_virtual_signature_progress",
                        "shard_index": shard_index,
                        "complete_parents": parent_number,
                        "remaining_parents": len(parents) - parent_number,
                        "retained_rows": counts["retained_rows"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            counts["retained_rows"] + counts["benchmark_contaminated_rows"]
            != counts["selected_rows_replayed"]
        ):
            raise PleiasVirtualSubdocumentSignatureError(
                "benchmark route accounting differs"
            )
    except BaseException:
        for writer in writers:
            writer.close()
        locator_writer.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    finally:
        connection.close()
        for boundary in [*words, *code]:
            boundary.close()
    for writer in writers:
        writer.close()
    locator_writer.close()
    outputs = [
        {
            "bucket": index,
            "path": path.name,
            "rows": counts[f"bucket_{index:02x}_signatures"],
            "ordered_signature_digests_sha256": ordered_by_bucket[index].hexdigest(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for index, path in enumerate(output_paths)
    ]
    locator = {
        "path": locator_path.name,
        "rows": counts["retained_rows"],
        "bytes": locator_path.stat().st_size,
        "sha256": sha256_file(locator_path),
        "ordered_locator_digests_sha256": ordered_locators.hexdigest(),
    }
    payload = {
        "schema": SHARD_SCHEMA,
        "status": VIRTUAL_SHARD_STATUS,
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "selection_receipt_sha256": selection["receipt_sha256"],
            "semantic_decision_receipt_sha256": semantic_receipt["receipt_sha256"],
            "boundary_receipts_sha256": canonical_sha256(
                [row["receipt_sha256"] for row in boundary_receipts]
            ),
            "selected_parent_count": len(parent_receipts),
            "ordered_selected_parent_receipts_sha256": canonical_sha256(
                parent_receipts
            ),
        },
        "policy": {
            "source_reconstruction": "pinned_parent_and_row_locator",
            "full_document_benchmark_screen_before_signature": True,
            "source_text_persisted": False,
        },
        "counts": dict(sorted(counts.items())),
        "ordered_selected_identities_sha256": ordered_selected.hexdigest(),
        "ordered_retained_identities_sha256": ordered_retained.hexdigest(),
        "ordered_contaminated_identities_sha256": ordered_contaminated.hexdigest(),
        "ordered_signature_digests_sha256": ordered.hexdigest(),
        "hash_partition": {
            "buckets": HASH_BUCKETS,
            "key": "first_normalized_sha256_hex_nibble",
        },
        "outputs": outputs,
        "retained_locators": locator,
        "virtual_reconstruction_manifest_complete": True,
        "full_document_benchmark_decontamination_complete": True,
        "complete_virtual_document_coverage": True,
        "source_text_persisted": False,
        "global_subdocument_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(temporary_root / "receipt.json", payload)
    os.replace(temporary_root, output_root)
    return payload


def aggregate_virtual(
    selection_root: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Seal exact virtual locator and signature coverage without source text."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise PleiasVirtualSubdocumentSignatureError("aggregate arguments differ")
    selection, selection_path = _selection_database(selection_root)
    connection = sqlite3.connect(f"file:{selection_path.resolve()}?mode=ro", uri=True)
    selected_rows, selected_bytes = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(text_utf8_bytes), 0) FROM selected"
    ).fetchone()
    connection.close()
    totals: Counter[str] = Counter()
    receipts = []
    selection_receipts = set()
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_virtual_receipt(root / "receipt.json")
        outputs = receipt.get("outputs")
        locator = receipt.get("retained_locators")
        if (
            receipt.get("status") != VIRTUAL_SHARD_STATUS
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("complete_virtual_document_coverage") is not True
            or receipt.get("source_text_persisted") is not False
            or not isinstance(outputs, list)
            or len(outputs) != HASH_BUCKETS
            or not isinstance(locator, dict)
        ):
            raise PleiasVirtualSubdocumentSignatureError(
                "virtual signature shard differs"
            )
        for index, descriptor in enumerate(outputs):
            path = root / descriptor.get("path", "")
            if (
                descriptor.get("bucket") != index
                or descriptor.get("rows")
                != receipt.get("counts", {}).get(f"bucket_{index:02x}_signatures", 0)
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise PleiasVirtualSubdocumentSignatureError(
                    "virtual signature output differs"
                )
            totals["signature_output_bytes"] += descriptor["bytes"]
        locator_path = root / locator.get("path", "")
        if (
            locator.get("rows") != receipt.get("counts", {}).get("retained_rows")
            or not locator_path.is_file()
            or locator_path.is_symlink()
            or locator_path.stat().st_nlink != 1
            or locator_path.stat().st_size != locator.get("bytes")
            or sha256_file(locator_path) != locator.get("sha256")
        ):
            raise PleiasVirtualSubdocumentSignatureError(
                "virtual locator output differs"
            )
        totals["locator_output_bytes"] += locator["bytes"]
        totals.update(receipt["counts"])
        receipts.append(receipt["receipt_sha256"])
        selection_receipts.add(receipt["source"]["selection_receipt_sha256"])
    if (
        selection_receipts != {selection["receipt_sha256"]}
        or totals["selected_rows_replayed"] != selected_rows
        or totals["selected_text_utf8_bytes_replayed"] != selected_bytes
        or totals["retained_rows"] + totals["benchmark_contaminated_rows"]
        != selected_rows
    ):
        raise PleiasVirtualSubdocumentSignatureError(
            "virtual selection coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": VIRTUAL_AGGREGATE_STATUS,
        "source": {
            "selection_receipt_sha256": selection["receipt_sha256"],
            "selection_rows": selected_rows,
            "selection_text_utf8_bytes": selected_bytes,
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "complete_materialized_document_coverage": False,
        "complete_virtual_document_coverage": True,
        "full_document_benchmark_decontamination_complete": True,
        "source_text_persisted": False,
        "global_subdocument_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def _load_virtual_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasVirtualSubdocumentSignatureError("virtual receipt is unsafe")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasVirtualSubdocumentSignatureError(
            "virtual receipt is invalid"
        ) from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != SHARD_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasVirtualSubdocumentSignatureError("virtual receipt differs")
    return value


class _Union:
    def __init__(self, members: list[Any]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--selection-root", type=Path, required=True)
    shard.add_argument("--semantic-decision", type=Path, required=True)
    shard.add_argument("--boundary-index", type=Path, action="append", required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--selection-root", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.selection_root,
            args.semantic_decision,
            args.boundary_index,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate_virtual(
            args.selection_root,
            args.shards_root,
            args.logical_shards,
            args.output,
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
