"""Reconstruct, internally deduplicate, and re-sign virtual PleIAs rows."""

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
from sai.data.pleias_bounded_mechanical_candidates import _download
from sai.data.pleias_final_subdocument_signature import (
    AGGREGATE_SCHEMA,
    COMPONENT,
    SHARD_SCHEMA,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_production_materializer import (
    _selection_database,
    _semantic_metadata,
    replay_selected_row,
)
from sai.data.pleias_subdocument_rewrite import (
    _decision_database,
    rewrite_candidate,
)
from sai.data.pleias_subdocument_signature import (
    HASH_BUCKETS,
    signature_rows_for_text,
)
from sai.data.pleias_subdocument_signature import (
    _schema as signature_schema,
)
from sai.data.pleias_virtual_subdocument_signature import (
    AGGREGATE_SCHEMA as VIRTUAL_AGGREGATE_SCHEMA,
)
from sai.data.pleias_virtual_subdocument_signature import (
    LOCATOR_SCHEMA,
    VIRTUAL_AGGREGATE_STATUS,
    _load_virtual_receipt,
    _selected_rows,
)
from sai.data.token_stream import canonical_sha256, sha256_file

TRANSFORMED_LOCATOR_SCHEMA = "sai-pleias-virtual-internal-transform-locator-v1"
SHARD_STATUS = "complete_nontraining_pleias_virtual_internal_rewrite_signatures"
AGGREGATE_STATUS = (
    "complete_nontraining_pleias_virtual_internal_rewrite_signature_aggregate"
)


class PleiasVirtualInternalRewriteSignatureError(RuntimeError):
    """Virtual locator, deletion, reconstruction, or signature differs."""


def _transformed_locator_schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasVirtualInternalRewriteSignatureError(
            "pyarrow is required"
        ) from error
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
            ("pre_internal_content_sha256", pa.string()),
            ("content_sha256", pa.string()),
            ("source_text_utf8_bytes", pa.int64()),
            ("output_text_utf8_bytes", pa.int64()),
            ("source_text_characters", pa.int64()),
            ("output_text_characters", pa.int64()),
            ("source_word_count", pa.int64()),
            ("output_word_count", pa.int64()),
            ("source_token_count", pa.int64()),
            ("token_count_requires_recomputation", pa.bool_()),
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
            ("internal_subdocument_transform_sha256", pa.string()),
            ("locator_sha256", pa.string()),
            ("training_ready", pa.bool_()),
        ]
    )


def transformed_locator_row(
    source_locator: dict[str, Any], rewritten: dict[str, Any]
) -> dict[str, Any]:
    """Bind an exact internal transformation without retaining its text."""

    text = rewritten.get("text")
    unsigned_source = {
        key: value for key, value in source_locator.items() if key != "locator_sha256"
    }
    if (
        source_locator.get("schema") != LOCATOR_SCHEMA
        or source_locator.get("locator_sha256") != canonical_sha256(unsigned_source)
        or not isinstance(text, str)
        or not text
        or rewritten.get("source_row_identity_sha256")
        != source_locator.get("source_row_identity_sha256")
        or rewritten.get("pre_dedup_content_sha256")
        != source_locator.get("content_sha256")
        or hashlib.sha256(text.encode()).hexdigest() != rewritten.get("content_sha256")
        or rewritten.get("token_count_requires_recomputation") is not True
    ):
        raise PleiasVirtualInternalRewriteSignatureError(
            "transformed locator source differs"
        )
    row = {
        "schema": TRANSFORMED_LOCATOR_SCHEMA,
        "virtual_row_index": source_locator["virtual_row_index"],
        "source_repository": source_locator["source_repository"],
        "source_revision": source_locator["source_revision"],
        "source_path": source_locator["source_path"],
        "source_parent_sha256": source_locator["source_parent_sha256"],
        "source_row_index": source_locator["source_row_index"],
        "source_row_identity_sha256": source_locator["source_row_identity_sha256"],
        "pre_internal_content_sha256": source_locator["content_sha256"],
        "content_sha256": rewritten["content_sha256"],
        "source_text_utf8_bytes": source_locator["text_utf8_bytes"],
        "output_text_utf8_bytes": len(text.encode()),
        "source_text_characters": source_locator["text_characters"],
        "output_text_characters": len(text),
        "source_word_count": source_locator["source_word_count"],
        "output_word_count": rewritten["word_count"],
        "source_token_count": source_locator["source_token_count"],
        "token_count_requires_recomputation": True,
        "collection": source_locator["collection"],
        "open_type": source_locator["open_type"],
        "license": source_locator["license"],
        "language": source_locator["language"],
        "semantic_stratum": source_locator["semantic_stratum"],
        "semantic_quality_floor_milli": source_locator["semantic_quality_floor_milli"],
        "semantic_quality_mean_milli": source_locator["semantic_quality_mean_milli"],
        "semantic_difficulty_mean_milli": source_locator[
            "semantic_difficulty_mean_milli"
        ],
        "semantic_prerequisite_burden_mean_milli": source_locator[
            "semantic_prerequisite_burden_mean_milli"
        ],
        "semantic_curriculum_phase": source_locator["semantic_curriculum_phase"],
        "semantic_domains": source_locator["semantic_domains"],
        "semantic_recurring_concepts": source_locator["semantic_recurring_concepts"],
        "semantic_recurring_prerequisites": source_locator[
            "semantic_recurring_prerequisites"
        ],
        "code_document": source_locator["code_document"],
        "internal_subdocument_transform_sha256": rewritten[
            "subdocument_transform_sha256"
        ],
        "training_ready": False,
    }
    row["locator_sha256"] = canonical_sha256(row)
    return row


def _locator_database(
    virtual_root: Path,
    shard_index: int,
    logical_shards: int,
    database_path: Path,
) -> tuple[sqlite3.Connection, dict[str, Any], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualInternalRewriteSignatureError(
            "pyarrow is required"
        ) from error
    root = virtual_root / "shards" / f"shard_{shard_index:05d}"
    receipt = _load_virtual_receipt(root / "receipt.json")
    descriptor = receipt.get("retained_locators")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("complete_virtual_document_coverage") is not True
        or not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PleiasVirtualInternalRewriteSignatureError(
            "virtual locator descriptor differs"
        )
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE locators ("
        "virtual_row_index INTEGER PRIMARY KEY, source_path TEXT NOT NULL, "
        "source_row_index INTEGER NOT NULL, source_row_identity_sha256 TEXT NOT NULL, "
        "content_sha256 TEXT NOT NULL, locator_json TEXT NOT NULL, "
        "UNIQUE(source_path, source_row_index))"
    )
    rows = 0
    ordered = hashlib.sha256()
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
            for locator in batch.to_pylist():
                unsigned = {
                    key: value
                    for key, value in locator.items()
                    if key != "locator_sha256"
                }
                if (
                    locator.get("schema") != LOCATOR_SCHEMA
                    or locator.get("training_ready") is not False
                    or locator.get("locator_sha256") != canonical_sha256(unsigned)
                    or locator.get("virtual_row_index") != rows
                ):
                    raise PleiasVirtualInternalRewriteSignatureError(
                        "virtual locator row differs"
                    )
                connection.execute(
                    "INSERT INTO locators VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        locator["virtual_row_index"],
                        locator["source_path"],
                        locator["source_row_index"],
                        locator["source_row_identity_sha256"],
                        locator["content_sha256"],
                        json.dumps(locator, sort_keys=True, separators=(",", ":")),
                    ),
                )
                ordered.update(bytes.fromhex(locator["locator_sha256"]))
                rows += 1
        connection.commit()
        if rows != descriptor.get("rows") or ordered.hexdigest() != descriptor.get(
            "ordered_locator_digests_sha256"
        ):
            raise PleiasVirtualInternalRewriteSignatureError(
                "virtual locator coverage differs"
            )
    except BaseException:
        connection.close()
        raise
    return connection, receipt, rows


def run_shard(
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    virtual_root: Path,
    decision_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct one clean virtual partition and emit post-deletion signatures."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasVirtualInternalRewriteSignatureError(
            "virtual rewrite arguments differ"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualInternalRewriteSignatureError(
            "pyarrow is required"
        ) from error
    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    selection, selection_path = _selection_database(selection_root)
    semantic_by_stratum, semantic_receipt = _semantic_metadata(semantic_decision_path)
    selection_connection = sqlite3.connect(
        f"file:{selection_path.resolve()}?mode=ro", uri=True
    )
    temporary_name = f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary_root = output_root.parent / temporary_name
    temporary_root.mkdir(parents=True)
    output_paths = [
        temporary_root / f"bucket-{index:02x}.parquet" for index in range(HASH_BUCKETS)
    ]
    locator_path = temporary_root / "transformed-locators.parquet"
    writers = [
        pq.ParquetWriter(path, signature_schema(), compression="zstd")
        for path in output_paths
    ]
    locator_writer = pq.ParquetWriter(
        locator_path, _transformed_locator_schema(), compression="zstd"
    )
    counts: Counter[str] = Counter()
    ordered = hashlib.sha256()
    ordered_documents = hashlib.sha256()
    ordered_locators = hashlib.sha256()
    ordered_transforms = hashlib.sha256()
    ordered_by_bucket = [hashlib.sha256() for _ in range(HASH_BUCKETS)]
    parent_receipts = []
    with tempfile.TemporaryDirectory(
        prefix="sai-pleias-virtual-rewrite-state-", dir=scratch_root
    ) as state_directory:
        state = Path(state_directory)
        locator_connection, virtual_receipt, locator_rows = _locator_database(
            virtual_root,
            shard_index,
            logical_shards,
            state / "locators.sqlite3",
        )
        decision_connection, decision_receipts, decision_rows = _decision_database(
            decision_root,
            shard_index,
            logical_shards,
            state / "deletions.sqlite3",
        )
        try:
            for parent_number, parent in enumerate(parents, start=1):
                selected_rows = _selected_rows(
                    selection_connection, parent["source_path"]
                )
                if not selected_rows:
                    counts["parents_without_selected_rows"] += 1
                    continue
                by_index = {row[0]: row[1:] for row in selected_rows}
                with tempfile.TemporaryDirectory(
                    prefix="sai-pleias-virtual-rewrite-source-", dir=scratch_root
                ) as directory:
                    source_path = _download(parent, token, Path(directory))
                    parquet = pq.ParquetFile(source_path)
                    row_offset = 0
                    for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                        signature_batches = [[] for _ in range(HASH_BUCKETS)]
                        locator_batch = []
                        for relative, source_row in enumerate(batch.to_pylist()):
                            source_row_index = row_offset + relative
                            expected = by_index.get(source_row_index)
                            if expected is None:
                                continue
                            match = locator_connection.execute(
                                "SELECT virtual_row_index, source_row_identity_sha256, "
                                "content_sha256, locator_json FROM locators "
                                "WHERE source_path=? AND source_row_index=?",
                                (parent["source_path"], source_row_index),
                            ).fetchone()
                            if match is None:
                                counts["benchmark_contaminated_rows_skipped"] += 1
                                continue
                            candidate = replay_selected_row(
                                source_row,
                                parent,
                                source_row_index,
                                expected,
                                semantic_by_stratum.get(expected[3], {}),
                            )
                            virtual_row_index, identity, content_sha256, encoded = match
                            if (
                                candidate["source_row_identity_sha256"] != identity
                                or candidate["content_sha256"] != content_sha256
                            ):
                                raise PleiasVirtualInternalRewriteSignatureError(
                                    "reconstructed locator identity differs"
                                )
                            decisions = decision_connection.execute(
                                "SELECT document_identity_sha256, chunk_index, "
                                "character_start, character_end, normalized_sha256, "
                                "frequency, budget FROM deletions "
                                "WHERE source_row_index=? ORDER BY chunk_index",
                                (virtual_row_index,),
                            ).fetchall()
                            rewritten, row_counts = rewrite_candidate(
                                candidate, virtual_row_index, decisions
                            )
                            source_locator = json.loads(encoded)
                            locator = transformed_locator_row(source_locator, rewritten)
                            locator_batch.append(locator)
                            ordered_locators.update(
                                bytes.fromhex(locator["locator_sha256"])
                            )
                            ordered_transforms.update(
                                bytes.fromhex(
                                    locator["internal_subdocument_transform_sha256"]
                                )
                            )
                            ordered_documents.update(bytes.fromhex(identity))
                            rows = signature_rows_for_text(
                                component=COMPONENT,
                                text=rewritten["text"],
                                identity=identity,
                                content_sha256=rewritten["content_sha256"],
                                source_shard=shard_index,
                                source_row_index=virtual_row_index,
                                code_document=locator["code_document"],
                            )
                            for row in rows:
                                bucket = int(row["normalized_sha256"][0], 16)
                                signature_batches[bucket].append(row)
                                ordered.update(bytes.fromhex(row["signature_sha256"]))
                                ordered_by_bucket[bucket].update(
                                    bytes.fromhex(row["signature_sha256"])
                                )
                                counts[f"bucket_{bucket:02x}_signatures"] += 1
                            counts["documents"] += 1
                            counts["source_text_utf8_bytes"] += len(
                                candidate["text"].encode()
                            )
                            counts["output_text_utf8_bytes"] += len(
                                rewritten["text"].encode()
                            )
                            counts["signatures"] += len(rows)
                            counts["code_signatures"] += sum(
                                row["code"] for row in rows
                            )
                            counts.update(row_counts)
                        for bucket, rows in enumerate(signature_batches):
                            if rows:
                                writers[bucket].write_table(
                                    pa.Table.from_pylist(
                                        rows, schema=signature_schema()
                                    )
                                )
                        if locator_batch:
                            locator_writer.write_table(
                                pa.Table.from_pylist(
                                    locator_batch,
                                    schema=_transformed_locator_schema(),
                                )
                            )
                        row_offset += batch.num_rows
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
                            "event": "pleias_virtual_internal_rewrite_progress",
                            "shard_index": shard_index,
                            "complete_parents": parent_number,
                            "remaining_parents": len(parents) - parent_number,
                            "documents": counts["documents"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if (
                counts["documents"] != locator_rows
                or counts["candidate_deletion_chunks"] != decision_rows
            ):
                raise PleiasVirtualInternalRewriteSignatureError(
                    "virtual rewrite coverage differs"
                )
        except BaseException:
            for writer in writers:
                writer.close()
            locator_writer.close()
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise
        finally:
            locator_connection.close()
            decision_connection.close()
            selection_connection.close()
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
        "rows": counts["documents"],
        "bytes": locator_path.stat().st_size,
        "sha256": sha256_file(locator_path),
        "ordered_locator_digests_sha256": ordered_locators.hexdigest(),
    }
    payload = {
        "schema": SHARD_SCHEMA,
        "status": SHARD_STATUS,
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "virtual_signature_shard_receipt_sha256": virtual_receipt["receipt_sha256"],
            "selection_receipt_sha256": selection["receipt_sha256"],
            "semantic_decision_receipt_sha256": semantic_receipt["receipt_sha256"],
            "ordered_decision_receipts_sha256": canonical_sha256(decision_receipts),
            "ordered_parent_receipts_sha256": canonical_sha256(parent_receipts),
        },
        "counts": dict(sorted(counts.items())),
        "ordered_document_identities_sha256": ordered_documents.hexdigest(),
        "ordered_signature_digests_sha256": ordered.hexdigest(),
        "ordered_internal_transform_digests_sha256": ordered_transforms.hexdigest(),
        "hash_partition": {
            "buckets": HASH_BUCKETS,
            "key": "first_normalized_sha256_hex_nibble",
        },
        "outputs": outputs,
        "transformed_locators": locator,
        "complete_virtual_document_coverage": True,
        "complete_final_pleias_document_coverage": True,
        "full_document_benchmark_decontamination_complete": True,
        "pleias_internal_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
        "source_text_persisted": False,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(temporary_root / "receipt.json", payload)
    os.replace(temporary_root, output_root)
    return payload


def aggregate_virtual_rewrite(
    virtual_root: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Seal post-internal exact signatures for cross-source comparison."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise PleiasVirtualInternalRewriteSignatureError("aggregate arguments differ")
    virtual = _load_aggregate(virtual_root / "aggregate.json")
    if (
        virtual.get("status") != VIRTUAL_AGGREGATE_STATUS
        or virtual.get("complete_virtual_document_coverage") is not True
        or virtual.get("source_text_persisted") is not False
    ):
        raise PleiasVirtualInternalRewriteSignatureError("virtual aggregate differs")
    totals: Counter[str] = Counter()
    receipts = []
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_final_receipt(root / "receipt.json")
        outputs = receipt.get("outputs")
        locator = receipt.get("transformed_locators")
        if (
            receipt.get("status") != SHARD_STATUS
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("complete_final_pleias_document_coverage") is not True
            or receipt.get("source_text_persisted") is not False
            or not isinstance(outputs, list)
            or len(outputs) != HASH_BUCKETS
            or not isinstance(locator, dict)
        ):
            raise PleiasVirtualInternalRewriteSignatureError(
                "virtual rewrite shard differs"
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
                raise PleiasVirtualInternalRewriteSignatureError(
                    "virtual rewrite signature differs"
                )
            totals["signature_output_bytes"] += descriptor["bytes"]
        locator_path = root / locator.get("path", "")
        if (
            locator.get("rows") != receipt.get("counts", {}).get("documents")
            or not locator_path.is_file()
            or locator_path.is_symlink()
            or locator_path.stat().st_nlink != 1
            or locator_path.stat().st_size != locator.get("bytes")
            or sha256_file(locator_path) != locator.get("sha256")
        ):
            raise PleiasVirtualInternalRewriteSignatureError(
                "virtual transformed locator differs"
            )
        totals["locator_output_bytes"] += locator["bytes"]
        totals.update(receipt["counts"])
        receipts.append(receipt["receipt_sha256"])
    if totals["documents"] != virtual.get("totals", {}).get("retained_rows"):
        raise PleiasVirtualInternalRewriteSignatureError(
            "virtual rewrite aggregate coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "source": {"virtual_aggregate_receipt_sha256": virtual["receipt_sha256"]},
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "complete_final_pleias_document_coverage": True,
        "complete_virtual_document_coverage": True,
        "full_document_benchmark_decontamination_complete": True,
        "pleias_internal_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
        "source_text_persisted": False,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def _load_aggregate(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasVirtualInternalRewriteSignatureError("virtual aggregate is unsafe")
    value = json.loads(path.read_text())
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != VIRTUAL_AGGREGATE_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasVirtualInternalRewriteSignatureError(
            "virtual aggregate receipt differs"
        )
    return value


def _load_final_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasVirtualInternalRewriteSignatureError("final receipt is unsafe")
    value = json.loads(path.read_text())
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != SHARD_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasVirtualInternalRewriteSignatureError("final receipt differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--selection-root", type=Path, required=True)
    shard.add_argument("--semantic-decision", type=Path, required=True)
    shard.add_argument("--virtual-root", type=Path, required=True)
    shard.add_argument("--decision-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--virtual-root", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.selection_root,
            args.semantic_decision,
            args.virtual_root,
            args.decision_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate_virtual_rewrite(
            args.virtual_root,
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
