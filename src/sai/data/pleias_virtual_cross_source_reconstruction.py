"""Replay both deletion layers and seal final source-safe PleIAs locators."""

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
from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as CROSS_DECISION_AGGREGATE_SCHEMA,
)
from sai.data.cross_source_subdocument_rewrite import decision_database
from sai.data.foundation_source_split import POLICY_SHA256 as SPLIT_POLICY_SHA256
from sai.data.pleias_bounded_mechanical_candidates import _download
from sai.data.pleias_cross_source_subdocument_rewrite import (
    COMPONENT_PRIORITY,
    rewrite_row,
)
from sai.data.pleias_cross_source_subdocument_rewrite import (
    OUTPUT_SCHEMA as CROSS_OUTPUT_SCHEMA,
)
from sai.data.pleias_final_subdocument_signature import (
    AGGREGATE_SCHEMA as INTERNAL_SIGNATURE_AGGREGATE_SCHEMA,
)
from sai.data.pleias_final_subdocument_signature import COMPONENT
from sai.data.pleias_final_subdocument_signature import (
    SHARD_SCHEMA as INTERNAL_SIGNATURE_SHARD_SCHEMA,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_production_materializer import (
    _load_signed,
    _selection_database,
    _semantic_metadata,
    replay_selected_row,
)
from sai.data.pleias_subdocument_rewrite import _decision_database, rewrite_candidate
from sai.data.pleias_virtual_internal_rewrite_signature import (
    AGGREGATE_STATUS as INTERNAL_AGGREGATE_STATUS,
)
from sai.data.pleias_virtual_internal_rewrite_signature import (
    SHARD_STATUS as INTERNAL_SHARD_STATUS,
)
from sai.data.pleias_virtual_internal_rewrite_signature import (
    TRANSFORMED_LOCATOR_SCHEMA,
)
from sai.data.pleias_virtual_subdocument_signature import _selected_rows
from sai.data.token_stream import canonical_sha256, sha256_file

LOCATOR_SCHEMA = "sai-pleias-virtual-final-reconstruction-locator-v1"
SHARD_SCHEMA = "sai-pleias-virtual-final-reconstruction-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-virtual-final-reconstruction-aggregate-v1"
SHARD_STATUS = "complete_nontraining_pleias_virtual_final_reconstruction_shard"
AGGREGATE_STATUS = "complete_nontraining_pleias_virtual_final_reconstruction"


class PleiasVirtualCrossSourceReconstructionError(RuntimeError):
    """Pinned source, deletion replay, final identity, or coverage differs."""


def _locator_schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasVirtualCrossSourceReconstructionError(
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
            ("post_internal_content_sha256", pa.string()),
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
            ("cross_source_subdocument_transform_sha256", pa.string()),
            ("source_group_sha256", pa.string()),
            ("source_group_bucket", pa.int32()),
            ("corpus_split", pa.string()),
            ("source_split_policy_sha256", pa.string()),
            ("locator_sha256", pa.string()),
            ("training_ready", pa.bool_()),
        ]
    )


def final_locator_row(
    internal_locator: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Bind an exact two-layer transformation without retaining final text."""

    text = result.get("text")
    unsigned_internal = {
        key: value for key, value in internal_locator.items() if key != "locator_sha256"
    }
    if (
        internal_locator.get("schema") != TRANSFORMED_LOCATOR_SCHEMA
        or internal_locator.get("locator_sha256") != canonical_sha256(unsigned_internal)
        or result.get("schema") != CROSS_OUTPUT_SCHEMA
        or result.get("training_ready") is not False
        or result.get("token_count_requires_recomputation") is not True
        or not isinstance(text, str)
        or not text
        or result.get("source_row_identity_sha256")
        != internal_locator.get("source_row_identity_sha256")
        or result.get("pre_cross_source_content_sha256")
        != internal_locator.get("content_sha256")
        or result.get("subdocument_transform_sha256")
        != internal_locator.get("internal_subdocument_transform_sha256")
        or result.get("source_split_policy_sha256") != SPLIT_POLICY_SHA256
        or hashlib.sha256(text.encode()).hexdigest() != result.get("content_sha256")
    ):
        raise PleiasVirtualCrossSourceReconstructionError(
            "final locator source differs"
        )
    row = {
        "schema": LOCATOR_SCHEMA,
        "virtual_row_index": internal_locator["virtual_row_index"],
        "source_repository": internal_locator["source_repository"],
        "source_revision": internal_locator["source_revision"],
        "source_path": internal_locator["source_path"],
        "source_parent_sha256": internal_locator["source_parent_sha256"],
        "source_row_index": internal_locator["source_row_index"],
        "source_row_identity_sha256": internal_locator["source_row_identity_sha256"],
        "pre_internal_content_sha256": internal_locator["pre_internal_content_sha256"],
        "post_internal_content_sha256": internal_locator["content_sha256"],
        "content_sha256": result["content_sha256"],
        "source_text_utf8_bytes": internal_locator["source_text_utf8_bytes"],
        "output_text_utf8_bytes": len(text.encode()),
        "source_text_characters": internal_locator["source_text_characters"],
        "output_text_characters": len(text),
        "source_word_count": internal_locator["source_word_count"],
        "output_word_count": result["word_count"],
        "source_token_count": internal_locator["source_token_count"],
        "token_count_requires_recomputation": True,
        "collection": internal_locator["collection"],
        "open_type": internal_locator["open_type"],
        "license": internal_locator["license"],
        "language": internal_locator["language"],
        "semantic_stratum": internal_locator["semantic_stratum"],
        "semantic_quality_floor_milli": internal_locator[
            "semantic_quality_floor_milli"
        ],
        "semantic_quality_mean_milli": internal_locator["semantic_quality_mean_milli"],
        "semantic_difficulty_mean_milli": internal_locator[
            "semantic_difficulty_mean_milli"
        ],
        "semantic_prerequisite_burden_mean_milli": internal_locator[
            "semantic_prerequisite_burden_mean_milli"
        ],
        "semantic_curriculum_phase": internal_locator["semantic_curriculum_phase"],
        "semantic_domains": internal_locator["semantic_domains"],
        "semantic_recurring_concepts": internal_locator["semantic_recurring_concepts"],
        "semantic_recurring_prerequisites": internal_locator[
            "semantic_recurring_prerequisites"
        ],
        "code_document": internal_locator["code_document"],
        "internal_subdocument_transform_sha256": internal_locator[
            "internal_subdocument_transform_sha256"
        ],
        "cross_source_subdocument_transform_sha256": result[
            "cross_source_subdocument_transform_sha256"
        ],
        "source_group_sha256": result["source_group_sha256"],
        "source_group_bucket": result["source_group_bucket"],
        "corpus_split": result["corpus_split"],
        "source_split_policy_sha256": result["source_split_policy_sha256"],
        "training_ready": False,
    }
    row["locator_sha256"] = canonical_sha256(row)
    return row


def _internal_locator_database(
    internal_root: Path,
    shard_index: int,
    logical_shards: int,
    database_path: Path,
) -> tuple[sqlite3.Connection, dict[str, Any], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualCrossSourceReconstructionError(
            "pyarrow is required"
        ) from error
    root = internal_root / "shards" / f"shard_{shard_index:05d}"
    receipt = _load_signed(root / "receipt.json", INTERNAL_SIGNATURE_SHARD_SCHEMA)
    descriptor = receipt.get("transformed_locators")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        receipt.get("status") != INTERNAL_SHARD_STATUS
        or receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("complete_final_pleias_document_coverage") is not True
        or receipt.get("source_text_persisted") is not False
        or not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PleiasVirtualCrossSourceReconstructionError(
            "internal locator descriptor differs"
        )
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE locators ("
        "virtual_row_index INTEGER PRIMARY KEY, source_path TEXT NOT NULL, "
        "source_row_index INTEGER NOT NULL, source_row_identity_sha256 TEXT NOT NULL, "
        "pre_internal_content_sha256 TEXT NOT NULL, content_sha256 TEXT NOT NULL, "
        "internal_transform_sha256 TEXT NOT NULL, locator_json TEXT NOT NULL, "
        "UNIQUE(source_path, source_row_index))"
    )
    rows = 0
    ordered = hashlib.sha256()
    try:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1024, use_threads=False
        ):
            for locator in batch.to_pylist():
                unsigned = {
                    key: value
                    for key, value in locator.items()
                    if key != "locator_sha256"
                }
                if (
                    locator.get("schema") != TRANSFORMED_LOCATOR_SCHEMA
                    or locator.get("training_ready") is not False
                    or locator.get("locator_sha256") != canonical_sha256(unsigned)
                    or locator.get("virtual_row_index") != rows
                ):
                    raise PleiasVirtualCrossSourceReconstructionError(
                        "internal locator row differs"
                    )
                connection.execute(
                    "INSERT INTO locators VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        locator["virtual_row_index"],
                        locator["source_path"],
                        locator["source_row_index"],
                        locator["source_row_identity_sha256"],
                        locator["pre_internal_content_sha256"],
                        locator["content_sha256"],
                        locator["internal_subdocument_transform_sha256"],
                        json.dumps(locator, sort_keys=True, separators=(",", ":")),
                    ),
                )
                ordered.update(bytes.fromhex(locator["locator_sha256"]))
                rows += 1
        connection.commit()
        if rows != descriptor.get("rows") or ordered.hexdigest() != descriptor.get(
            "ordered_locator_digests_sha256"
        ):
            raise PleiasVirtualCrossSourceReconstructionError(
                "internal locator coverage differs"
            )
    except BaseException:
        connection.close()
        raise
    return connection, receipt, rows


def _decision_rows(connection: sqlite3.Connection, row_index: int) -> list[Any]:
    return connection.execute(
        "SELECT document_identity_sha256, chunk_index, character_start, "
        "character_end, normalized_sha256, frequency, budget FROM deletions "
        "WHERE source_row_index=? ORDER BY chunk_index",
        (row_index,),
    ).fetchall()


def run_shard(
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    internal_root: Path,
    internal_decision_root: Path,
    cross_decision_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct one partition and persist only final locators and accounting."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasVirtualCrossSourceReconstructionError(
            "final reconstruction arguments differ"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualCrossSourceReconstructionError(
            "pyarrow is required"
        ) from error
    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    selection, selection_path = _selection_database(selection_root)
    semantic_by_stratum, semantic_receipt = _semantic_metadata(semantic_decision_path)
    selection_connection = sqlite3.connect(
        f"file:{selection_path.resolve()}?mode=ro", uri=True
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = (
        output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    )
    temporary_root.mkdir()
    locator_path = temporary_root / "final-locators.parquet"
    writer = pq.ParquetWriter(locator_path, _locator_schema(), compression="zstd")
    counts: Counter[str] = Counter()
    ordered_locators = hashlib.sha256()
    ordered_documents = hashlib.sha256()
    ordered_internal = hashlib.sha256()
    ordered_cross = hashlib.sha256()
    parent_receipts = []
    connections: list[sqlite3.Connection] = [selection_connection]
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-pleias-virtual-final-state-", dir=scratch_root
        ) as state_directory:
            state = Path(state_directory)
            internal_locators, internal_receipt, expected_documents = (
                _internal_locator_database(
                    internal_root,
                    shard_index,
                    logical_shards,
                    state / "internal-locators.sqlite3",
                )
            )
            connections.append(internal_locators)
            internal_decisions, internal_receipts, expected_internal_decisions = (
                _decision_database(
                    internal_decision_root,
                    shard_index,
                    logical_shards,
                    state / "internal-deletions.sqlite3",
                )
            )
            connections.append(internal_decisions)
            cross_decisions, cross_receipts, expected_cross_decisions = (
                decision_database(
                    cross_decision_root,
                    COMPONENT,
                    COMPONENT_PRIORITY,
                    shard_index,
                    logical_shards,
                    state / "cross-deletions.sqlite3",
                )
            )
            connections.append(cross_decisions)
            for parent_number, parent in enumerate(parents, start=1):
                selected_rows = _selected_rows(
                    selection_connection, parent["source_path"]
                )
                if not selected_rows:
                    counts["parents_without_selected_rows"] += 1
                    continue
                by_index = {row[0]: row[1:] for row in selected_rows}
                seen = set()
                with tempfile.TemporaryDirectory(
                    prefix="sai-pleias-virtual-final-source-", dir=scratch_root
                ) as directory:
                    source = _download(parent, token, Path(directory))
                    parquet = pq.ParquetFile(source)
                    row_offset = 0
                    for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                        outputs = []
                        for relative, source_row in enumerate(batch.to_pylist()):
                            source_row_index = row_offset + relative
                            expected = by_index.get(source_row_index)
                            if expected is None:
                                continue
                            seen.add(source_row_index)
                            match = internal_locators.execute(
                                "SELECT virtual_row_index, source_row_identity_sha256, "
                                "pre_internal_content_sha256, content_sha256, "
                                "internal_transform_sha256, locator_json FROM locators "
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
                            (
                                virtual_row_index,
                                identity,
                                pre_internal,
                                post_internal,
                                internal_transform,
                                encoded_locator,
                            ) = match
                            if (
                                candidate["source_row_identity_sha256"] != identity
                                or candidate["content_sha256"] != pre_internal
                            ):
                                raise PleiasVirtualCrossSourceReconstructionError(
                                    "reconstructed source identity differs"
                                )
                            internally_rewritten, internal_counts = rewrite_candidate(
                                candidate,
                                virtual_row_index,
                                _decision_rows(internal_decisions, virtual_row_index),
                            )
                            if (
                                internally_rewritten["content_sha256"] != post_internal
                                or internally_rewritten["subdocument_transform_sha256"]
                                != internal_transform
                            ):
                                raise PleiasVirtualCrossSourceReconstructionError(
                                    "internal rewrite replay differs"
                                )
                            final, cross_counts = rewrite_row(
                                internally_rewritten,
                                virtual_row_index,
                                _decision_rows(cross_decisions, virtual_row_index),
                            )
                            locator = final_locator_row(
                                json.loads(encoded_locator), final
                            )
                            outputs.append(locator)
                            ordered_locators.update(
                                bytes.fromhex(locator["locator_sha256"])
                            )
                            ordered_documents.update(bytes.fromhex(identity))
                            ordered_internal.update(bytes.fromhex(internal_transform))
                            ordered_cross.update(
                                bytes.fromhex(
                                    locator["cross_source_subdocument_transform_sha256"]
                                )
                            )
                            counts["documents"] += 1
                            counts["source_text_utf8_bytes"] += len(
                                candidate["text"].encode()
                            )
                            counts["post_internal_text_utf8_bytes"] += len(
                                internally_rewritten["text"].encode()
                            )
                            counts["output_text_utf8_bytes"] += len(
                                final["text"].encode()
                            )
                            counts[f"split::{locator['corpus_split']}::documents"] += 1
                            counts[
                                f"split::{locator['corpus_split']}::text_utf8_bytes"
                            ] += len(final["text"].encode())
                            counts[
                                "curriculum_phase::"
                                f"{locator['semantic_curriculum_phase']}::documents"
                            ] += 1
                            counts[
                                "curriculum_phase::"
                                f"{locator['semantic_curriculum_phase']}::text_utf8_bytes"
                            ] += len(final["text"].encode())
                            counts[
                                f"semantic_stratum::{locator['semantic_stratum']}::documents"
                            ] += 1
                            counts[
                                f"semantic_stratum::{locator['semantic_stratum']}::text_utf8_bytes"
                            ] += len(final["text"].encode())
                            counts[
                                "quality_floor_milli::"
                                f"{locator['semantic_quality_floor_milli']}::documents"
                            ] += 1
                            counts[
                                "difficulty_mean_milli::"
                                f"{locator['semantic_difficulty_mean_milli']}::documents"
                            ] += 1
                            for domain in locator["semantic_domains"]:
                                counts[f"semantic_domain::{domain}::documents"] += 1
                                counts[
                                    f"semantic_domain::{domain}::text_utf8_bytes"
                                ] += len(final["text"].encode())
                            for key, value in internal_counts.items():
                                counts[f"internal::{key}"] += value
                            for key, value in cross_counts.items():
                                counts[f"cross::{key}"] += value
                        if outputs:
                            writer.write_table(
                                pa.Table.from_pylist(outputs, schema=_locator_schema())
                            )
                        row_offset += batch.num_rows
                if seen != set(by_index):
                    raise PleiasVirtualCrossSourceReconstructionError(
                        "selected source coverage differs"
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
                            "event": "pleias_virtual_final_reconstruction_progress",
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
                counts["documents"] != expected_documents
                or counts["internal::candidate_deletion_chunks"]
                != expected_internal_decisions
                or counts["cross::candidate_deletion_chunks"]
                != expected_cross_decisions
            ):
                raise PleiasVirtualCrossSourceReconstructionError(
                    "final reconstruction coverage differs"
                )
        writer.close()
        writer = None
        descriptor = {
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
                "manifest_sha256": sha256_file(manifest_path),
                "selection_receipt_sha256": selection["receipt_sha256"],
                "semantic_decision_receipt_sha256": semantic_receipt["receipt_sha256"],
                "internal_rewrite_shard_receipt_sha256": internal_receipt[
                    "receipt_sha256"
                ],
                "ordered_internal_decision_receipts_sha256": canonical_sha256(
                    internal_receipts
                ),
                "ordered_cross_decision_receipts_sha256": canonical_sha256(
                    cross_receipts
                ),
                "ordered_parent_receipts_sha256": canonical_sha256(parent_receipts),
            },
            "counts": dict(sorted(counts.items())),
            "ordered_document_identities_sha256": ordered_documents.hexdigest(),
            "ordered_internal_transform_digests_sha256": ordered_internal.hexdigest(),
            "ordered_cross_transform_digests_sha256": ordered_cross.hexdigest(),
            "final_locators": descriptor,
            "complete_final_pleias_document_coverage": True,
            "benchmark_decontamination_complete": True,
            "pleias_internal_subdocument_deduplication_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "source_disjoint_split_complete": True,
            "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
            "physical_train_development_partition_complete": True,
            "semantic_quality_metadata_complete": True,
            "curriculum_metadata_complete": True,
            "source_text_persisted": False,
            "token_count_requires_recomputation": True,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(temporary_root / "receipt.json", payload)
        os.replace(temporary_root, output_root)
        return payload
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    finally:
        for connection in reversed(connections):
            connection.close()


def aggregate(
    internal_root: Path,
    cross_decision_root: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify all final virtual locators and exact two-layer accounting."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise PleiasVirtualCrossSourceReconstructionError(
            "final reconstruction aggregate arguments differ"
        )
    internal = _load_signed(
        internal_root / "aggregate.json",
        INTERNAL_SIGNATURE_AGGREGATE_SCHEMA,
    )
    cross = _load_signed(
        cross_decision_root / "aggregate.json", CROSS_DECISION_AGGREGATE_SCHEMA
    )
    if (
        internal.get("status") != INTERNAL_AGGREGATE_STATUS
        or internal.get("complete_final_pleias_document_coverage") is not True
        or internal.get("source_text_persisted") is not False
        or cross.get("cross_source_subdocument_decision_complete") is not True
        or cross.get("decision_contains_source_text") is not False
    ):
        raise PleiasVirtualCrossSourceReconstructionError(
            "final reconstruction aggregate source differs"
        )
    expected_cross = cross.get("totals", {}).get(
        f"component::{COMPONENT}::deletion_occurrences", 0
    )
    totals: Counter[str] = Counter()
    receipts = []
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        descriptor = receipt.get("final_locators")
        path = (
            root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
        )
        if (
            receipt.get("status") != SHARD_STATUS
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("complete_final_pleias_document_coverage") is not True
            or receipt.get("cross_source_subdocument_deduplication_complete")
            is not True
            or receipt.get("source_disjoint_split_policy_sha256") != SPLIT_POLICY_SHA256
            or receipt.get("physical_train_development_partition_complete") is not True
            or receipt.get("semantic_quality_metadata_complete") is not True
            or receipt.get("curriculum_metadata_complete") is not True
            or receipt.get("source_text_persisted") is not False
            or not isinstance(descriptor, dict)
            or descriptor.get("rows") != receipt.get("counts", {}).get("documents")
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise PleiasVirtualCrossSourceReconstructionError(
                "final reconstruction shard differs"
            )
        totals.update(receipt["counts"])
        totals["locator_output_bytes"] += descriptor["bytes"]
        receipts.append(receipt["receipt_sha256"])

    def dimension(prefix: str) -> int:
        return sum(
            value
            for key, value in totals.items()
            if key.startswith(prefix) and key.endswith("::documents")
        )

    if (
        totals["documents"] != internal.get("totals", {}).get("documents")
        or totals["cross::candidate_deletion_chunks"] != expected_cross
        or totals["split::train::documents"] + totals["split::development::documents"]
        != totals["documents"]
        or totals["split::train::text_utf8_bytes"]
        + totals["split::development::text_utf8_bytes"]
        != totals["output_text_utf8_bytes"]
        or dimension("semantic_stratum::") != totals["documents"]
        or dimension("quality_floor_milli::") != totals["documents"]
        or dimension("difficulty_mean_milli::") != totals["documents"]
        or dimension("curriculum_phase::") != totals["documents"]
        or dimension("semantic_domain::") < totals["documents"]
    ):
        raise PleiasVirtualCrossSourceReconstructionError(
            "final reconstruction aggregate coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "source": {
            "internal_aggregate_receipt_sha256": internal["receipt_sha256"],
            "cross_decision_aggregate_receipt_sha256": cross["receipt_sha256"],
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "complete_final_pleias_document_coverage": True,
        "benchmark_decontamination_complete": True,
        "pleias_internal_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": True,
        "source_disjoint_split_complete": True,
        "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
        "physical_train_development_partition_complete": True,
        "semantic_quality_metadata_complete": True,
        "curriculum_metadata_complete": True,
        "source_text_persisted": False,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--selection-root", type=Path, required=True)
    shard.add_argument("--semantic-decision", type=Path, required=True)
    shard.add_argument("--internal-root", type=Path, required=True)
    shard.add_argument("--internal-decision-root", type=Path, required=True)
    shard.add_argument("--cross-decision-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--internal-root", type=Path, required=True)
    combine.add_argument("--cross-decision-root", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.selection_root,
            args.semantic_decision,
            args.internal_root,
            args.internal_decision_root,
            args.cross_decision_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate(
            args.internal_root,
            args.cross_decision_root,
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
