"""Apply verified PleIAs subdocument deletions and upload rewritten shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import _WORD
from sai.data.frequency_length_subdocument_deduplication import (
    DEFAULT_DELETE_CHARACTERS,
    DEFAULT_SEGMENT_CHARACTERS,
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_production_materializer import (
    DESTINATION_REPOSITORY,
    _load_signed,
    upload_verified,
)
from sai.data.pleias_production_materializer import (
    SHARD_SCHEMA as MATERIALIZED_SCHEMA,
)
from sai.data.pleias_subdocument_decision import _DELETE, _records
from sai.data.pleias_subdocument_decision import (
    SCHEMA as DECISION_SCHEMA,
)
from sai.data.pleias_subdocument_signature import HASH_BUCKETS, _download
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-subdocument-rewritten-shard-v1"
OUTPUT_SCHEMA = "sai-pleias-subdocument-deduplicated-candidate-v1"
DESTINATION_PREFIX = "final/nontraining/pleias/20260826-r1"


class PleiasSubdocumentRewriteError(RuntimeError):
    """Deletion custody, chunk replay, rewrite, or remote identity differs."""


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasSubdocumentRewriteError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("source_id", pa.string()),
            ("source_repository", pa.string()),
            ("source_revision", pa.string()),
            ("source_path", pa.string()),
            ("source_parent_sha256", pa.string()),
            ("source_row_index", pa.int64()),
            ("source_row_identity_sha256", pa.string()),
            ("identifier", pa.string()),
            ("collection", pa.string()),
            ("open_type", pa.string()),
            ("license", pa.string()),
            ("language", pa.string()),
            ("source_word_count", pa.int64()),
            ("source_token_count", pa.int64()),
            ("semantic_stratum", pa.string()),
            ("semantic_quality_floor_milli", pa.int32()),
            ("semantic_quality_mean_milli", pa.int32()),
            ("semantic_difficulty_mean_milli", pa.int32()),
            ("semantic_prerequisite_burden_mean_milli", pa.int32()),
            ("semantic_curriculum_phase", pa.string()),
            ("semantic_domains", pa.list_(pa.string())),
            ("semantic_recurring_concepts", pa.list_(pa.string())),
            ("semantic_recurring_prerequisites", pa.list_(pa.string())),
            ("word_count", pa.int64()),
            ("token_count_requires_recomputation", pa.bool_()),
            ("pre_dedup_content_sha256", pa.string()),
            ("content_sha256", pa.string()),
            ("subdocument_transform_sha256", pa.string()),
            ("text", pa.string()),
            ("training_ready", pa.bool_()),
        ]
    )


def rewrite_candidate(
    candidate: dict[str, Any],
    source_row_index: int,
    decisions: list[tuple[str, int, int, int, str, int, int]],
    delete_characters: int = DEFAULT_DELETE_CHARACTERS,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Replay exact chunk decisions and preserve coherence for tiny deletions."""

    text = candidate.get("text")
    identity = candidate.get("source_row_identity_sha256")
    content_sha256 = candidate.get("content_sha256")
    source_word_count = candidate.get("word_count")
    source_token_count = candidate.get("token_count")
    semantic_stratum = candidate.get("semantic_stratum")
    semantic_quality_floor = candidate.get("semantic_quality_floor_milli")
    semantic_quality_mean = candidate.get("semantic_quality_mean_milli")
    semantic_difficulty = candidate.get("semantic_difficulty_mean_milli")
    semantic_burden = candidate.get("semantic_prerequisite_burden_mean_milli")
    semantic_phase = candidate.get("semantic_curriculum_phase")
    semantic_domains = candidate.get("semantic_domains")
    semantic_concepts = candidate.get("semantic_recurring_concepts")
    semantic_prerequisites = candidate.get("semantic_recurring_prerequisites")
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("training_ready") is not False
        or not isinstance(text, str)
        or not isinstance(identity, str)
        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
        or not isinstance(source_word_count, int)
        or isinstance(source_word_count, bool)
        or source_word_count <= 0
        or not isinstance(source_token_count, int)
        or isinstance(source_token_count, bool)
        or source_token_count <= 0
        or not isinstance(semantic_stratum, str)
        or not semantic_stratum
        or isinstance(semantic_quality_floor, bool)
        or not isinstance(semantic_quality_floor, int)
        or not 0 <= semantic_quality_floor <= 10_000
        or isinstance(semantic_quality_mean, bool)
        or not isinstance(semantic_quality_mean, int)
        or not semantic_quality_floor <= semantic_quality_mean <= 10_000
        or isinstance(semantic_difficulty, bool)
        or not isinstance(semantic_difficulty, int)
        or not 0 <= semantic_difficulty <= 4_000
        or isinstance(semantic_burden, bool)
        or not isinstance(semantic_burden, int)
        or not 0 <= semantic_burden <= 4_000
        or not isinstance(semantic_phase, str)
        or not semantic_phase
        or not isinstance(semantic_domains, list)
        or not semantic_domains
        or any(not isinstance(value, str) or not value for value in semantic_domains)
        or not isinstance(semantic_concepts, list)
        or any(not isinstance(value, str) or not value for value in semantic_concepts)
        or not isinstance(semantic_prerequisites, list)
        or any(
            not isinstance(value, str) or not value
            for value in semantic_prerequisites
        )
        or delete_characters <= 0
    ):
        raise PleiasSubdocumentRewriteError("rewrite candidate differs")
    collection = candidate.get("collection")
    code_document = isinstance(collection, str) and "github" in collection.casefold()
    chunks = segment_subdocuments(
        text,
        minimum_characters=DEFAULT_SEGMENT_CHARACTERS,
        code_document=code_document,
    )
    by_chunk = {}
    decision_digests = []
    for (
        expected_identity,
        chunk_index,
        character_start,
        character_end,
        normalized_sha256,
        frequency,
        budget,
    ) in decisions:
        if (
            expected_identity != identity
            or chunk_index in by_chunk
            or not 0 <= chunk_index < len(chunks)
            or frequency <= budget
            or budget <= 0
        ):
            raise PleiasSubdocumentRewriteError("rewrite decision differs")
        chunk = chunks[chunk_index]
        normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
        decision = {
            "source_row_index": source_row_index,
            "document_identity_sha256": identity,
            "chunk_index": chunk_index,
            "character_start": character_start,
            "character_end": character_end,
            "normalized_sha256": normalized_sha256,
            "frequency": frequency,
            "budget": budget,
        }
        if (
            chunk["character_start"] != character_start
            or chunk["character_end"] != character_end
            or hashlib.sha256(normalized.encode()).hexdigest() != normalized_sha256
        ):
            raise PleiasSubdocumentRewriteError("rewrite chunk replay differs")
        by_chunk[chunk_index] = chunk
        decision_digests.append(canonical_sha256(decision))
    delete_total = sum(len(chunk["text"]) for chunk in by_chunk.values())
    counts = {
        "candidate_deletion_chunks": len(by_chunk),
        "candidate_deletion_characters": delete_total,
        "deleted_chunks": 0,
        "deleted_characters": 0,
        "coherence_restored_chunks": 0,
    }
    rewritten = text
    if by_chunk and delete_total >= delete_characters:
        spans = sorted(
            (chunk["character_start"], chunk["character_end"])
            for chunk in by_chunk.values()
        )
        if any(
            first[1] > second[0]
            for first, second in zip(spans, spans[1:], strict=False)
        ):
            raise PleiasSubdocumentRewriteError("rewrite spans overlap")
        parts = []
        cursor = 0
        for start, end in spans:
            parts.append(text[cursor:start])
            cursor = end
        parts.append(text[cursor:])
        candidate_text = "".join(parts)
        if candidate_text.strip():
            rewritten = candidate_text
            counts["deleted_chunks"] = len(by_chunk)
            counts["deleted_characters"] = delete_total
        else:
            counts["coherence_restored_chunks"] = len(by_chunk)
    elif by_chunk:
        counts["coherence_restored_chunks"] = len(by_chunk)
    post_sha256 = hashlib.sha256(rewritten.encode()).hexdigest()
    transform = {
        "source_row_identity_sha256": identity,
        "pre_dedup_content_sha256": content_sha256,
        "ordered_decision_digests_sha256": canonical_sha256(decision_digests),
        "post_dedup_content_sha256": post_sha256,
    }
    result = {
        key: value
        for key, value in candidate.items()
        if key not in {"schema", "content_sha256", "text", "word_count", "token_count"}
    }
    result.update(
        {
            "schema": OUTPUT_SCHEMA,
            "source_word_count": source_word_count,
            "source_token_count": source_token_count,
            "word_count": len(_WORD.findall(rewritten)),
            "token_count_requires_recomputation": True,
            "pre_dedup_content_sha256": content_sha256,
            "content_sha256": post_sha256,
            "subdocument_transform_sha256": canonical_sha256(transform),
            "text": rewritten,
            "training_ready": False,
        }
    )
    return result, counts


def _decision_database(
    decision_root: Path,
    shard_index: int,
    logical_shards: int,
    database_path: Path,
) -> tuple[sqlite3.Connection, list[str], int]:
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE deletions ("
        "source_row_index INTEGER NOT NULL, document_identity_sha256 TEXT NOT NULL, "
        "chunk_index INTEGER NOT NULL, character_start INTEGER NOT NULL, "
        "character_end INTEGER NOT NULL, normalized_sha256 TEXT NOT NULL, "
        "frequency INTEGER NOT NULL, budget INTEGER NOT NULL, "
        "PRIMARY KEY(source_row_index, chunk_index)"
        ") WITHOUT ROWID"
    )
    receipts = []
    total = 0
    try:
        for bucket_index in range(HASH_BUCKETS):
            root = decision_root / "buckets" / f"bucket_{bucket_index:02x}"
            receipt = _load_signed(root / "receipt.json", DECISION_SCHEMA)
            descriptors = receipt.get("deletions")
            descriptor = (
                descriptors[shard_index]
                if isinstance(descriptors, list) and len(descriptors) == logical_shards
                else {}
            )
            path = root / "deletions" / descriptor.get("path", "")
            if (
                receipt.get("hash_bucket", {}).get("index") != bucket_index
                or receipt.get("hash_bucket", {}).get("buckets") != HASH_BUCKETS
                or receipt.get("decision_contains_source_text") is not False
                or descriptor.get("shard_index") != shard_index
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise PleiasSubdocumentRewriteError("deletion descriptor differs")
            rows = 0
            with path.open("rb") as handle:
                for record in _records(handle, _DELETE):
                    if record[0] != shard_index:
                        raise PleiasSubdocumentRewriteError("deletion shard differs")
                    connection.execute(
                        "INSERT INTO deletions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            record[1],
                            record[2].hex(),
                            record[3],
                            record[4],
                            record[5],
                            record[6].hex(),
                            record[7],
                            record[8],
                        ),
                    )
                    rows += 1
                    total += 1
            if rows != descriptor.get("rows"):
                raise PleiasSubdocumentRewriteError("deletion coverage differs")
            connection.commit()
            receipts.append(receipt["receipt_sha256"])
        connection.execute("CREATE INDEX deletions_row ON deletions(source_row_index)")
        connection.commit()
    except BaseException:
        connection.close()
        raise
    return connection, receipts, total


def run_shard(
    materialized_root: Path,
    decision_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Rewrite one remote shard, verify its final upload, and remove local text."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasSubdocumentRewriteError("rewrite arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasSubdocumentRewriteError("pyarrow is required") from error
    materialized = _load_signed(
        materialized_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
        MATERIALIZED_SCHEMA,
    )
    if (
        materialized.get("logical_shards") != logical_shards
        or materialized.get("shard_index") != shard_index
        or materialized.get("full_document_benchmark_decontamination_complete")
        is not True
    ):
        raise PleiasSubdocumentRewriteError("materialized source differs")
    counts: Counter[str] = Counter()
    ordered_transforms = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        prefix="sai-pleias-subdocument-rewrite-", dir=scratch_root
    ) as directory:
        scratch = Path(directory)
        temporary = scratch / f"rewrite-partial-{uuid.uuid4().hex}.parquet"
        connection, decision_receipts, decision_rows = _decision_database(
            decision_root,
            shard_index,
            logical_shards,
            scratch / "deletions.sqlite3",
        )
        source_path = _download(materialized, token, scratch)
        parquet = pq.ParquetFile(source_path)
        writer = pq.ParquetWriter(temporary, _schema(), compression="zstd")
        try:
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                output_rows = []
                for relative, candidate in enumerate(batch.to_pylist()):
                    source_row_index = row_offset + relative
                    decisions = connection.execute(
                        "SELECT document_identity_sha256, chunk_index, "
                        "character_start, character_end, normalized_sha256, "
                        "frequency, budget FROM deletions WHERE source_row_index=? "
                        "ORDER BY chunk_index",
                        (source_row_index,),
                    ).fetchall()
                    result, row_counts = rewrite_candidate(
                        candidate, source_row_index, decisions
                    )
                    output_rows.append(result)
                    counts["documents"] += 1
                    counts["input_text_utf8_bytes"] += len(candidate["text"].encode())
                    counts["output_text_utf8_bytes"] += len(result["text"].encode())
                    for key, value in row_counts.items():
                        counts[key] += value
                    ordered_transforms.update(
                        bytes.fromhex(result["subdocument_transform_sha256"])
                    )
                if output_rows:
                    writer.write_table(
                        pa.Table.from_pylist(output_rows, schema=_schema())
                    )
                row_offset += batch.num_rows
            if row_offset != parquet.metadata.num_rows:
                raise PleiasSubdocumentRewriteError("rewrite document coverage differs")
            if counts["candidate_deletion_chunks"] != decision_rows:
                raise PleiasSubdocumentRewriteError(
                    "rewrite decision accounting differs"
                )
        except BaseException:
            writer.close()
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        writer.close()
        connection.close()
        remote_path = (
            f"{DESTINATION_PREFIX}/"
            f"shard-{shard_index:05d}-of-{logical_shards:05d}.parquet"
        )
        remote = upload_verified(
            temporary,
            remote_path,
            token,
            repository=DESTINATION_REPOSITORY,
        )
    if counts["documents"] != materialized.get("counts", {}).get("retained_rows", 0):
        raise PleiasSubdocumentRewriteError("rewrite source count differs")
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_subdocument_rewritten_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "materialized_receipt_sha256": materialized["receipt_sha256"],
            "ordered_decision_receipts_sha256": canonical_sha256(decision_receipts),
        },
        "policy": {
            "minimum_segment_characters": DEFAULT_SEGMENT_CHARACTERS,
            "minimum_delete_characters_per_document": DEFAULT_DELETE_CHARACTERS,
            "restore_empty_document": True,
            "restore_below_delete_threshold": True,
        },
        "counts": dict(sorted(counts.items())),
        "ordered_transform_digests_sha256": ordered_transforms.hexdigest(),
        "remote_output": remote,
        "local_payload_removed_after_remote_verification": True,
        "pleias_global_subdocument_rewrite_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output_root.mkdir(parents=True)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = run_shard(
        args.materialized_root,
        args.decision_root,
        args.output_root,
        args.logical_shards,
        args.shard_index,
        os.environ.get(args.token_env, ""),
        args.scratch_root,
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
