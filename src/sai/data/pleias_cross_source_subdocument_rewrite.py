"""Apply verified cross-source deletions to final PleIAs shards."""

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
from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as DECISION_AGGREGATE_SCHEMA,
)
from sai.data.cross_source_subdocument_rewrite import (
    decision_database,
    rewrite_text,
)
from sai.data.decontamination import _WORD
from sai.data.foundation_source_split import (
    POLICY as SPLIT_POLICY,
)
from sai.data.foundation_source_split import (
    POLICY_SHA256 as SPLIT_POLICY_SHA256,
)
from sai.data.foundation_source_split import (
    assign_source_group,
)
from sai.data.pleias_final_subdocument_signature import COMPONENT
from sai.data.pleias_production_materializer import (
    DESTINATION_REPOSITORY,
    _load_signed,
    upload_verified,
)
from sai.data.pleias_subdocument_rewrite import OUTPUT_SCHEMA as SOURCE_ROW_SCHEMA
from sai.data.pleias_subdocument_rewrite import SHARD_SCHEMA as SOURCE_SHARD_SCHEMA
from sai.data.pleias_subdocument_rewrite import _schema as _source_schema
from sai.data.pleias_subdocument_signature import _download
from sai.data.token_stream import canonical_sha256

SHARD_SCHEMA = "sai-pleias-cross-source-subdocument-rewritten-shard-v2"
OUTPUT_SCHEMA = "sai-pleias-cross-source-subdocument-deduplicated-candidate-v1"
DESTINATION_PREFIX = "final/nontraining/pleias-cross-source/20260826-r1"
COMPONENT_PRIORITY = 1


class PleiasCrossSourceSubdocumentRewriteError(RuntimeError):
    """Source, global decision, rewrite, or remote output differs."""


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasCrossSourceSubdocumentRewriteError(
            "pyarrow is required"
        ) from error
    source = _source_schema()
    fields = [
        field
        for field in source
        if field.name
        not in {
            "schema",
            "word_count",
            "content_sha256",
            "text",
            "training_ready",
        }
    ]
    return pa.schema(
        [pa.field("schema", pa.string())]
        + fields
        + [
            pa.field("word_count", pa.int64()),
            pa.field("source_group_sha256", pa.string()),
            pa.field("source_group_bucket", pa.int32()),
            pa.field("corpus_split", pa.string()),
            pa.field("source_split_policy_sha256", pa.string()),
            pa.field("pre_cross_source_content_sha256", pa.string()),
            pa.field("content_sha256", pa.string()),
            pa.field("cross_source_subdocument_transform_sha256", pa.string()),
            pa.field("text", pa.string()),
            pa.field("training_ready", pa.bool_()),
        ]
    )


def rewrite_row(
    row: dict[str, Any],
    source_row_index: int,
    decisions: list[tuple[str, int, int, int, str, int, int]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Replay one final PleIAs row against its exact global decisions."""

    text = row.get("text")
    content_sha256 = row.get("content_sha256")
    identity = row.get("source_row_identity_sha256")
    if (
        row.get("schema") != SOURCE_ROW_SCHEMA
        or row.get("training_ready") is not False
        or row.get("token_count_requires_recomputation") is not True
        or not isinstance(text, str)
        or not isinstance(content_sha256, str)
        or not isinstance(identity, str)
    ):
        raise PleiasCrossSourceSubdocumentRewriteError("PleIAs source row differs")
    collection = row.get("collection")
    rewritten, counts, transform = rewrite_text(
        text=text,
        content_sha256=content_sha256,
        identity=identity,
        source_row_index=source_row_index,
        decisions=decisions,
        code_document=(
            isinstance(collection, str) and "github" in collection.casefold()
        ),
    )
    source_group, corpus_split, source_group_bucket = assign_source_group(
        COMPONENT,
        {
            "source_repository": row.get("source_repository"),
            "source_revision": row.get("source_revision"),
            "source_path": row.get("source_path"),
            "source_parent_sha256": row.get("source_parent_sha256"),
        },
    )
    result = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "schema",
            "word_count",
            "content_sha256",
            "text",
            "training_ready",
        }
    }
    result.update(
        {
            "schema": OUTPUT_SCHEMA,
            "word_count": len(_WORD.findall(rewritten)),
            "source_group_sha256": source_group,
            "source_group_bucket": source_group_bucket,
            "corpus_split": corpus_split,
            "source_split_policy_sha256": SPLIT_POLICY_SHA256,
            "pre_cross_source_content_sha256": content_sha256,
            "content_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
            "cross_source_subdocument_transform_sha256": transform,
            "text": rewritten,
            "training_ready": False,
        }
    )
    return result, counts


def run_shard(
    source_root: Path,
    decision_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Rewrite, upload, verify, and remove one local PleIAs text shard."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasCrossSourceSubdocumentRewriteError("rewrite arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasCrossSourceSubdocumentRewriteError(
            "pyarrow is required"
        ) from error
    source = _load_signed(
        source_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
        SOURCE_SHARD_SCHEMA,
    )
    decision = _load_signed(
        decision_root / "aggregate.json", DECISION_AGGREGATE_SCHEMA
    )
    if (
        source.get("logical_shards") != logical_shards
        or source.get("shard_index") != shard_index
        or source.get("pleias_global_subdocument_rewrite_complete") is not True
        or source.get("local_payload_removed_after_remote_verification") is not True
        or decision.get("cross_source_subdocument_decision_complete") is not True
    ):
        raise PleiasCrossSourceSubdocumentRewriteError("rewrite source differs")
    output_root.mkdir(parents=True)
    local_paths = {
        split: output_root / f"{split}.parquet"
        for split in ("train", "development")
    }
    temporary = {
        split: output_root / f".{split}.rewrite.partial.{uuid.uuid4().hex}.parquet"
        for split in local_paths
    }
    counts: Counter[str] = Counter()
    ordered_transforms = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        prefix="sai-pleias-cross-source-rewrite-", dir=scratch_root
    ) as directory:
        scratch = Path(directory)
        connection, bucket_receipts, decision_rows = decision_database(
            decision_root,
            COMPONENT,
            COMPONENT_PRIORITY,
            shard_index,
            logical_shards,
            scratch / "deletions.sqlite3",
        )
        source_path = _download(source, token, scratch)
        parquet = pq.ParquetFile(source_path)
        writers = {}
        try:
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                output_rows = {split: [] for split in local_paths}
                for relative, row in enumerate(batch.to_pylist()):
                    source_row_index = row_offset + relative
                    decisions = connection.execute(
                        "SELECT document_identity_sha256, chunk_index, "
                        "character_start, character_end, normalized_sha256, "
                        "frequency, budget FROM deletions WHERE source_row_index=? "
                        "ORDER BY chunk_index",
                        (source_row_index,),
                    ).fetchall()
                    result, row_counts = rewrite_row(
                        row, source_row_index, decisions
                    )
                    output_rows[result["corpus_split"]].append(result)
                    counts["documents"] += 1
                    counts["input_text_utf8_bytes"] += len(row["text"].encode())
                    counts["output_text_utf8_bytes"] += len(
                        result["text"].encode()
                    )
                    counts[f"split::{result['corpus_split']}::documents"] += 1
                    counts[
                        f"split::{result['corpus_split']}::text_utf8_bytes"
                    ] += len(result["text"].encode())
                    stratum = result["semantic_stratum"]
                    quality_floor = result["semantic_quality_floor_milli"]
                    counts[f"semantic_stratum::{stratum}::documents"] += 1
                    counts[f"semantic_stratum::{stratum}::text_utf8_bytes"] += len(
                        result["text"].encode()
                    )
                    counts[
                        f"quality_floor_milli::{quality_floor}::documents"
                    ] += 1
                    phase = result["semantic_curriculum_phase"]
                    counts[f"curriculum_phase::{phase}::documents"] += 1
                    counts[f"curriculum_phase::{phase}::text_utf8_bytes"] += len(
                        result["text"].encode()
                    )
                    difficulty = result["semantic_difficulty_mean_milli"]
                    counts[f"difficulty_mean_milli::{difficulty}::documents"] += 1
                    for domain in result["semantic_domains"]:
                        counts[f"semantic_domain::{domain}::documents"] += 1
                        counts[
                            f"semantic_domain::{domain}::text_utf8_bytes"
                        ] += len(result["text"].encode())
                    for key, value in row_counts.items():
                        counts[key] += value
                    ordered_transforms.update(
                        bytes.fromhex(
                            result["cross_source_subdocument_transform_sha256"]
                        )
                    )
                for split, rows in output_rows.items():
                    if not rows:
                        continue
                    if split not in writers:
                        writers[split] = pq.ParquetWriter(
                            temporary[split], _schema(), compression="zstd"
                        )
                    writers[split].write_table(
                        pa.Table.from_pylist(rows, schema=_schema())
                    )
                row_offset += batch.num_rows
            if row_offset != parquet.metadata.num_rows:
                raise PleiasCrossSourceSubdocumentRewriteError(
                    "rewrite document coverage differs"
                )
            if counts["candidate_deletion_chunks"] != decision_rows:
                raise PleiasCrossSourceSubdocumentRewriteError(
                    "rewrite decision accounting differs"
                )
        except BaseException:
            for writer in writers.values():
                writer.close()
            connection.close()
            for path in temporary.values():
                path.unlink(missing_ok=True)
            raise
        for writer in writers.values():
            writer.close()
        connection.close()
    remote_outputs = {}
    for split, local_path in local_paths.items():
        if temporary[split].exists():
            os.replace(temporary[split], local_path)
            remote_path = (
                f"{DESTINATION_PREFIX}/{split}/"
                f"shard-{shard_index:05d}-of-{logical_shards:05d}.parquet"
            )
            remote_outputs[split] = upload_verified(
                local_path, remote_path, token, repository=DESTINATION_REPOSITORY
            )
            local_path.unlink()
        else:
            remote_outputs[split] = None
        if bool(remote_outputs[split]) is not bool(
            counts[f"split::{split}::documents"]
        ):
            raise PleiasCrossSourceSubdocumentRewriteError(
                "physical split output accounting differs"
            )
    if counts["documents"] != source.get("counts", {}).get("documents"):
        raise PleiasCrossSourceSubdocumentRewriteError(
            "rewrite source count differs"
        )
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_cross_source_rewritten_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "rewritten_shard_receipt_sha256": source["receipt_sha256"],
            "cross_source_decision_aggregate_receipt_sha256": decision[
                "receipt_sha256"
            ],
            "ordered_decision_bucket_receipts_sha256": canonical_sha256(
                bucket_receipts
            ),
        },
        "counts": dict(sorted(counts.items())),
        "source_disjoint_split_policy": SPLIT_POLICY,
        "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
        "ordered_transform_digests_sha256": ordered_transforms.hexdigest(),
        "remote_outputs": remote_outputs,
        "physical_train_development_partition_complete": True,
        "local_payload_removed_after_remote_verification": True,
        "benchmark_decontamination_complete": True,
        "pleias_internal_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": True,
        "source_disjoint_split_complete": True,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = run_shard(
        args.source_root,
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
