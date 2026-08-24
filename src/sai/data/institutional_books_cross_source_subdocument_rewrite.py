"""Apply verified cross-source deletions to private Institutional Books."""

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
from sai.data.institutional_books_full_decontamination import CLEAN_SCHEMA
from sai.data.institutional_books_materializer import OUTPUT_SCHEMA as SOURCE_ROW_SCHEMA
from sai.data.institutional_books_subdocument_signature import (
    COMPONENT,
    _clean_books,
    _filtered_shard,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-institutional-books-cross-source-rewritten-shard-v1"
OUTPUT_SCHEMA = "sai-institutional-books-cross-source-deduplicated-row-v1"
COMPONENT_PRIORITY = 0


class InstitutionalBooksCrossSourceSubdocumentRewriteError(RuntimeError):
    """Private source, global decision, or exact rewrite differs."""


def _schema(source_schema):
    try:
        import pyarrow as pa
    except ImportError as error:
        raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
            "pyarrow is required"
        ) from error
    fields = [
        field
        for field in source_schema
        if field.name
        not in {
            "schema",
            "source_content_sha256",
            "text",
            "training_ready",
        }
    ]
    return pa.schema(
        [pa.field("schema", pa.string())]
        + fields
        + [
            pa.field("source_content_sha256", pa.string()),
            pa.field("pre_cross_source_content_sha256", pa.string()),
            pa.field("content_sha256", pa.string()),
            pa.field("word_count", pa.int64()),
            pa.field("token_count_requires_recomputation", pa.bool_()),
            pa.field("cross_source_subdocument_transform_sha256", pa.string()),
            pa.field("text", pa.string()),
            pa.field("training_ready", pa.bool_()),
        ]
    )


def rewrite_row(
    row: dict[str, Any],
    clean: dict[str, Any],
    source_row_index: int,
    decisions: list[tuple[str, int, int, int, str, int, int]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Replay one clean private book against its exact global decisions."""

    text = row.get("text")
    content_sha256 = row.get("source_content_sha256")
    identity = clean.get("candidate_identity_sha256")
    if (
        row.get("schema") != SOURCE_ROW_SCHEMA
        or row.get("training_ready") is not False
        or clean.get("schema") != CLEAN_SCHEMA
        or clean.get("training_ready") is not False
        or clean.get("benchmark_decontamination_complete") is not True
        or row.get("barcode_src") != clean.get("source_book_id")
        or content_sha256 != clean.get("full_source_content_sha256")
        or not isinstance(text, str)
        or not isinstance(content_sha256, str)
        or not isinstance(identity, str)
    ):
        raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
            "clean book source row differs"
        )
    rewritten, counts, transform = rewrite_text(
        text=text,
        content_sha256=content_sha256,
        identity=identity,
        source_row_index=source_row_index,
        decisions=decisions,
        code_document=False,
    )
    result = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "schema",
            "source_content_sha256",
            "text",
            "training_ready",
        }
    }
    result.update(
        {
            "schema": OUTPUT_SCHEMA,
            "source_content_sha256": content_sha256,
            "pre_cross_source_content_sha256": content_sha256,
            "content_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
            "word_count": len(_WORD.findall(rewritten)),
            "token_count_requires_recomputation": True,
            "cross_source_subdocument_transform_sha256": transform,
            "text": rewritten,
            "training_ready": False,
        }
    )
    return result, counts


def run_shard(
    filtered_root: Path,
    decontamination_root: Path,
    decision_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Rewrite every benchmark-disjoint book in one private source shard."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
            "rewrite arguments differ"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
            "pyarrow is required"
        ) from error
    clean_books, decontamination = _clean_books(decontamination_root)
    source_path, filtered = _filtered_shard(
        filtered_root, logical_shards, shard_index
    )
    decision = _load_signed(
        decision_root / "aggregate.json", DECISION_AGGREGATE_SCHEMA
    )
    if decision.get("cross_source_subdocument_decision_complete") is not True:
        raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
            "global decision differs"
        )
    output_root.mkdir(parents=True)
    local_path = output_root / "cross_source_deduplicated.parquet"
    temporary = output_root / f".rewrite.partial.{uuid.uuid4().hex}.parquet"
    counts: Counter[str] = Counter()
    ordered_identities = hashlib.sha256()
    ordered_transforms = hashlib.sha256()
    writer = None
    with tempfile.TemporaryDirectory(
        prefix="sai-book-cross-source-rewrite-", dir=scratch_root
    ) as directory:
        connection, bucket_receipts, decision_rows = decision_database(
            decision_root,
            COMPONENT,
            COMPONENT_PRIORITY,
            shard_index,
            logical_shards,
            Path(directory) / "deletions.sqlite3",
        )
        try:
            if source_path is not None:
                parquet = pq.ParquetFile(source_path)
                output_schema = _schema(parquet.schema_arrow)
                row_offset = 0
                for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                    output_rows = []
                    for relative, row in enumerate(batch.to_pylist()):
                        source_row_index = row_offset + relative
                        counts["filtered_source_rows"] += 1
                        clean = clean_books.get(row.get("barcode_src"))
                        if clean is None:
                            continue
                        decisions = connection.execute(
                            "SELECT document_identity_sha256, chunk_index, "
                            "character_start, character_end, normalized_sha256, "
                            "frequency, budget FROM deletions "
                            "WHERE source_row_index=? ORDER BY chunk_index",
                            (source_row_index,),
                        ).fetchall()
                        result, row_counts = rewrite_row(
                            row, clean, source_row_index, decisions
                        )
                        output_rows.append(result)
                        counts["documents"] += 1
                        counts["input_text_utf8_bytes"] += len(row["text"].encode())
                        counts["output_text_utf8_bytes"] += len(
                            result["text"].encode()
                        )
                        for key, value in row_counts.items():
                            counts[key] += value
                        ordered_identities.update(
                            bytes.fromhex(clean["candidate_identity_sha256"])
                        )
                        ordered_transforms.update(
                            bytes.fromhex(
                                result[
                                    "cross_source_subdocument_transform_sha256"
                                ]
                            )
                        )
                    if output_rows:
                        if writer is None:
                            writer = pq.ParquetWriter(
                                temporary, output_schema, compression="zstd"
                            )
                        writer.write_table(
                            pa.Table.from_pylist(output_rows, schema=output_schema)
                        )
                    row_offset += batch.num_rows
                if row_offset != parquet.metadata.num_rows:
                    raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
                        "filtered source coverage differs"
                    )
            if counts["filtered_source_rows"] != filtered.get("retained_rows", 0):
                raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
                    "filtered source accounting differs"
                )
            if counts["candidate_deletion_chunks"] != decision_rows:
                raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
                    "rewrite decision accounting differs"
                )
        except BaseException:
            if writer is not None:
                writer.close()
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        if writer is not None:
            writer.close()
        connection.close()
    output = None
    if temporary.exists():
        os.replace(temporary, local_path)
        output = {
            "path": local_path.name,
            "rows": counts["documents"],
            "bytes": local_path.stat().st_size,
            "sha256": sha256_file(local_path),
        }
    elif counts["documents"]:
        raise InstitutionalBooksCrossSourceSubdocumentRewriteError(
            "private rewrite output is missing"
        )
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_institutional_books_cross_source_rewritten",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "filtered_shard_receipt_sha256": filtered["receipt_sha256"],
            "decontamination_receipt_sha256": decontamination["receipt_sha256"],
            "cross_source_decision_aggregate_receipt_sha256": decision[
                "receipt_sha256"
            ],
            "ordered_decision_bucket_receipts_sha256": canonical_sha256(
                bucket_receipts
            ),
        },
        "counts": dict(sorted(counts.items())),
        "ordered_document_identities_sha256": ordered_identities.hexdigest(),
        "ordered_transform_digests_sha256": ordered_transforms.hexdigest(),
        "output": output,
        "private_storage_only": True,
        "huggingface_redistribution_authorized": False,
        "benchmark_decontamination_complete": True,
        "cross_source_subdocument_deduplication_complete": True,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = run_shard(
        args.filtered_root,
        args.decontamination_root,
        args.decision_root,
        args.output_root,
        args.logical_shards,
        args.shard_index,
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
