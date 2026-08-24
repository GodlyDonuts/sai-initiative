"""Re-sign final internally deduplicated PleIAs shards for cross-source work."""

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
from sai.data.frequency_length_subdocument_deduplication import (
    DEFAULT_SEGMENT_CHARACTERS,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_subdocument_rewrite import OUTPUT_SCHEMA
from sai.data.pleias_subdocument_rewrite import (
    SHARD_SCHEMA as REWRITTEN_SCHEMA,
)
from sai.data.pleias_subdocument_signature import (
    HASH_BUCKETS,
    _download,
    _schema,
    signature_rows_for_text,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-final-subdocument-signature-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-final-subdocument-signature-aggregate-v1"
COMPONENT = "pleias_common_corpus"


class PleiasFinalSubdocumentSignatureError(RuntimeError):
    """Rewritten custody, final content identity, or signature coverage differs."""


def final_signature_rows(
    row: dict[str, Any], source_shard: int, source_row_index: int
) -> list[dict[str, Any]]:
    """Replay one rewritten row and emit universal source-safe signatures."""

    if row.get("schema") != OUTPUT_SCHEMA or row.get("training_ready") is not False:
        raise PleiasFinalSubdocumentSignatureError("rewritten candidate differs")
    collection = row.get("collection")
    try:
        return signature_rows_for_text(
            component=COMPONENT,
            text=row.get("text"),
            identity=row.get("source_row_identity_sha256"),
            content_sha256=row.get("content_sha256"),
            source_shard=source_shard,
            source_row_index=source_row_index,
            code_document=(
                isinstance(collection, str) and "github" in collection.casefold()
            ),
        )
    except RuntimeError as error:
        raise PleiasFinalSubdocumentSignatureError(
            "rewritten candidate identity differs"
        ) from error


def run_shard(
    rewritten_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Download one final remote shard and hash-partition its exact signatures."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasFinalSubdocumentSignatureError(
            "final signature arguments differ"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasFinalSubdocumentSignatureError("pyarrow is required") from error
    rewritten = _load_signed(
        rewritten_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
        REWRITTEN_SCHEMA,
    )
    expected_documents = rewritten.get("counts", {}).get("documents")
    expected_bytes = rewritten.get("counts", {}).get("output_text_utf8_bytes")
    if (
        rewritten.get("logical_shards") != logical_shards
        or rewritten.get("shard_index") != shard_index
        or rewritten.get("pleias_global_subdocument_rewrite_complete") is not True
        or rewritten.get("local_payload_removed_after_remote_verification") is not True
        or isinstance(expected_documents, bool)
        or not isinstance(expected_documents, int)
        or expected_documents <= 0
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise PleiasFinalSubdocumentSignatureError("rewritten source differs")
    output_root.mkdir(parents=True)
    output_paths = [
        output_root / f"bucket-{index:02x}.parquet" for index in range(HASH_BUCKETS)
    ]
    temporary_paths = [
        output_root / f".bucket-{index:02x}.partial.{uuid.uuid4().hex}.parquet"
        for index in range(HASH_BUCKETS)
    ]
    writers = [
        pq.ParquetWriter(path, _schema(), compression="zstd")
        for path in temporary_paths
    ]
    counts: Counter[str] = Counter()
    ordered = hashlib.sha256()
    ordered_documents = hashlib.sha256()
    ordered_by_bucket = [hashlib.sha256() for _ in range(HASH_BUCKETS)]
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-pleias-final-signature-", dir=scratch_root
        ) as directory:
            source = _download(rewritten, token, Path(directory))
            parquet = pq.ParquetFile(source)
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                outputs = [[] for _ in range(HASH_BUCKETS)]
                for relative, row in enumerate(batch.to_pylist()):
                    rows = final_signature_rows(
                        row, shard_index, row_offset + relative
                    )
                    text = row["text"]
                    identity = row["source_row_identity_sha256"]
                    counts["documents"] += 1
                    counts["source_text_utf8_bytes"] += len(text.encode())
                    counts["signatures"] += len(rows)
                    counts["code_signatures"] += sum(item["code"] for item in rows)
                    ordered_documents.update(bytes.fromhex(identity))
                    for signature in rows:
                        ordered.update(bytes.fromhex(signature["signature_sha256"]))
                        bucket = int(signature["normalized_sha256"][0], 16)
                        outputs[bucket].append(signature)
                        ordered_by_bucket[bucket].update(
                            bytes.fromhex(signature["signature_sha256"])
                        )
                        counts[f"bucket_{bucket:02x}_signatures"] += 1
                for bucket, rows in enumerate(outputs):
                    if rows:
                        writers[bucket].write_table(
                            pa.Table.from_pylist(rows, schema=_schema())
                        )
                row_offset += batch.num_rows
            if row_offset != parquet.metadata.num_rows:
                raise PleiasFinalSubdocumentSignatureError(
                    "final signature row coverage differs"
                )
    except BaseException:
        for writer in writers:
            writer.close()
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    for writer in writers:
        writer.close()
    for temporary, output in zip(temporary_paths, output_paths, strict=True):
        os.replace(temporary, output)
    if (
        counts["documents"] != expected_documents
        or counts["source_text_utf8_bytes"] != expected_bytes
    ):
        raise PleiasFinalSubdocumentSignatureError(
            "final signature source accounting differs"
        )
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_final_subdocument_signatures",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "rewritten_shard_receipt_sha256": rewritten["receipt_sha256"],
            "remote_output_sha256": rewritten["remote_output"]["sha256"],
        },
        "policy": {
            "minimum_segment_characters": DEFAULT_SEGMENT_CHARACTERS,
            "normalization": "NFKC_casefold_number_placeholder_whitespace_collapse",
            "code_normalization": "identity",
            "source_text_persisted": False,
        },
        "counts": dict(sorted(counts.items())),
        "ordered_document_identities_sha256": ordered_documents.hexdigest(),
        "ordered_signature_digests_sha256": ordered.hexdigest(),
        "hash_partition": {
            "buckets": HASH_BUCKETS,
            "key": "first_normalized_sha256_hex_nibble",
        },
        "outputs": [
            {
                "bucket": index,
                "path": path.name,
                "rows": counts[f"bucket_{index:02x}_signatures"],
                "ordered_signature_digests_sha256": ordered_by_bucket[
                    index
                ].hexdigest(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for index, path in enumerate(output_paths)
        ],
        "pleias_internal_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate(
    rewritten_root: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify complete final PleIAs signature coverage."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise PleiasFinalSubdocumentSignatureError(
            "final signature aggregate arguments differ"
        )
    totals: Counter[str] = Counter()
    receipts = []
    document_hashes = []
    for shard_index in range(logical_shards):
        rewritten = _load_signed(
            rewritten_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
            REWRITTEN_SCHEMA,
        )
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        outputs = receipt.get("outputs")
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("rewritten_shard_receipt_sha256")
            != rewritten["receipt_sha256"]
            or receipt.get("counts", {}).get("documents")
            != rewritten.get("counts", {}).get("documents")
            or receipt.get("counts", {}).get("source_text_utf8_bytes")
            != rewritten.get("counts", {}).get("output_text_utf8_bytes")
            or receipt.get("hash_partition", {}).get("buckets") != HASH_BUCKETS
            or not isinstance(outputs, list)
            or len(outputs) != HASH_BUCKETS
        ):
            raise PleiasFinalSubdocumentSignatureError(
                "final signature shard differs"
            )
        for index, descriptor in enumerate(outputs):
            path = root / descriptor.get("path", "")
            if (
                descriptor.get("bucket") != index
                or descriptor.get("rows")
                != receipt.get("counts", {}).get(
                    f"bucket_{index:02x}_signatures", 0
                )
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise PleiasFinalSubdocumentSignatureError(
                    "final signature bucket differs"
                )
            totals["signature_output_bytes"] += descriptor["bytes"]
        for key, value in receipt["counts"].items():
            totals[key] += value
        receipts.append(receipt["receipt_sha256"])
        document_hashes.append(receipt["ordered_document_identities_sha256"])
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_final_subdocument_signatures",
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
            "ordered_document_partition_digests_sha256": canonical_sha256(
                document_hashes
            ),
        },
        "totals": dict(sorted(totals.items())),
        "complete_final_pleias_document_coverage": True,
        "source_text_persisted": False,
        "pleias_internal_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
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
    combine = commands.add_parser("aggregate")
    for child in (shard, combine):
        child.add_argument("--rewritten-root", type=Path, required=True)
        child.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        run_shard(
            args.rewritten_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
        if args.command == "shard"
        else aggregate(
            args.rewritten_root,
            args.shards_root,
            args.logical_shards,
            args.output,
        )
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
