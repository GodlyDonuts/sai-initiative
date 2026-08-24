"""Extract source-safe subdocument signatures from remote PleIAs candidates."""

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
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_production_materializer import SHARD_SCHEMA as MATERIALIZED_SCHEMA
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-subdocument-signature-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-subdocument-signature-aggregate-v1"
SIGNATURE_SCHEMA = "sai-subdocument-signature-v1"
HASH_BUCKETS = 16


class PleiasSubdocumentSignatureError(RuntimeError):
    """Remote custody, candidate identity, or signature coverage differs."""


def _download(receipt: dict[str, Any], token: str, scratch: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise PleiasSubdocumentSignatureError("huggingface_hub is required") from error
    remote = receipt.get("remote_output")
    if not isinstance(remote, dict) or not token:
        raise PleiasSubdocumentSignatureError("remote descriptor differs")
    path = Path(
        hf_hub_download(
            repo_id=remote["repository"],
            filename=remote["path"],
            repo_type="dataset",
            revision=remote["commit"],
            token=token,
            cache_dir=scratch / "cache",
            local_dir=scratch / "local",
        )
    )
    if (
        not path.is_file()
        or path.stat().st_size != remote["bytes"]
        or sha256_file(path) != remote["sha256"]
    ):
        raise PleiasSubdocumentSignatureError("remote payload identity differs")
    return path


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasSubdocumentSignatureError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("component", pa.string()),
            ("source_shard", pa.int32()),
            ("source_row_index", pa.int64()),
            ("document_identity_sha256", pa.string()),
            ("content_sha256", pa.string()),
            ("chunk_index", pa.int32()),
            ("character_start", pa.int64()),
            ("character_end", pa.int64()),
            ("normalized_sha256", pa.string()),
            ("normalized_characters", pa.int64()),
            ("code", pa.bool_()),
            ("signature_sha256", pa.string()),
            ("training_ready", pa.bool_()),
        ]
    )


def signature_rows(
    candidate: dict[str, Any], source_shard: int, source_row_index: int
) -> list[dict[str, Any]]:
    """Losslessly segment one candidate and emit no source text."""

    text = candidate.get("text")
    identity = candidate.get("source_row_identity_sha256")
    content_sha256 = candidate.get("content_sha256")
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("training_ready") is not False
        or not isinstance(text, str)
        or not isinstance(identity, str)
        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
    ):
        raise PleiasSubdocumentSignatureError("candidate row differs")
    collection = candidate.get("collection")
    code_document = isinstance(collection, str) and "github" in collection.casefold()
    chunks = segment_subdocuments(
        text,
        minimum_characters=DEFAULT_SEGMENT_CHARACTERS,
        code_document=code_document,
    )
    rows = []
    for chunk_index, chunk in enumerate(chunks):
        normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
        if not normalized:
            continue
        row = {
            "schema": SIGNATURE_SCHEMA,
            "component": "pleias_common_corpus",
            "source_shard": source_shard,
            "source_row_index": source_row_index,
            "document_identity_sha256": identity,
            "content_sha256": content_sha256,
            "chunk_index": chunk_index,
            "character_start": chunk["character_start"],
            "character_end": chunk["character_end"],
            "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "normalized_characters": len(normalized),
            "code": chunk["code"],
            "training_ready": False,
        }
        row["signature_sha256"] = canonical_sha256(row)
        rows.append(row)
    return rows


def run_shard(
    materialized_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Download one verified remote shard and emit only exact chunk signatures."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasSubdocumentSignatureError("signature arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasSubdocumentSignatureError("pyarrow is required") from error
    receipt = _load_signed(
        materialized_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
        MATERIALIZED_SCHEMA,
    )
    if (
        receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("full_document_benchmark_decontamination_complete") is not True
    ):
        raise PleiasSubdocumentSignatureError("materialized receipt differs")
    output_root.mkdir(parents=True)
    output_paths = [
        output_root / f"bucket-{index:02x}.parquet" for index in range(HASH_BUCKETS)
    ]
    temporary_paths = [
        output_root / f".bucket-{index:02x}.partial.{uuid.uuid4().hex}.parquet"
        for index in range(HASH_BUCKETS)
    ]
    counts: Counter[str] = Counter()
    ordered = hashlib.sha256()
    ordered_by_bucket = [hashlib.sha256() for _ in range(HASH_BUCKETS)]
    writers = [
        pq.ParquetWriter(path, _schema(), compression="zstd")
        for path in temporary_paths
    ]
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-pleias-subdocument-signature-", dir=scratch_root
        ) as directory:
            source_path = _download(receipt, token, Path(directory))
            parquet = pq.ParquetFile(source_path)
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                output_rows = [[] for _ in range(HASH_BUCKETS)]
                for relative, candidate in enumerate(batch.to_pylist()):
                    rows = signature_rows(candidate, shard_index, row_offset + relative)
                    counts["documents"] += 1
                    counts["source_text_utf8_bytes"] += len(candidate["text"].encode())
                    counts["signatures"] += len(rows)
                    counts["code_signatures"] += sum(row["code"] for row in rows)
                    for row in rows:
                        ordered.update(bytes.fromhex(row["signature_sha256"]))
                    for row in rows:
                        bucket = int(row["normalized_sha256"][0], 16)
                        output_rows[bucket].append(row)
                        ordered_by_bucket[bucket].update(
                            bytes.fromhex(row["signature_sha256"])
                        )
                        counts[f"bucket_{bucket:02x}_signatures"] += 1
                for bucket, bucket_rows in enumerate(output_rows):
                    if bucket_rows:
                        writers[bucket].write_table(
                            pa.Table.from_pylist(bucket_rows, schema=_schema())
                        )
                row_offset += batch.num_rows
            if row_offset != parquet.metadata.num_rows:
                raise PleiasSubdocumentSignatureError(
                    "signature document coverage differs"
                )
    except BaseException:
        for writer in writers:
            writer.close()
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    for writer in writers:
        writer.close()
    for temporary, output_path in zip(temporary_paths, output_paths, strict=True):
        os.replace(temporary, output_path)
    if counts["documents"] != receipt.get("counts", {}).get("retained_rows", 0):
        raise PleiasSubdocumentSignatureError("signature receipt coverage differs")
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_subdocument_signature_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "materialized_shard_receipt_sha256": receipt["receipt_sha256"],
            "remote_output_sha256": receipt["remote_output"]["sha256"],
        },
        "policy": {
            "minimum_segment_characters": DEFAULT_SEGMENT_CHARACTERS,
            "normalization": "NFKC_casefold_number_placeholder_whitespace_collapse",
            "code_normalization": "identity",
            "source_text_persisted": False,
        },
        "counts": dict(sorted(counts.items())),
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
        "global_subdocument_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate(
    materialized_root: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Seal exact signature coverage across all remotely materialized shards."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise PleiasSubdocumentSignatureError("aggregate arguments differ")
    totals: Counter[str] = Counter()
    receipts = []
    for shard_index in range(logical_shards):
        materialized = _load_signed(
            materialized_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
            MATERIALIZED_SCHEMA,
        )
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        outputs = receipt.get("outputs")
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("materialized_shard_receipt_sha256")
            != materialized["receipt_sha256"]
            or receipt.get("counts", {}).get("documents")
            != materialized.get("counts", {}).get("retained_rows", 0)
            or receipt.get("hash_partition", {}).get("buckets") != HASH_BUCKETS
            or not isinstance(outputs, list)
            or len(outputs) != HASH_BUCKETS
        ):
            raise PleiasSubdocumentSignatureError("signature shard differs")
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
                raise PleiasSubdocumentSignatureError("signature bucket differs")
            totals["signature_output_bytes"] += descriptor["bytes"]
        for key, value in receipt["counts"].items():
            totals[key] += value
        receipts.append(receipt["receipt_sha256"])
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_subdocument_signatures",
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "complete_materialized_document_coverage": True,
        "source_text_persisted": False,
        "global_subdocument_deduplication_complete": False,
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
    shard.add_argument("--materialized-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--materialized-root", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.materialized_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate(
            args.materialized_root,
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
