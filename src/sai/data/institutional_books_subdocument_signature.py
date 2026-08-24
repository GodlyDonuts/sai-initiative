"""Extract source-safe signatures from benchmark-disjoint Institutional Books."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.frequency_length_subdocument_deduplication import (
    DEFAULT_SEGMENT_CHARACTERS,
)
from sai.data.institutional_books_full_decontamination import (
    CLEAN_SCHEMA,
)
from sai.data.institutional_books_full_decontamination import (
    SCHEMA as DECONTAMINATION_SCHEMA,
)
from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA as BOOK_SCHEMA,
)
from sai.data.institutional_books_materializer import _load_json, _valid_receipt
from sai.data.institutional_books_mechanical_filter import (
    AGGREGATE_SCHEMA as FILTER_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_mechanical_filter import (
    SHARD_SCHEMA as FILTER_SHARD_SCHEMA,
)
from sai.data.pleias_subdocument_signature import (
    HASH_BUCKETS,
    _schema,
    signature_rows_for_text,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-institutional-books-subdocument-signature-shard-v1"
AGGREGATE_SCHEMA = "sai-institutional-books-subdocument-signature-aggregate-v1"
COMPONENT = "institutional_books"


class InstitutionalBooksSubdocumentSignatureError(RuntimeError):
    """Book selection, full-text identity, signature, or coverage differs."""


def _clean_books(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(root / "receipt.json")
    descriptor = receipt.get("benchmark_disjoint_books")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        not _valid_receipt(receipt, DECONTAMINATION_SCHEMA)
        or receipt.get("full_selected_source_population_decontaminated") is not True
        or not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise InstitutionalBooksSubdocumentSignatureError(
            "book decontamination differs"
        )
    books: dict[str, dict[str, Any]] = {}
    ordered = []
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                unsigned = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                barcode = row.get("source_book_id")
                if (
                    row.get("schema") != CLEAN_SCHEMA
                    or not isinstance(barcode, str)
                    or not barcode
                    or barcode in books
                    or row.get("record_sha256") != canonical_sha256(unsigned)
                    or row.get("benchmark_decontamination_complete") is not True
                    or row.get("training_ready") is not False
                ):
                    raise InstitutionalBooksSubdocumentSignatureError(
                        "benchmark-disjoint book row differs"
                    )
                books[barcode] = row
                ordered.append(row["record_sha256"])
    except (OSError, json.JSONDecodeError) as error:
        raise InstitutionalBooksSubdocumentSignatureError(
            "benchmark-disjoint book stream differs"
        ) from error
    if (
        len(books) != descriptor.get("rows")
        or len(books) != receipt.get("clean_rows")
        or canonical_sha256(ordered) != descriptor.get("ordered_records_sha256")
    ):
        raise InstitutionalBooksSubdocumentSignatureError(
            "benchmark-disjoint book coverage differs"
        )
    return books, receipt


def _filtered_shard(
    root: Path, logical_shards: int, shard_index: int
) -> tuple[Path | None, dict[str, Any]]:
    shard_root = root / "shards" / f"shard_{shard_index:05d}"
    receipt = _load_json(shard_root / "receipt.json")
    descriptor = receipt.get("output")
    if (
        not _valid_receipt(receipt, FILTER_SHARD_SCHEMA)
        or receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
    ):
        raise InstitutionalBooksSubdocumentSignatureError(
            "filtered book shard differs"
        )
    if descriptor is None:
        if receipt.get("retained_rows") != 0:
            raise InstitutionalBooksSubdocumentSignatureError(
                "empty filtered book shard differs"
            )
        return None, receipt
    path = shard_root / descriptor.get("path", "")
    if (
        not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
        or descriptor.get("rows") != receipt.get("retained_rows")
    ):
        raise InstitutionalBooksSubdocumentSignatureError(
            "filtered book payload differs"
        )
    return path, receipt


def run_shard(
    filtered_root: Path,
    decontamination_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Sign every clean book located in one private filtered shard."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise InstitutionalBooksSubdocumentSignatureError(
            "book signature arguments differ"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksSubdocumentSignatureError(
            "pyarrow is required"
        ) from error
    clean, decontamination = _clean_books(decontamination_root)
    source_path, filtered = _filtered_shard(
        filtered_root, logical_shards, shard_index
    )
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
        if source_path is not None:
            parquet = pq.ParquetFile(source_path)
            row_offset = 0
            for batch in parquet.iter_batches(batch_size=16, use_threads=False):
                outputs = [[] for _ in range(HASH_BUCKETS)]
                for relative, row in enumerate(batch.to_pylist()):
                    source_row_index = row_offset + relative
                    counts["filtered_source_rows"] += 1
                    barcode = row.get("barcode_src")
                    clean_row = clean.get(barcode)
                    if clean_row is None:
                        continue
                    text = row.get("text")
                    content_sha256 = row.get("source_content_sha256")
                    identity = clean_row.get("candidate_identity_sha256")
                    if (
                        row.get("schema") != BOOK_SCHEMA
                        or row.get("training_ready") is not False
                        or not isinstance(text, str)
                        or hashlib.sha256(text.encode()).hexdigest()
                        != content_sha256
                        or content_sha256
                        != clean_row.get("full_source_content_sha256")
                    ):
                        raise InstitutionalBooksSubdocumentSignatureError(
                            "clean full book identity differs"
                        )
                    rows = signature_rows_for_text(
                        component=COMPONENT,
                        text=text,
                        identity=identity,
                        content_sha256=content_sha256,
                        source_shard=shard_index,
                        source_row_index=source_row_index,
                        code_document=False,
                    )
                    counts["documents"] += 1
                    counts["source_text_utf8_bytes"] += len(text.encode())
                    counts["signatures"] += len(rows)
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
                raise InstitutionalBooksSubdocumentSignatureError(
                    "filtered book row coverage differs"
                )
        if counts["filtered_source_rows"] != filtered.get("retained_rows", 0):
            raise InstitutionalBooksSubdocumentSignatureError(
                "filtered book accounting differs"
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
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_institutional_books_subdocument_signatures",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "filtered_shard_receipt_sha256": filtered["receipt_sha256"],
            "decontamination_receipt_sha256": decontamination["receipt_sha256"],
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
        "benchmark_decontamination_complete": True,
        "global_subdocument_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate(
    filtered_root: Path,
    decontamination_root: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Seal exact signature coverage over every benchmark-disjoint book."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise InstitutionalBooksSubdocumentSignatureError(
            "book signature aggregate arguments differ"
        )
    clean, decontamination = _clean_books(decontamination_root)
    filtered = _load_json(filtered_root / "aggregate.json")
    if (
        not _valid_receipt(filtered, FILTER_AGGREGATE_SCHEMA)
        or filtered.get("shards", {}).get("logical_shards") != logical_shards
    ):
        raise InstitutionalBooksSubdocumentSignatureError(
            "filtered book aggregate differs"
        )
    totals: Counter[str] = Counter()
    receipts = []
    document_hashes = []
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_json(root / "receipt.json")
        _source_path, source = _filtered_shard(
            filtered_root, logical_shards, shard_index
        )
        outputs = receipt.get("outputs")
        if (
            not _valid_receipt(receipt, SHARD_SCHEMA)
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("filtered_shard_receipt_sha256")
            != source["receipt_sha256"]
            or receipt.get("source", {}).get("decontamination_receipt_sha256")
            != decontamination["receipt_sha256"]
            or receipt.get("hash_partition", {}).get("buckets") != HASH_BUCKETS
            or not isinstance(outputs, list)
            or len(outputs) != HASH_BUCKETS
        ):
            raise InstitutionalBooksSubdocumentSignatureError(
                "book signature shard differs"
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
                raise InstitutionalBooksSubdocumentSignatureError(
                    "book signature bucket differs"
                )
            totals["signature_output_bytes"] += descriptor["bytes"]
        for key, value in receipt["counts"].items():
            totals[key] += value
        receipts.append(receipt["receipt_sha256"])
        document_hashes.append(receipt["ordered_document_identities_sha256"])
    if totals["documents"] != len(clean):
        raise InstitutionalBooksSubdocumentSignatureError(
            "clean book signature coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_institutional_books_subdocument_signatures",
        "source": {
            "filtered_aggregate_receipt_sha256": filtered["receipt_sha256"],
            "decontamination_receipt_sha256": decontamination["receipt_sha256"],
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
            "ordered_document_partition_digests_sha256": canonical_sha256(
                document_hashes
            ),
        },
        "totals": dict(sorted(totals.items())),
        "complete_benchmark_disjoint_book_coverage": True,
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
    combine = commands.add_parser("aggregate")
    for child in (shard, combine):
        child.add_argument("--filtered-root", type=Path, required=True)
        child.add_argument("--decontamination-root", type=Path, required=True)
        child.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        run_shard(
            args.filtered_root,
            args.decontamination_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
        )
        if args.command == "shard"
        else aggregate(
            args.filtered_root,
            args.decontamination_root,
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
