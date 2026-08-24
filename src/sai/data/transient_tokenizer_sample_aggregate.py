"""Aggregate and replay every bounded transient tokenizer sample shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file
from sai.data.transient_tokenizer_sample import (
    SAMPLE_NAME,
)
from sai.data.transient_tokenizer_sample import (
    SCHEMA as SHARD_SCHEMA,
)
from sai.data.transient_tokenizer_sample import (
    STATUS as SHARD_STATUS,
)

SCHEMA = "sai-transient-tokenizer-sample-aggregate-v1"
STATUS = "complete_nontraining_tokenizer_sample_aggregate"


class TransientTokenizerSampleAggregateError(RuntimeError):
    """A shard, source receipt, document identity, or byte total differs."""


def aggregate(
    samples_root: Path,
    output: Path,
    *,
    logical_shards: int,
    maximum_bytes_per_shard: int,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Replay all sample text while persisting only global hash evidence."""

    if (
        output.exists()
        or output.is_symlink()
        or isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 512
        or isinstance(maximum_bytes_per_shard, bool)
        or not isinstance(maximum_bytes_per_shard, int)
        or maximum_bytes_per_shard < 1_000_000
    ):
        raise TransientTokenizerSampleAggregateError(
            "tokenizer sample aggregate arguments differ"
        )
    totals: Counter[str] = Counter()
    ordered_receipts = []
    ordered_documents = hashlib.sha256()
    ordered_samples = hashlib.sha256()
    source_receipts: set[str] = set()
    with tempfile.TemporaryDirectory(
        prefix="sai-tokenizer-sample-aggregate-", dir=scratch_root
    ) as directory:
        database = sqlite3.connect(Path(directory) / "identities.sqlite3")
        database.execute("PRAGMA journal_mode=DELETE")
        database.execute("PRAGMA synchronous=FULL")
        database.execute(
            "CREATE TABLE documents ("
            "identity_sha256 TEXT PRIMARY KEY, text_sha256 TEXT NOT NULL UNIQUE, "
            "shard_index INTEGER NOT NULL, row_index INTEGER NOT NULL) WITHOUT ROWID"
        )
        try:
            for shard_index in range(logical_shards):
                root = samples_root / f"shard_{shard_index:05d}"
                receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
                descriptor = receipt.get("sample")
                path = (
                    root / descriptor.get("path", "")
                    if isinstance(descriptor, dict)
                    else root
                )
                if (
                    receipt.get("status") != SHARD_STATUS
                    or receipt.get("tokenizer_measurement_only") is not True
                    or receipt.get("source_text_persisted_only_in_bounded_sample")
                    is not True
                    or not isinstance(descriptor, dict)
                    or descriptor.get("path") != SAMPLE_NAME
                    or descriptor.get("bytes", maximum_bytes_per_shard + 1)
                    > maximum_bytes_per_shard
                    or not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_nlink != 1
                    or path.stat().st_size != descriptor.get("bytes")
                    or sha256_file(path) != descriptor.get("sha256")
                    or receipt.get("selected_counts", {}).get("documents")
                    != descriptor.get("documents")
                    or receipt.get("selected_counts", {}).get("jsonl_bytes")
                    != descriptor.get("bytes")
                ):
                    raise TransientTokenizerSampleAggregateError(
                        "tokenizer sample shard differs"
                    )
                source_receipt = receipt.get("source_receipt_sha256")
                if (
                    not isinstance(source_receipt, str)
                    or len(source_receipt) != 64
                    or source_receipt in source_receipts
                ):
                    raise TransientTokenizerSampleAggregateError(
                        "tokenizer sample source custody differs"
                    )
                source_receipts.add(source_receipt)
                rows = 0
                sample_digest = hashlib.sha256()
                with path.open(encoding="utf-8") as handle:
                    for row_index, line in enumerate(handle):
                        encoded = line.encode()
                        sample_digest.update(encoded)
                        if not line.strip():
                            raise TransientTokenizerSampleAggregateError(
                                "tokenizer sample contains blank rows"
                            )
                        try:
                            document = normalize_document(json.loads(line))
                        except (json.JSONDecodeError, RuntimeError) as error:
                            raise TransientTokenizerSampleAggregateError(
                                "tokenizer sample document differs"
                            ) from error
                        identity = document["identity_sha256"]
                        text_sha256 = hashlib.sha256(
                            document["text"].encode()
                        ).hexdigest()
                        try:
                            database.execute(
                                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                                (identity, text_sha256, shard_index, row_index),
                            )
                        except sqlite3.IntegrityError as error:
                            raise TransientTokenizerSampleAggregateError(
                                "tokenizer sample contains a cross-shard duplicate"
                            ) from error
                        ordered_documents.update(bytes.fromhex(identity))
                        totals["documents"] += 1
                        totals["jsonl_bytes"] += len(encoded)
                        totals["text_utf8_bytes"] += len(document["text"].encode())
                        totals[
                            f"domain::{document['source']['domain']}::documents"
                        ] += 1
                        rows += 1
                database.commit()
                if rows != descriptor.get(
                    "documents"
                ) or sample_digest.hexdigest() != descriptor.get(
                    "ordered_jsonl_sha256"
                ):
                    raise TransientTokenizerSampleAggregateError(
                        "tokenizer sample row coverage differs"
                    )
                totals["shards"] += 1
                ordered_receipts.append(receipt["receipt_sha256"])
                ordered_samples.update(bytes.fromhex(descriptor["sha256"]))
        finally:
            database.close()
    if (
        totals["shards"] != logical_shards
        or totals["documents"] <= 0
        or totals["jsonl_bytes"] > logical_shards * maximum_bytes_per_shard
    ):
        raise TransientTokenizerSampleAggregateError(
            "tokenizer sample aggregate coverage differs"
        )
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "shards": {
            "logical_shards": logical_shards,
            "maximum_bytes_per_shard": maximum_bytes_per_shard,
            "maximum_total_bytes": logical_shards * maximum_bytes_per_shard,
            "ordered_receipts_sha256": canonical_sha256(ordered_receipts),
            "ordered_sample_content_digests_sha256": ordered_samples.hexdigest(),
        },
        "totals": dict(sorted(totals.items())),
        "ordered_document_identities_sha256": ordered_documents.hexdigest(),
        "exact_document_identity_unique": True,
        "exact_text_content_unique": True,
        "benchmark_decontamination_inherited_and_replayed": True,
        "development_partition_excluded": True,
        "tokenizer_measurement_only": True,
        "source_text_persisted_only_in_bounded_samples": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--maximum-bytes-per-shard", type=int, required=True)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = aggregate(
        args.samples_root,
        args.output,
        logical_shards=args.logical_shards,
        maximum_bytes_per_shard=args.maximum_bytes_per_shard,
        scratch_root=args.scratch_root,
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
