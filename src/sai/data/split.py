"""Deterministically separate admitted Sai documents into train and development."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-document-split-receipt-v1"


class DocumentSplitError(RuntimeError):
    """The admitted corpus, split geometry, output, or receipt differs."""


def split(
    source: Path,
    train: Path,
    development: Path,
    receipt: Path,
    *,
    development_modulus: int = 100,
    development_bucket: int = 0,
) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise DocumentSplitError("admitted corpus is missing or unsafe")
    if any(path.exists() for path in (train, development, receipt)):
        raise DocumentSplitError("document split output already exists")
    if (
        isinstance(development_modulus, bool)
        or not isinstance(development_modulus, int)
        or development_modulus <= 1
        or isinstance(development_bucket, bool)
        or not isinstance(development_bucket, int)
        or not 0 <= development_bucket < development_modulus
    ):
        raise DocumentSplitError("document split geometry differs")
    train.parent.mkdir(parents=True, exist_ok=True)
    development.parent.mkdir(parents=True, exist_ok=True)
    train_stage = train.with_name(f".{train.name}.partial.{os.getpid()}")
    development_stage = development.with_name(
        f".{development.name}.partial.{os.getpid()}"
    )
    train_count = development_count = 0
    train_identity = hashlib.sha256()
    development_identity = hashlib.sha256()
    try:
        with (
            source.open() as source_handle,
            train_stage.open("w") as train_handle,
            development_stage.open("w") as development_handle,
        ):
            for line in source_handle:
                if not line.strip():
                    continue
                try:
                    row = normalize_document(json.loads(line))
                except (json.JSONDecodeError, RuntimeError) as error:
                    raise DocumentSplitError("admitted corpus row differs") from error
                identity = row["identity_sha256"]
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                is_development = (
                    int(identity[:16], 16) % development_modulus == development_bucket
                )
                if is_development:
                    development_handle.write(encoded)
                    development_identity.update(bytes.fromhex(identity))
                    development_count += 1
                else:
                    train_handle.write(encoded)
                    train_identity.update(bytes.fromhex(identity))
                    train_count += 1
    except BaseException:
        train_stage.unlink(missing_ok=True)
        development_stage.unlink(missing_ok=True)
        raise
    if not train_count or not development_count:
        train_stage.unlink(missing_ok=True)
        development_stage.unlink(missing_ok=True)
        raise DocumentSplitError("document split produced an empty population")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "method": "document_identity_modulus",
        "development_modulus": development_modulus,
        "development_bucket": development_bucket,
        "exact_duplicates_previously_removed": True,
        "near_duplicate_cluster_split_qualified": False,
        "scientific_promotion_allowed": False,
        "train": {
            "path": str(train.resolve()),
            "documents": train_count,
            "bytes": train_stage.stat().st_size,
            "sha256": sha256_file(train_stage),
            "identity_sha256": train_identity.hexdigest(),
        },
        "development": {
            "path": str(development.resolve()),
            "documents": development_count,
            "bytes": development_stage.stat().st_size,
            "sha256": sha256_file(development_stage),
            "identity_sha256": development_identity.hexdigest(),
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_stage = receipt.with_name(f".{receipt.name}.partial.{os.getpid()}")
    receipt_stage.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(train_stage, train)
    os.replace(development_stage, development)
    os.replace(receipt_stage, receipt)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = split(args.source, args.train, args.development, args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
