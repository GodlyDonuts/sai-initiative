"""Replay every bounded sample used to fit Sai's production 1B tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_tokenizer_sample import SAMPLE_NAME, STATUS
from sai.data.one_b_tokenizer_sample import SCHEMA as SHARD_SCHEMA
from sai.data.token_stream import (
    TOKENIZER_ROW_SCHEMA,
    canonical_sha256,
    normalize_tokenizer_document,
    sha256_file,
)

SCHEMA = "sai-1b-production-tokenizer-sample-aggregate-v1"


class OneBTokenizerSampleAggregateError(RuntimeError):
    """A tokenizer sample receipt, row, identity, or aggregate differs."""


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBTokenizerSampleAggregateError("sample receipt differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != SHARD_SCHEMA
        or value.get("status") != STATUS
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBTokenizerSampleAggregateError("sample receipt differs")
    return value


def _paths(root: Path) -> list[Path]:
    paths = [root / "books" / f"shard_{index:05d}" for index in range(64)]
    paths.extend(root / component for component in ("pleias", "code", "connections"))
    return paths


def aggregate(root: Path, output: Path) -> dict[str, Any]:
    """Verify exact sample hashes, rows, source identities, and component coverage."""

    if output.exists() or output.is_symlink():
        raise OneBTokenizerSampleAggregateError("sample aggregate output exists")
    counts: Counter[str] = Counter()
    identities = set()
    ordered_receipts = []
    ordered_samples = []
    component_receipts: Counter[str] = Counter()
    for directory in _paths(root):
        receipt = _load_receipt(directory / "receipt.json")
        descriptor = receipt.get("sample", {})
        sample = directory / descriptor.get("path", "")
        component = receipt.get("component")
        if (
            descriptor.get("path") != SAMPLE_NAME
            or component not in {"books", "pleias", "code", "connections"}
            or not sample.is_file()
            or sample.is_symlink()
            or sample.stat().st_nlink != 1
            or sample.stat().st_size != descriptor.get("bytes")
            or sha256_file(sample) != descriptor.get("sha256")
        ):
            raise OneBTokenizerSampleAggregateError("sample file differs")
        digest = hashlib.sha256()
        rows = text_bytes = 0
        with sample.open(encoding="utf-8") as handle:
            for line in handle:
                digest.update(line.encode())
                try:
                    document = normalize_tokenizer_document(json.loads(line))
                except (json.JSONDecodeError, ValueError) as error:
                    raise OneBTokenizerSampleAggregateError(
                        "sample row differs"
                    ) from error
                if document.get("schema") != TOKENIZER_ROW_SCHEMA:
                    raise OneBTokenizerSampleAggregateError("sample row differs")
                identity = document["identity_sha256"]
                if identity in identities:
                    raise OneBTokenizerSampleAggregateError(
                        "sample identities overlap"
                    )
                identities.add(identity)
                rows += 1
                text_bytes += len(document["text"].encode())
        if (
            rows != descriptor.get("documents")
            or text_bytes != descriptor.get("text_utf8_bytes")
            or digest.hexdigest() != descriptor.get("ordered_jsonl_sha256")
        ):
            raise OneBTokenizerSampleAggregateError("sample accounting differs")
        counts["documents"] += rows
        counts["text_utf8_bytes"] += text_bytes
        counts["jsonl_bytes"] += descriptor["bytes"]
        counts[f"component::{component}::documents"] += rows
        counts[f"component::{component}::text_utf8_bytes"] += text_bytes
        component_receipts[component] += 1
        ordered_receipts.append(receipt["receipt_sha256"])
        ordered_samples.append(
            {
                "path": str(sample.resolve()),
                "bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
            }
        )
    if component_receipts != Counter(
        {"books": 64, "pleias": 1, "code": 1, "connections": 1}
    ):
        raise OneBTokenizerSampleAggregateError("sample component coverage differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_1b_tokenizer_sample_aggregate",
        "sample_receipts": len(ordered_receipts),
        "ordered_sample_receipts_sha256": canonical_sha256(ordered_receipts),
        "ordered_samples": ordered_samples,
        "ordered_samples_sha256": canonical_sha256(ordered_samples),
        "component_receipts": dict(sorted(component_receipts.items())),
        "counts": dict(sorted(counts.items())),
        "sample_identities_globally_unique": True,
        "all_source_text_bounded_to_tokenizer_samples": True,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root, args.output)
    print(json.dumps({"receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
