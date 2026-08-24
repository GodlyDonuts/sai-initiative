"""Verify every hash bucket in the cross-source subdocument decision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.cross_source_subdocument_decision import SCHEMA as DECISION_SCHEMA
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_subdocument_signature import HASH_BUCKETS
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-cross-source-subdocument-decision-aggregate-v1"


class CrossSourceSubdocumentDecisionAggregateError(RuntimeError):
    """Bucket coverage, component binding, or deletion custody differs."""


def _binding_key(row: dict[str, Any]) -> tuple[str, int]:
    name = row.get("component")
    priority = row.get("priority")
    shards = row.get("logical_shards")
    if (
        not isinstance(name, str)
        or not name
        or isinstance(priority, bool)
        or not isinstance(priority, int)
        or priority < 0
        or isinstance(shards, bool)
        or not isinstance(shards, int)
        or shards <= 0
        or not isinstance(row.get("aggregate_receipt_sha256"), str)
        or not isinstance(row.get("ordered_shard_receipts_sha256"), str)
    ):
        raise CrossSourceSubdocumentDecisionAggregateError(
            "component binding differs"
        )
    return name, priority


def build_aggregate(
    decision_root: Path,
    output: Path,
    bucket_indexes: list[int] | None = None,
) -> dict[str, Any]:
    """Replay exact deletion files and seal complete bucket coverage."""

    indexes = list(range(HASH_BUCKETS)) if bucket_indexes is None else bucket_indexes
    if (
        output.exists()
        or output.is_symlink()
        or not indexes
        or len(indexes) != len(set(indexes))
        or any(index < 0 or index >= HASH_BUCKETS for index in indexes)
    ):
        raise CrossSourceSubdocumentDecisionAggregateError(
            "aggregate arguments differ"
        )
    totals: Counter[str] = Counter()
    receipts = []
    canonical_bindings = None
    bucket_descriptors = []
    for bucket_index in sorted(indexes):
        root = decision_root / "buckets" / f"bucket_{bucket_index:02x}"
        receipt = _load_signed(root / "receipt.json", DECISION_SCHEMA)
        bindings = receipt.get("components")
        deletions = receipt.get("deletions")
        if (
            receipt.get("hash_bucket", {}).get("index") != bucket_index
            or receipt.get("hash_bucket", {}).get("buckets") != HASH_BUCKETS
            or receipt.get("cross_source_subdocument_decision_complete") is not True
            or receipt.get("decision_contains_source_text") is not False
            or not isinstance(bindings, list)
            or len(bindings) < 2
            or not isinstance(deletions, list)
        ):
            raise CrossSourceSubdocumentDecisionAggregateError(
                "decision bucket differs"
            )
        keys = [_binding_key(row) for row in bindings]
        if (
            len(keys) != len(set(keys))
            or [priority for _name, priority in keys] != list(range(len(keys)))
        ):
            raise CrossSourceSubdocumentDecisionAggregateError(
                "component priority differs"
            )
        stable_bindings = [
            {
                key: value
                for key, value in row.items()
                if key != "bucket_signatures"
            }
            for row in bindings
        ]
        if canonical_bindings is None:
            canonical_bindings = stable_bindings
        elif stable_bindings != canonical_bindings:
            raise CrossSourceSubdocumentDecisionAggregateError(
                "component bindings drift across buckets"
            )
        expected_keys = {
            (row["component"], shard)
            for row in bindings
            for shard in range(row["logical_shards"])
        }
        observed_keys = set()
        deletion_bytes = 0
        deletion_rows = 0
        deletion_root = root / "deletions"
        for descriptor in deletions:
            if not isinstance(descriptor, dict):
                raise CrossSourceSubdocumentDecisionAggregateError(
                    "deletion descriptor differs"
                )
            key = (descriptor.get("component"), descriptor.get("source_shard"))
            relative = descriptor.get("path")
            path = deletion_root / relative if isinstance(relative, str) else root
            try:
                path.resolve().relative_to(deletion_root.resolve())
            except ValueError as error:
                raise CrossSourceSubdocumentDecisionAggregateError(
                    "deletion path escapes its root"
                ) from error
            if (
                key in observed_keys
                or key not in expected_keys
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
                or descriptor.get("bytes")
                != descriptor.get("rows")
                * receipt.get("external_sort", {}).get("deletion_record_bytes", 0)
            ):
                raise CrossSourceSubdocumentDecisionAggregateError(
                    "deletion file differs"
                )
            observed_keys.add(key)
            deletion_bytes += descriptor["bytes"]
            deletion_rows += descriptor["rows"]
        if observed_keys != expected_keys:
            raise CrossSourceSubdocumentDecisionAggregateError(
                "deletion partition coverage differs"
            )
        if deletion_rows != receipt.get("counts", {}).get(
            "deletion_occurrences", 0
        ):
            raise CrossSourceSubdocumentDecisionAggregateError(
                "deletion accounting differs"
            )
        for key, value in receipt.get("counts", {}).items():
            totals[key] += value
        totals["deletion_file_bytes"] += deletion_bytes
        receipts.append(receipt["receipt_sha256"])
        bucket_descriptors.append(
            {
                "bucket": bucket_index,
                "receipt_sha256": receipt["receipt_sha256"],
                "deletion_rows": deletion_rows,
                "deletion_bytes": deletion_bytes,
                "ordered_deletion_descriptors_sha256": receipt[
                    "ordered_deletion_descriptors_sha256"
                ],
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_cross_source_subdocument_decision",
        "hash_partition": {
            "completed_bucket_indexes": sorted(indexes),
            "required_buckets": HASH_BUCKETS,
            "complete": set(indexes) == set(range(HASH_BUCKETS)),
        },
        "components": canonical_bindings,
        "totals": dict(sorted(totals.items())),
        "buckets": bucket_descriptors,
        "ordered_bucket_receipts_sha256": canonical_sha256(receipts),
        "ordered_bucket_descriptors_sha256": canonical_sha256(bucket_descriptors),
        "decision_contains_source_text": False,
        "cross_source_subdocument_decision_complete": (
            set(indexes) == set(range(HASH_BUCKETS))
        ),
        "rewrite_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_aggregate(args.decision_root, args.output)
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
