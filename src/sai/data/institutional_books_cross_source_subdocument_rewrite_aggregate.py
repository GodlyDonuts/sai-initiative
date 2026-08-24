"""Verify all private cross-source-rewritten Institutional Books shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as DECISION_SCHEMA,
)
from sai.data.foundation_source_split import POLICY_SHA256 as SPLIT_POLICY_SHA256
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    SHARD_SCHEMA,
)
from sai.data.institutional_books_materializer import _load_json, _valid_receipt
from sai.data.institutional_books_mechanical_filter import (
    AGGREGATE_SCHEMA as FILTER_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_subdocument_signature import (
    COMPONENT,
    _clean_books,
    _filtered_shard,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-cross-source-rewritten-aggregate-v1"


class InstitutionalBooksCrossSourceSubdocumentRewriteAggregateError(RuntimeError):
    """Private shard coverage, deletion accounting, or custody differs."""


def _metadata_coverage_complete(totals: Counter[str]) -> bool:
    """Require agreed genre, domain, and curriculum evidence for every book."""

    documents = totals["documents"]

    def dimension(prefix: str) -> int:
        values = [
            value
            for key, value in totals.items()
            if key.startswith(prefix) and key.endswith("::documents")
        ]
        if not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            return -1
        return sum(values)

    return (
        documents > 0
        and totals["documents_with_consensus_curriculum_metadata"] == documents
        and dimension("semantic_genre::") == documents
        and dimension("semantic_domain::") >= documents
        and dimension("curriculum_band_vote::") >= documents
    )


def build_aggregate(
    filtered_root: Path,
    decontamination_root: Path,
    decision_root: Path,
    rewrite_root: Path,
    output: Path,
    logical_shards: int,
) -> dict[str, Any]:
    """Verify exact coverage over all benchmark-disjoint private books."""

    if output.exists() or output.is_symlink() or logical_shards <= 0:
        raise InstitutionalBooksCrossSourceSubdocumentRewriteAggregateError(
            "aggregate arguments differ"
        )
    clean, decontamination = _clean_books(decontamination_root)
    filtered = _load_json(filtered_root / "aggregate.json")
    decision = _load_signed(decision_root / "aggregate.json", DECISION_SCHEMA)
    expected_decisions = decision.get("totals", {}).get(
        f"component::{COMPONENT}::deletion_occurrences", 0
    )
    if (
        not _valid_receipt(filtered, FILTER_AGGREGATE_SCHEMA)
        or filtered.get("shards", {}).get("logical_shards") != logical_shards
        or decision.get("cross_source_subdocument_decision_complete") is not True
        or isinstance(expected_decisions, bool)
        or not isinstance(expected_decisions, int)
        or expected_decisions < 0
    ):
        raise InstitutionalBooksCrossSourceSubdocumentRewriteAggregateError(
            "aggregate source differs"
        )
    totals: Counter[str] = Counter()
    receipts = []
    identity_digests = []
    for shard_index in range(logical_shards):
        _path, source = _filtered_shard(filtered_root, logical_shards, shard_index)
        root = rewrite_root / "shards" / f"shard_{shard_index:05d}"
        receipt = _load_json(root / "receipt.json")
        descriptor = receipt.get("output")
        output_valid = (
            descriptor is None and receipt.get("counts", {}).get("documents", 0) == 0
        )
        if isinstance(descriptor, dict):
            path = root / descriptor.get("path", "")
            output_valid = (
                path.is_file()
                and not path.is_symlink()
                and path.stat().st_nlink == 1
                and path.stat().st_size == descriptor.get("bytes")
                and sha256_file(path) == descriptor.get("sha256")
                and descriptor.get("rows") == receipt.get("counts", {}).get("documents")
            )
            if output_valid:
                totals["private_output_file_bytes"] += descriptor["bytes"]
        counts = receipt.get("counts")
        if (
            not _valid_receipt(receipt, SHARD_SCHEMA)
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("filtered_shard_receipt_sha256")
            != source["receipt_sha256"]
            or receipt.get("source", {}).get("decontamination_receipt_sha256")
            != decontamination["receipt_sha256"]
            or receipt.get("source", {}).get(
                "cross_source_decision_aggregate_receipt_sha256"
            )
            != decision["receipt_sha256"]
            or receipt.get("private_storage_only") is not True
            or receipt.get("huggingface_redistribution_authorized") is not False
            or receipt.get("cross_source_subdocument_deduplication_complete")
            is not True
            or receipt.get("source_disjoint_split_complete") is not True
            or receipt.get("source_disjoint_split_policy_sha256") != SPLIT_POLICY_SHA256
            or not isinstance(counts, dict)
            or counts.get("filtered_source_rows") != source.get("retained_rows", 0)
            or counts.get("output_text_utf8_bytes", 0)
            > counts.get("input_text_utf8_bytes", 0)
            or not output_valid
        ):
            raise InstitutionalBooksCrossSourceSubdocumentRewriteAggregateError(
                "private rewrite shard differs"
            )
        for key, value in counts.items():
            totals[key] += value
        receipts.append(receipt["receipt_sha256"])
        identity_digests.append(receipt["ordered_document_identities_sha256"])
    if (
        totals["documents"] != len(clean)
        or totals["candidate_deletion_chunks"] != expected_decisions
        or totals["split::train::documents"] + totals["split::development::documents"]
        != totals["documents"]
        or totals["split::train::text_utf8_bytes"]
        + totals["split::development::text_utf8_bytes"]
        != totals["output_text_utf8_bytes"]
    ):
        raise InstitutionalBooksCrossSourceSubdocumentRewriteAggregateError(
            "private global accounting differs"
        )
    if not _metadata_coverage_complete(totals):
        raise InstitutionalBooksCrossSourceSubdocumentRewriteAggregateError(
            "semantic curriculum metadata coverage differs"
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_institutional_books_cross_source_rewritten",
        "source": {
            "filtered_aggregate_receipt_sha256": filtered["receipt_sha256"],
            "decontamination_receipt_sha256": decontamination["receipt_sha256"],
            "cross_source_decision_aggregate_receipt_sha256": decision[
                "receipt_sha256"
            ],
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
            "ordered_document_partition_digests_sha256": canonical_sha256(
                identity_digests
            ),
        },
        "totals": dict(sorted(totals.items())),
        "complete_benchmark_disjoint_book_coverage": True,
        "private_storage_only": True,
        "huggingface_redistribution_authorized": False,
        "benchmark_decontamination_complete": True,
        "cross_source_subdocument_deduplication_complete": True,
        "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
        "source_disjoint_split_complete": True,
        "semantic_quality_metadata_complete": True,
        "curriculum_metadata_complete": True,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--rewrite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    args = parser.parse_args()
    result = build_aggregate(
        args.filtered_root,
        args.decontamination_root,
        args.decision_root,
        args.rewrite_root,
        args.output,
        args.logical_shards,
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
