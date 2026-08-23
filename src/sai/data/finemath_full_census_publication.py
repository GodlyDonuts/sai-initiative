"""Publish source-safe evidence from the full FineMath-4plus census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.finemath_full_census import (
    AGGREGATE_SCHEMA,
    EXPECTED_SHARDS,
    REPOSITORY,
    REVISION,
    SHARD_SCHEMA,
    SUBSET,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-finemath-full-census-publication-v1"


class FineMathFullCensusPublicationError(RuntimeError):
    """The aggregate, shard evidence, or source-safe boundary differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise FineMathFullCensusPublicationError("publication input is unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FineMathFullCensusPublicationError(
            "publication input is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise FineMathFullCensusPublicationError("publication input differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("source_text_persisted") is not False
        or payload.get("training_ready") is not False
    ):
        raise FineMathFullCensusPublicationError("publication receipt differs")
    return payload


def summarize_publication(
    aggregate: dict[str, Any], shard_receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify exact shard custody and return a compact text-free summary."""

    source = aggregate.get("source", {})
    summary = aggregate.get("summary", {})
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete_source_safe_full_mechanical_census"
        or source.get("repository") != REPOSITORY
        or source.get("revision") != REVISION
        or source.get("subset") != SUBSET
        or aggregate.get("all_source_file_sha256_values_verified") is not True
        or aggregate.get("all_source_rows_scanned") is not True
        or aggregate.get("global_exact_duplicate_census_complete") is not True
        or aggregate.get("source_text_persisted") is not False
        or aggregate.get("training_ready") is not False
        or len(shard_receipts) != EXPECTED_SHARDS
    ):
        raise FineMathFullCensusPublicationError("aggregate evidence differs")
    ordered_hashes = []
    for index, receipt in enumerate(shard_receipts):
        shard_source = receipt.get("source", {})
        if (
            receipt.get("schema") != SHARD_SCHEMA
            or receipt.get("status") != "complete_source_safe_mechanical_census_shard"
            or shard_source.get("shard_index") != index
            or shard_source.get("repository") != REPOSITORY
            or shard_source.get("revision") != REVISION
            or receipt.get("source_file_sha256_verified") is not True
            or receipt.get("full_shard_scanned") is not True
            or receipt.get("source_text_persisted") is not False
            or receipt.get("training_ready") is not False
        ):
            raise FineMathFullCensusPublicationError("shard evidence differs")
        ordered_hashes.append(receipt["receipt_sha256"])
    if aggregate.get("shards", {}).get("ordered_receipts_sha256") != canonical_sha256(
        ordered_hashes
    ):
        raise FineMathFullCensusPublicationError("ordered shard custody differs")
    return {
        "source": source,
        "rows": summary.get("rows"),
        "text_utf8_bytes": summary.get("text_utf8_bytes"),
        "token_count": summary.get("token_count"),
        "languages": summary.get("languages"),
        "language_score_bins": summary.get("language_score_bins"),
        "token_count_bins": summary.get("token_count_bins"),
        "int_scores": summary.get("int_scores"),
        "found_math_true_rows": summary.get("found_math_true_rows"),
        "nonzero_math_feature_rows": summary.get("nonzero_math_feature_rows"),
        "measurement_profiles": summary.get("measurement_profiles"),
        "measurement_profile_text_utf8_bytes": summary.get(
            "measurement_profile_text_utf8_bytes"
        ),
        "measurement_profile_tokens": summary.get("measurement_profile_tokens"),
        "exact_content_multiplicity": aggregate.get("exact_content_multiplicity"),
        "normalized_content_multiplicity": aggregate.get(
            "normalized_content_multiplicity"
        ),
        "aggregate_receipt_sha256": aggregate["receipt_sha256"],
        "ordered_shard_receipts_sha256": canonical_sha256(ordered_hashes),
    }


def build_publication(evidence_root: Path, output_path: Path) -> dict[str, Any]:
    """Replay the aggregate and all 64 shard receipts into one publication."""

    if output_path.exists() or output_path.is_symlink():
        raise FineMathFullCensusPublicationError("publication output exists")
    aggregate_path = evidence_root / "aggregate.json"
    aggregate = _load_signed(aggregate_path, AGGREGATE_SCHEMA)
    receipts = []
    receipt_files = []
    for index in range(EXPECTED_SHARDS):
        path = evidence_root / "shards" / f"shard_{index:05d}" / "receipt.json"
        receipts.append(_load_signed(path, SHARD_SCHEMA))
        receipt_files.append(
            {
                "path": str(path.relative_to(evidence_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_source_safe_full_census_publication",
        **summarize_publication(aggregate, receipts),
        "artifacts": {
            "aggregate": {
                "path": aggregate_path.name,
                "bytes": aggregate_path.stat().st_size,
                "sha256": sha256_file(aggregate_path),
            },
            "shard_receipts": receipt_files,
            "shard_receipt_files_sha256": canonical_sha256(receipt_files),
        },
        "source_text_persisted": False,
        "measurement_profiles_are_training_admissions": False,
        "global_exact_deduplication_applied": False,
        "benchmark_decontamination_complete": False,
        "global_semantic_deduplication_complete": False,
        "hermes_full_population_quality_compilation_complete": False,
        "rights_admission_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_publication(args.evidence_root, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
