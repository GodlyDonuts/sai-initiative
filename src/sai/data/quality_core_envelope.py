"""Freeze a byte-exact, fail-closed envelope for the Sai quality core."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.hf_source_removal_receipt import SCHEMA as REMOVAL_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-quality-core-candidate-envelope-v1"


class QualityCoreEnvelopeError(RuntimeError):
    """The source inventory, removal receipt, or target differs."""


def _load_signed_removal(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise QualityCoreEnvelopeError("removal receipt is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise QualityCoreEnvelopeError("removal receipt is invalid") from error
    if not isinstance(payload, dict):
        raise QualityCoreEnvelopeError("removal receipt is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != REMOVAL_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("status")
        != "complete_verified_recoverable_source_prefix_removal"
        or payload.get("remaining_prefix_files") != 0
        or payload.get("recoverable_from_repository_history") is not True
        or payload.get("training_ready") is not False
    ):
        raise QualityCoreEnvelopeError("removal receipt differs")
    return payload


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise QualityCoreEnvelopeError("source manifest is missing or unsafe")
    rows = []
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QualityCoreEnvelopeError("source manifest row differs")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise QualityCoreEnvelopeError("source manifest is invalid") from error
    if not rows:
        raise QualityCoreEnvelopeError("source manifest is empty")
    return rows


def build_envelope_payload(
    manifest_rows: list[dict[str, Any]],
    removal: dict[str, Any],
    target_bytes: int,
    bulk_source_id: str,
) -> dict[str, Any]:
    """Build the exact post-removal candidate envelope without admitting data."""

    if (
        isinstance(target_bytes, bool)
        or not isinstance(target_bytes, int)
        or target_bytes <= 0
        or not isinstance(bulk_source_id, str)
        or not bulk_source_id
    ):
        raise QualityCoreEnvelopeError("quality-core geometry differs")
    prefix = removal.get("prefix")
    if not isinstance(prefix, str) or not prefix:
        raise QualityCoreEnvelopeError("removal prefix differs")
    removed_files = 0
    removed_bytes = 0
    remaining_files = 0
    remaining_bytes = 0
    by_source_files: Counter[str] = Counter()
    by_source_bytes: Counter[str] = Counter()
    seen_paths = set()
    for row in manifest_rows:
        path = row.get("destination_path")
        source_id = row.get("source_id")
        size = row.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or path in seen_paths
            or not isinstance(source_id, str)
            or not source_id
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or row.get("raw_source_is_training_ready") is not False
        ):
            raise QualityCoreEnvelopeError("source manifest identity differs")
        seen_paths.add(path)
        if path == prefix or path.startswith(f"{prefix}/"):
            removed_files += 1
            removed_bytes += size
            continue
        remaining_files += 1
        remaining_bytes += size
        by_source_files[source_id] += 1
        by_source_bytes[source_id] += size
    post = removal.get("post_removal_source_tree")
    if (
        removed_files != removal.get("removed_objects")
        or removed_bytes != removal.get("removed_bytes")
        or not isinstance(post, dict)
        or remaining_files != post.get("data_files")
        or remaining_bytes != post.get("data_bytes")
    ):
        raise QualityCoreEnvelopeError("post-removal accounting differs")
    if bulk_source_id not in by_source_bytes:
        raise QualityCoreEnvelopeError("bulk source is absent")
    bulk_bytes = by_source_bytes[bulk_source_id]
    nonbulk_bytes = remaining_bytes - bulk_bytes
    provisional_bulk_ceiling = max(0, target_bytes - nonbulk_bytes)
    source_rows = [
        {
            "source_id": source_id,
            "candidate_files": by_source_files[source_id],
            "candidate_bytes": size,
            "automatic_admission": False,
            "training_ready": False,
        }
        for source_id, size in sorted(
            by_source_bytes.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "schema": SCHEMA,
        "status": "complete_nontraining_quality_core_candidate_envelope",
        "target": {
            "maximum_bytes": target_bytes,
            "is_padding_floor": False,
            "admit_less_when_quality_gates_require": True,
        },
        "post_removal_lake": {
            "repository": removal.get("repository"),
            "revision": removal.get("verified_current_revision"),
            "candidate_files": remaining_files,
            "candidate_bytes": remaining_bytes,
            "removal_receipt_sha256": removal.get("receipt_sha256"),
        },
        "bulk_source": {
            "source_id": bulk_source_id,
            "raw_candidate_files": by_source_files[bulk_source_id],
            "raw_candidate_bytes": bulk_bytes,
            "provisional_maximum_bytes_if_all_nonbulk_candidates_survive": min(
                bulk_bytes, provisional_bulk_ceiling
            ),
            "provisional_excess_bytes_if_all_nonbulk_candidates_survive": max(
                0, bulk_bytes - provisional_bulk_ceiling
            ),
            "maximum_is_admission": False,
        },
        "nonbulk_sources": {
            "candidate_sources": len(by_source_bytes) - 1,
            "raw_candidate_files": remaining_files - by_source_files[bulk_source_id],
            "raw_candidate_bytes": nonbulk_bytes,
            "all_nonbulk_candidates_are_admitted": False,
        },
        "sources": source_rows,
        "raw_bytes_are_training_bytes": False,
        "quality_gates_must_reduce_or_transform_candidates": True,
        "automatic_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }


def build_envelope(
    manifest_path: Path,
    removal_receipt_path: Path,
    target_bytes: int,
    bulk_source_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Load exact inputs and atomically write a source-safe envelope receipt."""

    if output_path.exists() or output_path.is_symlink():
        raise QualityCoreEnvelopeError("quality-core envelope output exists")
    rows = _load_manifest(manifest_path)
    removal = _load_signed_removal(removal_receipt_path)
    payload = build_envelope_payload(rows, removal, target_bytes, bulk_source_id)
    payload["inputs"] = {
        "source_manifest_file_sha256": sha256_file(manifest_path),
        "removal_receipt_file_sha256": sha256_file(removal_receipt_path),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--removal-receipt", type=Path, required=True)
    parser.add_argument("--target-bytes", type=int, default=2_000_000_000_000)
    parser.add_argument("--bulk-source-id", default="pleias_common_corpus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_envelope(
        args.source_manifest,
        args.removal_receipt,
        args.target_bytes,
        args.bulk_source_id,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "target": result["target"],
                "bulk_source": result["bulk_source"],
                "nonbulk_sources": result["nonbulk_sources"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
