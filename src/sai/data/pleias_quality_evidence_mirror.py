"""Mirror only source-safe PleIAs quality evidence into durable storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-durable-quality-evidence-mirror-v1"
SAFE_FILES = (
    ("metadata_census", "pleias-metadata-census-20260826-r2/aggregate.json"),
    ("quality_policy", "pleias-quality-core-policy-20260826-r1.json"),
    (
        "bounded_mechanical_aggregate",
        "pleias-bounded-mechanical-20260826-r1/aggregate.json",
    ),
    ("semantic_population", "pleias-semantic-sample-20260826-r1/receipt.json"),
    ("semantic_aggregate", "pleias-semantic-hermes-aggregate-20260826-r1.json"),
    (
        "independent_population",
        "pleias-independent-review-population-20260826-r1/receipt.json",
    ),
    (
        "independent_comparison",
        "pleias-independent-comparison-20260826-r1.json",
    ),
    (
        "semantic_stratum_decision",
        "pleias-semantic-stratum-decision-20260826-r1.json",
    ),
    (
        "full_decontamination_aggregate",
        "pleias-full-decontamination-20260826-r1/aggregate.json",
    ),
    (
        "global_exact_decision",
        "pleias-global-exact-decision-20260826-r1/receipt.json",
    ),
    (
        "global_exact_keep_database",
        "pleias-global-exact-decision-20260826-r1/global_exact_keep.sqlite3",
    ),
    (
        "global_exact_aggregate",
        "pleias-global-exact-filtered-20260826-r1/aggregate.json",
    ),
)


class PleiasQualityEvidenceMirrorError(RuntimeError):
    """Source-safe PleIAs evidence input or durable copy differs."""


def _copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_nlink != 1
        or destination.exists()
        or destination.is_symlink()
    ):
        raise PleiasQualityEvidenceMirrorError("evidence file differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial.{uuid.uuid4().hex}"
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    source_sha256 = sha256_file(source)
    if (
        destination.stat().st_size != source.stat().st_size
        or sha256_file(destination) != source_sha256
    ):
        destination.unlink(missing_ok=True)
        raise PleiasQualityEvidenceMirrorError("evidence copy differs")
    return {"bytes": destination.stat().st_size, "sha256": source_sha256}


def mirror_evidence(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Copy the fixed text-free allowlist and seal exact hashes."""

    if output_root.exists() or output_root.is_symlink():
        raise PleiasQualityEvidenceMirrorError("evidence root exists")
    output_root.mkdir(parents=True)
    try:
        records = []
        for label, relative in SAFE_FILES:
            source = source_root / relative
            destination = output_root / "files" / label / Path(relative).name
            descriptor = _copy_exact(source, destination)
            record = {
                "label": label,
                "source_relative_path": relative,
                "durable_relative_path": str(destination.relative_to(output_root)),
                **descriptor,
                "source_text_persisted": False,
            }
            record["record_sha256"] = canonical_sha256(record)
            records.append(record)
        payload = {
            "schema": SCHEMA,
            "status": "complete_source_safe_durable_pleias_quality_evidence_mirror",
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(row["bytes"] for row in records),
            "ordered_records_sha256": canonical_sha256(
                [row["record_sha256"] for row in records]
            ),
            "semantic_excerpts_copied": False,
            "compiler_judgments_copied": False,
            "candidate_text_copied": False,
            "source_text_persisted": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = mirror_evidence(args.source_root, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
