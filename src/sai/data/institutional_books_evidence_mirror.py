"""Mirror only source-safe Institutional Books evidence into durable storage."""

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

SCHEMA = "sai-institutional-books-durable-evidence-mirror-v1"
SAFE_FILES = (
    (
        "materialized_aggregate",
        "institutional-books-strict-english-materialized-20260826-r1/aggregate.json",
    ),
    (
        "mechanical_gate_aggregate",
        "institutional-books-mechanical-gate-20260826-r1/aggregate.json",
    ),
    (
        "mechanical_filter_aggregate",
        "institutional-books-mechanical-filtered-20260826-r1/aggregate.json",
    ),
    (
        "semantic_population_receipt",
        "institutional-books-semantic-population-20260826-r1/receipt.json",
    ),
    (
        "semantic_aggregate",
        "institutional-books-semantic-judgments-20260826-r1/aggregate.json",
    ),
    (
        "semantic_decision_receipt",
        "institutional-books-semantic-decisions-20260826-r1/receipt.json",
    ),
    (
        "semantic_decisions",
        "institutional-books-semantic-decisions-20260826-r1/decisions.jsonl",
    ),
    (
        "independent_population_receipt",
        "institutional-books-independent-population-20260826-r1/receipt.json",
    ),
    (
        "independent_aggregate",
        "institutional-books-independent-nemotron-20260826-r1/aggregate.json",
    ),
    (
        "agreement_receipt",
        "institutional-books-independent-agreement-20260826-r1/receipt.json",
    ),
    (
        "agreement_manifest",
        "institutional-books-independent-agreement-20260826-r1/agreement.jsonl",
    ),
    (
        "full_decontamination_receipt",
        "institutional-books-full-decontamination-20260826-r1/receipt.json",
    ),
    (
        "full_decontamination_decisions",
        "institutional-books-full-decontamination-20260826-r1/decisions.jsonl",
    ),
    (
        "benchmark_disjoint_books",
        (
            "institutional-books-full-decontamination-20260826-r1/"
            "benchmark_disjoint_books.jsonl"
        ),
    ),
)


class InstitutionalBooksEvidenceMirrorError(RuntimeError):
    """Source-safe evidence input or durable copy differs."""


def _copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_nlink != 1
        or destination.exists()
        or destination.is_symlink()
    ):
        raise InstitutionalBooksEvidenceMirrorError("evidence file differs")
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
        raise InstitutionalBooksEvidenceMirrorError("evidence copy differs")
    return {
        "bytes": destination.stat().st_size,
        "sha256": source_sha256,
    }


def mirror_evidence(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Copy the fixed source-safe allowlist and seal a durable receipt."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksEvidenceMirrorError("evidence root exists")
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
            "status": "complete_source_safe_durable_book_evidence_mirror",
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records),
            "ordered_records_sha256": canonical_sha256(
                [record["record_sha256"] for record in records]
            ),
            "candidate_excerpts_copied": False,
            "compiler_evidence_quotes_copied": False,
            "full_book_text_copied": False,
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
