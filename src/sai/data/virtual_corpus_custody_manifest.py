"""Hash-manifest the exact virtual Sai foundation corpus custody graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as CROSS_DECISION_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA as PLEIAS_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file
from sai.data.virtual_foundation_corpus_ledger import SCHEMA as LEDGER_SCHEMA

SCHEMA = "sai-virtual-corpus-custody-manifest-v1"


class VirtualCorpusCustodyManifestError(RuntimeError):
    """Source lake, component receipt, ledger, or runtime binding differs."""


def _regular(path: Path, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size <= 0
    ):
        raise VirtualCorpusCustodyManifestError(f"{label} is unsafe")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_manifest(path: Path) -> dict[str, Any]:
    descriptor = _regular(path, "source manifest")
    rows = 0
    physical_bytes = 0
    repositories: set[str] = set()
    destinations: set[str] = set()
    ordered = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise VirtualCorpusCustodyManifestError(
                    f"source manifest row {line_number} differs"
                ) from error
            repository = row.get("source_repository")
            destination = row.get("destination_path")
            content_sha256 = row.get("sha256")
            size = row.get("bytes")
            if (
                row.get("schema") != "sai-hf-materialized-source-file-v1"
                or not isinstance(repository, str)
                or not repository
                or not isinstance(destination, str)
                or not destination
                or destination in destinations
                or not isinstance(content_sha256, str)
                or len(content_sha256) != 64
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or row.get("raw_source_is_training_ready") is not False
            ):
                raise VirtualCorpusCustodyManifestError(
                    f"source manifest row {line_number} differs"
                )
            rows += 1
            physical_bytes += size
            repositories.add(repository)
            destinations.add(destination)
            ordered.append(
                {
                    "destination_path": destination,
                    "bytes": size,
                    "sha256": content_sha256,
                }
            )
    if rows == 0:
        raise VirtualCorpusCustodyManifestError("source manifest is empty")
    return {
        **descriptor,
        "rows": rows,
        "physical_source_object_bytes": physical_bytes,
        "source_repositories": sorted(repositories),
        "ordered_object_identities_sha256": canonical_sha256(ordered),
    }


def build_manifest(
    source_manifest_path: Path,
    book_aggregate_path: Path,
    pleias_aggregate_path: Path,
    cross_decision_path: Path,
    ledger_path: Path,
    output: Path,
    durable_output: Path,
    *,
    runtime_commit: str,
) -> dict[str, Any]:
    """Bind every exact component needed to reconstruct the admitted bytes."""

    if (
        output.exists()
        or output.is_symlink()
        or durable_output.exists()
        or durable_output.is_symlink()
        or output.resolve() == durable_output.resolve()
        or not isinstance(runtime_commit, str)
        or len(runtime_commit) != 40
        or any(character not in "0123456789abcdef" for character in runtime_commit)
    ):
        raise VirtualCorpusCustodyManifestError("custody arguments differ")
    source_manifest = _source_manifest(source_manifest_path)
    books = _load_signed(book_aggregate_path, BOOK_SCHEMA)
    pleias = _load_signed(pleias_aggregate_path, PLEIAS_SCHEMA)
    cross = _load_signed(cross_decision_path, CROSS_DECISION_SCHEMA)
    ledger = _load_signed(ledger_path, LEDGER_SCHEMA)
    components = {
        row.get("component"): row
        for row in ledger.get("components", [])
        if isinstance(row, dict)
    }
    if (
        set(components) != {"institutional_books", "pleias_common_corpus"}
        or components["institutional_books"].get("aggregate_receipt_sha256")
        != books.get("receipt_sha256")
        or components["pleias_common_corpus"].get("aggregate_receipt_sha256")
        != pleias.get("receipt_sha256")
        or pleias.get("source", {}).get("cross_decision_aggregate_receipt_sha256")
        != cross.get("receipt_sha256")
        or ledger.get("byte_ceiling_respected") is not True
        or ledger.get("pleias_virtual_reconstruction_complete") is not True
        or ledger.get("pleias_payload_materialization_complete") is not False
        or ledger.get("training_ready") is not False
        or books.get("cross_source_subdocument_deduplication_complete") is not True
        or pleias.get("cross_source_subdocument_deduplication_complete") is not True
        or pleias.get("source_text_persisted") is not False
        or cross.get("cross_source_subdocument_decision_complete") is not True
        or cross.get("decision_contains_source_text") is not False
    ):
        raise VirtualCorpusCustodyManifestError("custody component differs")
    inputs = [
        {
            "artifact": "source_lake_manifest",
            **source_manifest,
        },
        {
            "artifact": "private_book_aggregate",
            **_regular(book_aggregate_path, "book aggregate"),
            "receipt_sha256": books["receipt_sha256"],
        },
        {
            "artifact": "virtual_pleias_aggregate",
            **_regular(pleias_aggregate_path, "PleIAs aggregate"),
            "receipt_sha256": pleias["receipt_sha256"],
        },
        {
            "artifact": "cross_source_decision_aggregate",
            **_regular(cross_decision_path, "cross decision aggregate"),
            "receipt_sha256": cross["receipt_sha256"],
        },
        {
            "artifact": "foundation_ledger",
            **_regular(ledger_path, "foundation ledger"),
            "receipt_sha256": ledger["receipt_sha256"],
        },
    ]
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_virtual_corpus_custody_manifest",
        "runtime_commit": runtime_commit,
        "inputs": inputs,
        "ordered_inputs_sha256": canonical_sha256(inputs),
        "foundation": ledger["totals"],
        "reconstruction_contract": {
            "pleias_source": "pinned_huggingface_source_objects",
            "pleias_selection": "exact_parent_and_row_locators",
            "full_document_benchmark_boundary_applied": True,
            "internal_subdocument_decisions_applied": True,
            "cross_source_subdocument_decisions_applied": True,
            "source_disjoint_split_applied": True,
            "final_content_sha256_bound_per_row": True,
            "source_text_persisted_in_manifest": False,
        },
        "all_irreplaceable_receipts_hash_manifested": True,
        "custody_locations": [str(output.resolve()), str(durable_output.resolve())],
        "durable_evidence_copy_complete": True,
        "huggingface_metadata_publication_complete": False,
        "final_tokenization_complete": False,
        "curriculum_schedule_complete": False,
        "final_corpus_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(durable_output, payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--book-aggregate", type=Path, required=True)
    parser.add_argument("--pleias-aggregate", type=Path, required=True)
    parser.add_argument("--cross-decision", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durable-output", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    args = parser.parse_args()
    result = build_manifest(
        args.source_manifest,
        args.book_aggregate,
        args.pleias_aggregate,
        args.cross_decision,
        args.ledger,
        args.output,
        args.durable_output,
        runtime_commit=args.runtime_commit,
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
