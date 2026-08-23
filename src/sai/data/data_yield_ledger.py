"""Build a source-safe, receipt-verified data conversion yield ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-data-conversion-yield-ledger-v2"
TARGET_TRAINING_READY_BYTES = 8 * 1024**4


class DataYieldLedgerError(RuntimeError):
    """An input receipt, bound artifact, or accounting claim differs."""


def _load_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise DataYieldLedgerError(f"ledger receipt is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DataYieldLedgerError(f"ledger receipt is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise DataYieldLedgerError(f"ledger receipt is not an object: {path}")
    receipt_sha256 = payload.get("receipt_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if receipt_sha256 != canonical_sha256(unsigned):
        raise DataYieldLedgerError(f"ledger receipt hash differs: {path}")
    return payload


def _member_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DataYieldLedgerError("ledger bound-file path differs")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise DataYieldLedgerError("ledger bound-file path escapes its root") from error
    return candidate


def _bound_file(root: Path, descriptor: dict[str, Any]) -> Path:
    candidate = _member_path(root, descriptor.get("path"))
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.stat().st_nlink != 1
        or descriptor.get("bytes") != candidate.stat().st_size
        or descriptor.get("sha256") != sha256_file(candidate)
    ):
        raise DataYieldLedgerError(f"ledger bound file differs: {candidate}")
    return candidate


def _bound_nested_receipt(root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    candidate = _member_path(root, descriptor.get("receipt_path"))
    receipt = _load_receipt(candidate)
    if (
        descriptor.get("receipt_file_sha256") != sha256_file(candidate)
        or descriptor.get("receipt_sha256") != receipt["receipt_sha256"]
    ):
        raise DataYieldLedgerError(
            f"ledger nested receipt binding differs: {candidate}"
        )
    return receipt


def _reservoir_row(path: Path) -> dict[str, Any]:
    receipt = _load_receipt(path)
    schema = receipt.get("schema")
    if schema == "sai-source-reservoir-receipt-v1":
        referenced_bytes = receipt.get("selected_bytes")
    elif schema == "sai-frontier-source-reservoir-receipt-v1":
        referenced_bytes = receipt.get("selected_physical_bytes")
    else:
        raise DataYieldLedgerError(f"unsupported reservoir receipt: {path}")
    if not isinstance(referenced_bytes, int) or referenced_bytes <= 0:
        raise DataYieldLedgerError(f"reservoir byte accounting differs: {path}")
    manifest = receipt.get("manifest")
    if not isinstance(manifest, dict):
        raise DataYieldLedgerError(f"reservoir manifest binding differs: {path}")
    _bound_file(path.parent, manifest)
    if receipt.get("training_ready") is not False:
        raise DataYieldLedgerError("reservoir receipt makes an unsupported ready claim")
    return {
        "path": str(path.resolve()),
        "schema": schema,
        "receipt_sha256": receipt["receipt_sha256"],
        "manifest_sha256": manifest["sha256"],
        "referenced_candidate_bytes": referenced_bytes,
        "referenced_candidate_files": receipt.get(
            "selected_files", sum(receipt.get("by_source_files", {}).values())
        ),
        "physical_bytes_are_verified_text_payload_bytes": bool(
            receipt.get("text_payload_bytes_measured", False)
        ),
        "training_ready": False,
    }


def _audit_row(root: Path) -> dict[str, Any]:
    receipt_path = root / "receipt.json"
    receipt = _load_receipt(receipt_path)
    if receipt.get("schema") != "sai-reservoir-audit-population-receipt-v1":
        raise DataYieldLedgerError(f"unsupported audit receipt: {receipt_path}")
    population = receipt.get("population")
    lineage = receipt.get("lineage")
    if not isinstance(population, dict) or not isinstance(lineage, dict):
        raise DataYieldLedgerError("audit population binding differs")
    _bound_file(root, population)
    _bound_file(root, lineage)
    if (
        not isinstance(population.get("rows"), int)
        or population["rows"] <= 0
        or not isinstance(population.get("bytes"), int)
        or population["bytes"] <= 0
        or not receipt.get("by_source")
    ):
        raise DataYieldLedgerError("audit population accounting differs")
    if receipt.get("training_ready") is not False:
        raise DataYieldLedgerError("audit receipt makes an unsupported ready claim")
    return {
        "root": str(root.resolve()),
        "receipt_sha256": receipt["receipt_sha256"],
        "population_rows": population.get("rows"),
        "population_bytes": population.get("bytes"),
        "fully_verified_compressed_parent_bytes": receipt.get(
            "fully_verified_compressed_parent_bytes", 0
        ),
        "source_count": len(receipt.get("by_source", {})),
        "identity_disjoint_from_discovery": receipt.get(
            "identity_disjoint_from_discovery"
        ),
        "exact_content_disjoint_from_discovery": receipt.get(
            "exact_content_disjoint_from_discovery"
        ),
        "training_ready": False,
    }


def _pilot_row(root: Path) -> dict[str, Any]:
    receipt_path = root / "receipt.json"
    receipt = _load_receipt(receipt_path)
    if receipt.get("schema") != "sai-common-pile-streaming-pilot-v1":
        raise DataYieldLedgerError(f"unsupported pilot receipt: {receipt_path}")
    if receipt.get("training_ready") is not False:
        raise DataYieldLedgerError("pilot receipt makes an unsupported ready claim")
    raw = receipt.get("raw_population")
    decontamination = receipt.get("decontamination")
    near_duplicate = receipt.get("near_duplicate_filter")
    attribution = receipt.get("attribution_manifest")
    if not all(
        isinstance(row, dict)
        for row in (raw, decontamination, near_duplicate, attribution)
    ):
        raise DataYieldLedgerError("pilot conversion binding differs")
    _bound_file(root, raw)
    decontamination_output = {
        "path": decontamination.get("output_path"),
        "bytes": decontamination.get("output_bytes"),
        "sha256": decontamination.get("output_sha256"),
    }
    near_duplicate_output = {
        "path": near_duplicate.get("output_path"),
        "bytes": near_duplicate.get("output_bytes"),
        "sha256": near_duplicate.get("output_sha256"),
    }
    _bound_file(root, decontamination_output)
    _bound_file(root, near_duplicate_output)
    _bound_file(
        root,
        {
            "path": attribution.get("output_path"),
            "bytes": attribution.get("output_bytes"),
            "sha256": attribution.get("output_sha256"),
        },
    )
    decontamination_receipt = _bound_nested_receipt(root, decontamination)
    near_duplicate_receipt = _bound_nested_receipt(root, near_duplicate)
    attribution_receipt = _bound_nested_receipt(root, attribution)
    if (
        decontamination_receipt.get("output", {}).get("documents")
        != decontamination.get("output_documents")
        or near_duplicate_receipt.get("output", {}).get("documents")
        != near_duplicate.get("output_documents")
        or attribution_receipt.get("output", {}).get("records")
        != attribution.get("records")
        or attribution.get("records") != near_duplicate.get("output_documents")
    ):
        raise DataYieldLedgerError("pilot nested output coverage differs")
    return {
        "root": str(root.resolve()),
        "source_id": receipt.get("source_id"),
        "receipt_sha256": receipt["receipt_sha256"],
        "raw_rows": raw.get("rows"),
        "raw_bytes": raw.get("bytes"),
        "benchmark_disjoint_rows": decontamination.get("output_documents"),
        "benchmark_disjoint_bytes": decontamination.get("output_bytes"),
        "near_deduplicated_rows": near_duplicate.get("output_documents"),
        "near_deduplicated_bytes": near_duplicate.get("output_bytes"),
        "near_duplicate_documents_dropped": near_duplicate.get(
            "documents_dropped"
        ),
        "attribution_records": attribution.get("records"),
        "obligation_counts": attribution.get("obligation_counts"),
        "rights_declaration_lineage_replay_complete": receipt.get(
            "rights_declaration_lineage_replay_complete"
        ),
        "global_cross_source_near_duplicate_filter_complete": receipt.get(
            "global_cross_source_near_duplicate_filter_complete"
        ),
        "rights_verification_complete": receipt.get(
            "rights_verification_complete"
        ),
        "representation_verification_complete": receipt.get(
            "representation_verification_complete"
        ),
        "training_ready": False,
    }


def _rights_inventory(path: Path) -> dict[str, Any]:
    receipt = _load_receipt(path)
    if receipt.get("schema") != "sai-reservoir-rights-inventory-v2":
        raise DataYieldLedgerError("ledger rights inventory schema differs")
    source_rows = receipt.get("source_rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise DataYieldLedgerError("ledger rights inventory sources differ")
    route_bytes: dict[str, int] = {}
    source_ids = set()
    for row in source_rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("source_id"), str)
            or row["source_id"] in source_ids
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not isinstance(row.get("rights_work_route"), str)
        ):
            raise DataYieldLedgerError("ledger rights source row differs")
        source_ids.add(row["source_id"])
        route = row["rights_work_route"]
        route_bytes[route] = route_bytes.get(route, 0) + row["bytes"]
    summary = receipt.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("sources") != len(source_rows)
        or summary.get("physical_candidate_bytes") != sum(route_bytes.values())
        or receipt.get("training_ready") is not False
    ):
        raise DataYieldLedgerError("ledger rights summary differs")
    return {
        "path": str(path.resolve()),
        "receipt_sha256": receipt["receipt_sha256"],
        "source_count": len(source_rows),
        "physical_candidate_bytes": sum(route_bytes.values()),
        "route_bytes": dict(sorted(route_bytes.items())),
        "source_wide_rights_clearance_established": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }


def build_ledger(
    reservoir_receipts: list[Path],
    audit_roots: list[Path],
    pilot_roots: list[Path],
    output_path: Path,
    *,
    rights_inventory_path: Path | None = None,
) -> dict[str, Any]:
    """Verify all inputs and seal conversion-stage byte/row accounting."""

    if output_path.exists() or output_path.is_symlink() or not reservoir_receipts:
        raise DataYieldLedgerError("ledger output or reservoir inputs differ")
    reservoirs = [_reservoir_row(path) for path in reservoir_receipts]
    if len({row["manifest_sha256"] for row in reservoirs}) != len(reservoirs):
        raise DataYieldLedgerError("ledger repeats a reservoir manifest")
    audits = [_audit_row(root) for root in audit_roots]
    if len({row["receipt_sha256"] for row in audits}) != len(audits):
        raise DataYieldLedgerError("ledger repeats an audit population")
    pilots = [_pilot_row(root) for root in pilot_roots]
    rights = (
        _rights_inventory(rights_inventory_path)
        if rights_inventory_path is not None
        else None
    )
    source_ids = [row["source_id"] for row in pilots]
    if len(source_ids) != len(set(source_ids)):
        raise DataYieldLedgerError("ledger repeats a pilot source")
    referenced_bytes = sum(row["referenced_candidate_bytes"] for row in reservoirs)
    if rights is not None and rights["physical_candidate_bytes"] != referenced_bytes:
        raise DataYieldLedgerError("ledger rights and reservoir bytes differ")
    training_ready_bytes = 0
    payload = {
        "schema": SCHEMA,
        "status": "complete_source_safe_accounting",
        "reservoir_candidates": {
            "receipts": reservoirs,
            "referenced_candidate_bytes_sum": referenced_bytes,
            "referenced_candidate_tib_sum": referenced_bytes / 1024**4,
            "cross_inventory_overlap_resolved": False,
            "verified_text_payload_bytes": False,
            "training_ready": False,
        },
        "audit_populations": {
            "populations": audits,
            "population_rows_sum": sum(row["population_rows"] for row in audits),
            "population_bytes_sum": sum(row["population_bytes"] for row in audits),
            "training_ready": False,
        },
        "rights_routing": rights,
        "bounded_source_pilots": {
            "pilots": pilots,
            "source_count": len(pilots),
            "raw_rows_sum": sum(row["raw_rows"] for row in pilots),
            "raw_bytes_sum": sum(row["raw_bytes"] for row in pilots),
            "benchmark_disjoint_rows_sum": sum(
                row["benchmark_disjoint_rows"] for row in pilots
            ),
            "near_deduplicated_rows_sum": sum(
                row["near_deduplicated_rows"] for row in pilots
            ),
            "near_deduplicated_bytes_sum": sum(
                row["near_deduplicated_bytes"] for row in pilots
            ),
            "global_cross_source_near_duplicate_filter_complete": False,
            "training_ready": False,
        },
        "training_ready": {
            "exact_bytes": training_ready_bytes,
            "exact_tib": 0.0,
            "target_bytes": TARGET_TRAINING_READY_BYTES,
            "target_tib": 8,
            "remaining_bytes": TARGET_TRAINING_READY_BYTES - training_ready_bytes,
            "complete": False,
        },
        "claims": {
            "raw_reservoir_bytes_are_not_training_ready_bytes": True,
            "audit_rows_are_not_training_ready_rows": True,
            "bounded_pilot_rows_are_not_training_ready_rows": True,
            "source_text_persisted_in_ledger": False,
            "four_b_training_authorized": False,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reservoir-receipt", type=Path, action="append", required=True
    )
    parser.add_argument("--audit-root", type=Path, action="append", default=[])
    parser.add_argument("--pilot-root", type=Path, action="append", default=[])
    parser.add_argument("--rights-inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_ledger(
        args.reservoir_receipt,
        args.audit_root,
        args.pilot_root,
        args.output,
        rights_inventory_path=args.rights_inventory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
