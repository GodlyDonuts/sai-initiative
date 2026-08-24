"""Build a source-safe, receipt-verified data conversion yield ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.arxiv_abstracts_audit_population import (
    REPOSITORY as ARXIV_REPOSITORY,
)
from sai.data.arxiv_abstracts_audit_population import (
    REVISION as ARXIV_REVISION,
)
from sai.data.arxiv_abstracts_audit_population import (
    SOURCE_ID as ARXIV_SOURCE_ID,
)
from sai.data.arxiv_abstracts_audit_population import (
    SOURCE_ORIGINAL_BYTES as ARXIV_SOURCE_ORIGINAL_BYTES,
)
from sai.data.arxiv_abstracts_audit_population import (
    SOURCE_ROWS as ARXIV_SOURCE_ROWS,
)
from sai.data.arxiv_abstracts_full_census import SCHEMA as ARXIV_CENSUS_SCHEMA
from sai.data.common_pile_streaming_pilot import SCHEMA as COMMON_PILE_PILOT_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-data-conversion-yield-ledger-v5"
MAXIMUM_TRAINING_READY_BYTES = 2_000_000_000_000


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
        "path": f"{path.parent.name}/{path.name}",
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
        "root": root.name,
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
    if receipt.get("schema") != COMMON_PILE_PILOT_SCHEMA:
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
        decontamination_receipt.get("accepted") != decontamination.get("accepted")
        or decontamination_receipt.get("scanned") != decontamination.get("scanned")
        or decontamination_receipt.get("dropped") != decontamination.get("dropped")
        or near_duplicate_receipt.get("output", {}).get("documents")
        != near_duplicate.get("output_documents")
        or attribution_receipt.get("output", {}).get("records")
        != attribution.get("records")
        or attribution.get("records") != near_duplicate.get("output_documents")
    ):
        raise DataYieldLedgerError("pilot nested output coverage differs")
    return {
        "root": root.name,
        "source_id": receipt.get("source_id"),
        "receipt_sha256": receipt["receipt_sha256"],
        "raw_rows": raw.get("rows"),
        "raw_bytes": raw.get("bytes"),
        "benchmark_disjoint_rows": decontamination.get("accepted"),
        "benchmark_disjoint_bytes": decontamination.get("output_bytes"),
        "near_deduplicated_rows": near_duplicate.get("output_documents"),
        "near_deduplicated_bytes": near_duplicate.get("output_bytes"),
        "near_duplicate_documents_dropped": near_duplicate.get("documents_dropped"),
        "attribution_records": attribution.get("records"),
        "obligation_counts": attribution.get("obligation_counts"),
        "rights_declaration_lineage_replay_complete": receipt.get(
            "rights_declaration_lineage_replay_complete"
        ),
        "global_cross_source_near_duplicate_filter_complete": receipt.get(
            "global_cross_source_near_duplicate_filter_complete"
        ),
        "rights_verification_complete": receipt.get("rights_verification_complete"),
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
        "path": f"{path.parent.name}/{path.name}",
        "receipt_sha256": receipt["receipt_sha256"],
        "source_count": len(source_rows),
        "physical_candidate_bytes": sum(route_bytes.values()),
        "route_bytes": dict(sorted(route_bytes.items())),
        "source_wide_rights_clearance_established": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }


def _text_payload_probe(path: Path) -> dict[str, Any]:
    receipt = _load_receipt(path)
    plan_binding = receipt.get("plan")
    measurements = receipt.get("measurements")
    summary = receipt.get("summary")
    if (
        receipt.get("schema") != "sai-reservoir-text-payload-probe-receipt-v1"
        or receipt.get("status") != "complete_bounded_exact_member_measurement"
        or not isinstance(plan_binding, dict)
        or not isinstance(measurements, list)
        or not measurements
        or not isinstance(summary, dict)
        or receipt.get("temporary_members_removed") is not True
        or receipt.get("sample_is_statistical_yield_estimate") is not False
        or receipt.get("full_source_yield_extrapolation_allowed") is not False
        or receipt.get("source_text_persisted") is not False
        or receipt.get("training_ready") is not False
    ):
        raise DataYieldLedgerError("ledger text payload probe differs")
    plan_path_raw = plan_binding.get("path")
    if not isinstance(plan_path_raw, str) or not plan_path_raw:
        raise DataYieldLedgerError("ledger text payload plan path differs")
    plan_path = Path(plan_path_raw)
    plan = _load_receipt(plan_path)
    if (
        plan.get("schema") != "sai-reservoir-text-payload-probe-plan-v1"
        or plan_binding.get("file_sha256") != sha256_file(plan_path)
        or plan_binding.get("receipt_sha256") != plan["receipt_sha256"]
    ):
        raise DataYieldLedgerError("ledger text payload plan binding differs")
    measured = []
    blocked = 0
    for row in measurements:
        if not isinstance(row, dict):
            raise DataYieldLedgerError("ledger text payload measurement differs")
        if row.get("status") == "blocked_selected_member_exceeds_parent_byte_cap":
            blocked += 1
            continue
        measurement = row.get("measurement")
        if (
            row.get("status") != "measured_exact_member"
            or row.get("full_member_size_and_sha256_replayed") is not True
            or not isinstance(row.get("physical_bytes"), int)
            or row["physical_bytes"] <= 0
            or not isinstance(measurement, dict)
            or not isinstance(measurement.get("text_utf8_bytes"), int)
            or measurement["text_utf8_bytes"] < 0
            or not isinstance(measurement.get("useful_text_utf8_bytes"), int)
            or not 0
            <= measurement["useful_text_utf8_bytes"]
            <= measurement["text_utf8_bytes"]
        ):
            raise DataYieldLedgerError("ledger text payload measurement differs")
        measured.append(row)
    measured_physical = sum(row["physical_bytes"] for row in measured)
    measured_text = sum(row["measurement"]["text_utf8_bytes"] for row in measured)
    measured_useful = sum(
        row["measurement"]["useful_text_utf8_bytes"] for row in measured
    )
    if (
        summary.get("selected_members") != len(measurements)
        or summary.get("measured_members") != len(measured)
        or summary.get("blocked_members") != blocked
        or summary.get("measured_physical_bytes") != measured_physical
        or summary.get("measured_text_utf8_bytes") != measured_text
        or summary.get("measured_useful_text_utf8_bytes") != measured_useful
    ):
        raise DataYieldLedgerError("ledger text payload summary differs")
    return {
        "path": f"{path.parent.name}/{path.name}",
        "receipt_sha256": receipt["receipt_sha256"],
        "plan_receipt_sha256": plan["receipt_sha256"],
        "selected_members": len(measurements),
        "measured_members": len(measured),
        "blocked_members": blocked,
        "measured_physical_bytes": measured_physical,
        "measured_text_utf8_bytes": measured_text,
        "measured_useful_text_utf8_bytes": measured_useful,
        "statistical_yield_estimate": False,
        "full_source_yield_extrapolation_allowed": False,
        "source_text_persisted": False,
        "training_ready": False,
    }


def _full_source_census(path: Path) -> dict[str, Any]:
    receipt = _load_receipt(path)
    snapshot = receipt.get("source_snapshot")
    totals = receipt.get("totals")
    if (
        receipt.get("schema") != ARXIV_CENSUS_SCHEMA
        or receipt.get("status") != "complete_text_free_full_parent_census"
        or receipt.get("source_id") != ARXIV_SOURCE_ID
        or not isinstance(snapshot, dict)
        or snapshot.get("repository") != ARXIV_REPOSITORY
        or snapshot.get("revision") != ARXIV_REVISION
        or snapshot.get("compressed_bytes") != ARXIV_SOURCE_ORIGINAL_BYTES
        or snapshot.get("rows") != ARXIV_SOURCE_ROWS
        or not isinstance(totals, dict)
        or totals.get("scanned_rows") != ARXIV_SOURCE_ROWS
        or totals.get("provenance_valid_rows") != ARXIV_SOURCE_ROWS
        or totals.get("invalid_provenance_rows") != 0
        or totals.get("non_monotonic_provenance_rows") != 0
        or not isinstance(totals.get("text_bytes"), int)
        or totals["text_bytes"] <= 0
        or not isinstance(totals.get("mechanically_eligible_unique_rows"), int)
        or totals["mechanically_eligible_unique_rows"] <= 0
        or not isinstance(totals.get("mechanically_eligible_unique_text_bytes"), int)
        or not 0
        < totals["mechanically_eligible_unique_text_bytes"]
        <= totals["text_bytes"]
        or totals.get("audit_position_excluded_identities")
        != receipt.get("audit_excluded_positions")
        or receipt.get("complete_parent_census") is not True
        or receipt.get("parents_removed_after_census") is not True
        or receipt.get("source_text_persisted") is not False
        or receipt.get("benchmark_contamination_screen_complete") is not False
        or receipt.get("near_duplicate_filter_complete") is not False
        or receipt.get("hermes_judgments_complete") is not False
        or receipt.get("quality_compilation_complete") is not False
        or receipt.get("full_source_ingestion_authorized") is not False
        or receipt.get("training_ready") is not False
    ):
        raise DataYieldLedgerError("ledger full source census differs")
    return {
        "path": path.name,
        "source_id": ARXIV_SOURCE_ID,
        "receipt_sha256": receipt["receipt_sha256"],
        "source_rows": ARXIV_SOURCE_ROWS,
        "compressed_parent_bytes": ARXIV_SOURCE_ORIGINAL_BYTES,
        "text_utf8_bytes": totals["text_bytes"],
        "mechanically_eligible_unique_rows": totals[
            "mechanically_eligible_unique_rows"
        ],
        "mechanically_eligible_unique_text_bytes": totals[
            "mechanically_eligible_unique_text_bytes"
        ],
        "benchmark_contamination_screen_complete": False,
        "near_duplicate_filter_complete": False,
        "hermes_judgments_complete": False,
        "training_ready": False,
    }


def build_ledger(
    reservoir_receipts: list[Path],
    audit_roots: list[Path],
    pilot_roots: list[Path],
    output_path: Path,
    *,
    rights_inventory_path: Path | None = None,
    text_payload_probe_paths: list[Path] | None = None,
    full_source_census_paths: list[Path] | None = None,
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
    text_payload_probes = [
        _text_payload_probe(path) for path in (text_payload_probe_paths or [])
    ]
    if len({row["receipt_sha256"] for row in text_payload_probes}) != len(
        text_payload_probes
    ):
        raise DataYieldLedgerError("ledger repeats a text payload probe")
    full_source_censuses = [
        _full_source_census(path) for path in (full_source_census_paths or [])
    ]
    if len({row["receipt_sha256"] for row in full_source_censuses}) != len(
        full_source_censuses
    ):
        raise DataYieldLedgerError("ledger repeats a full source census")
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
        "bounded_text_payload_probes": {
            "probes": text_payload_probes,
            "probe_count": len(text_payload_probes),
            "measured_members": sum(
                row["measured_members"] for row in text_payload_probes
            ),
            "measured_physical_bytes": sum(
                row["measured_physical_bytes"] for row in text_payload_probes
            ),
            "measured_text_utf8_bytes": sum(
                row["measured_text_utf8_bytes"] for row in text_payload_probes
            ),
            "measured_useful_text_utf8_bytes": sum(
                row["measured_useful_text_utf8_bytes"] for row in text_payload_probes
            ),
            "full_reservoir_text_payload_bytes_measured": False,
            "training_ready": False,
        },
        "complete_source_censuses": {
            "censuses": full_source_censuses,
            "source_count": len(full_source_censuses),
            "source_rows_sum": sum(row["source_rows"] for row in full_source_censuses),
            "compressed_parent_bytes_sum": sum(
                row["compressed_parent_bytes"] for row in full_source_censuses
            ),
            "text_utf8_bytes_sum": sum(
                row["text_utf8_bytes"] for row in full_source_censuses
            ),
            "mechanically_eligible_unique_rows_sum": sum(
                row["mechanically_eligible_unique_rows"] for row in full_source_censuses
            ),
            "mechanically_eligible_unique_text_bytes_sum": sum(
                row["mechanically_eligible_unique_text_bytes"]
                for row in full_source_censuses
            ),
            "all_required_training_gates_complete": False,
            "training_ready": False,
        },
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
            "maximum_bytes": MAXIMUM_TRAINING_READY_BYTES,
            "maximum_tb_decimal": 2.0,
            "maximum_tib": MAXIMUM_TRAINING_READY_BYTES / 1024**4,
            "remaining_capacity_bytes": (
                MAXIMUM_TRAINING_READY_BYTES - training_ready_bytes
            ),
            "capacity_exhaustion_required": False,
            "complete": False,
        },
        "claims": {
            "raw_reservoir_bytes_are_not_training_ready_bytes": True,
            "audit_rows_are_not_training_ready_rows": True,
            "bounded_pilot_rows_are_not_training_ready_rows": True,
            "mechanically_eligible_census_bytes_are_not_training_ready_bytes": True,
            "source_text_persisted_in_ledger": False,
            "absolute_local_paths_persisted": False,
            "final_corpus_may_be_smaller_than_maximum": True,
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
    parser.add_argument("--text-payload-probe", type=Path, action="append", default=[])
    parser.add_argument("--full-source-census", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_ledger(
        args.reservoir_receipt,
        args.audit_root,
        args.pilot_root,
        args.output,
        rights_inventory_path=args.rights_inventory,
        text_payload_probe_paths=args.text_payload_probe,
        full_source_census_paths=args.full_source_census,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
