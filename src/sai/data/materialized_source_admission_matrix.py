"""Join physical source custody to fail-closed admission work routes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.hf_materialized_source_lake import RECEIPT_SCHEMA as LAKE_SCHEMA
from sai.data.reservoir_rights_inventory import SCHEMA as RIGHTS_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-materialized-source-admission-matrix-v1"
ADMISSION_GATES = (
    "source_wide_rights_admission_complete",
    "language_identification_complete",
    "english_routing_and_translation_complete",
    "full_benchmark_decontamination_complete",
    "global_exact_deduplication_complete",
    "global_semantic_deduplication_complete",
    "hermes_full_population_quality_compilation_complete",
    "accepted_representation_verification_complete",
    "prerequisite_graph_assignment_complete",
    "spiral_curriculum_assignment_complete",
)


class MaterializedSourceAdmissionMatrixError(RuntimeError):
    """Custody, rights, or admission accounting differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise MaterializedSourceAdmissionMatrixError("matrix input is unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializedSourceAdmissionMatrixError(
            "matrix input is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise MaterializedSourceAdmissionMatrixError("matrix input differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise MaterializedSourceAdmissionMatrixError("matrix receipt differs")
    return payload


def build_matrix_payload(
    lake: dict[str, Any], rights: dict[str, Any]
) -> dict[str, Any]:
    """Return one exact admission row for every materialized component."""

    lake_components = lake.get("components")
    rights_rows = rights.get("source_rows")
    if (
        lake.get("schema") != LAKE_SCHEMA
        or lake.get("target_met") is not True
        or lake.get("training_ready") is not False
        or lake.get("all_destination_lfs_identities_replayed_against_pinned_upstream")
        is not True
        or not isinstance(lake_components, list)
        or not lake_components
        or rights.get("schema") != RIGHTS_SCHEMA
        or rights.get("training_ready") is not False
        or not isinstance(rights_rows, list)
        or not rights_rows
    ):
        raise MaterializedSourceAdmissionMatrixError("matrix evidence differs")
    rights_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rights_rows:
        if not isinstance(row, dict):
            raise MaterializedSourceAdmissionMatrixError("rights row differs")
        identity = (row.get("repository"), row.get("revision"))
        if (
            not all(isinstance(value, str) and value for value in identity)
            or row.get("source_wide_rights_clearance_established") is not False
            or row.get("legal_clearance_established") is not False
            or row.get("training_ready") is not False
        ):
            raise MaterializedSourceAdmissionMatrixError("rights identity differs")
        rights_by_identity[identity].append(row)
    matrix_rows = []
    seen = set()
    route_counts: Counter[str] = Counter()
    route_bytes: Counter[str] = Counter()
    for component in lake_components:
        if not isinstance(component, dict):
            raise MaterializedSourceAdmissionMatrixError("lake component differs")
        identity = (
            component.get("source_repository"),
            component.get("source_revision"),
        )
        matching_rights = rights_by_identity.get(identity, [])
        if len(matching_rights) > 1:
            matching_rights = [
                row
                for row in matching_rights
                if row.get("source_id") == component.get("source_id")
            ]
        rights_row = matching_rights[0] if len(matching_rights) == 1 else None
        files = component.get("materialized_files")
        size = component.get("materialized_bytes")
        if (
            rights_row is None
            or identity in seen
            or isinstance(files, bool)
            or not isinstance(files, int)
            or files <= 0
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or rights_row.get("files", 0) < files
            or rights_row.get("bytes", 0) < size
            or component.get("training_ready") is not False
        ):
            raise MaterializedSourceAdmissionMatrixError(
                "lake-to-rights accounting differs"
            )
        seen.add(identity)
        route = rights_row.get("rights_work_route")
        if not isinstance(route, str) or not route:
            raise MaterializedSourceAdmissionMatrixError("rights route differs")
        route_counts[route] += 1
        route_bytes[route] += size
        gates = {gate: False for gate in ADMISSION_GATES}
        matrix_row = {
            "source_id": component.get("source_id"),
            "rights_inventory_source_id": rights_row.get("source_id"),
            "repository": identity[0],
            "revision": identity[1],
            "materialized_files": files,
            "materialized_bytes": size,
            "materialized_identity_verified": True,
            "complete_selected_source_snapshot": component.get(
                "complete_source_snapshot"
            ),
            "rights_work_route": route,
            "declared_license": rights_row.get("declared_license"),
            "card_license_declarations": rights_row.get("card_license_declarations"),
            "source_manifest_path": component.get("source_manifest_path"),
            "admission_gates": gates,
            "blocking_gates": list(ADMISSION_GATES),
            "source_text_persisted": False,
            "training_ready": False,
        }
        matrix_row["row_sha256"] = canonical_sha256(matrix_row)
        matrix_rows.append(matrix_row)
    matrix_rows.sort(key=lambda row: (row["repository"], row["source_id"]))
    total_files = sum(row["materialized_files"] for row in matrix_rows)
    total_bytes = sum(row["materialized_bytes"] for row in matrix_rows)
    if (
        total_files != lake.get("materialized_files")
        or total_bytes != lake.get("materialized_bytes")
        or lake.get("by_source") is None
    ):
        raise MaterializedSourceAdmissionMatrixError("matrix coverage differs")
    return {
        "schema": SCHEMA,
        "status": "complete_fail_closed_materialized_source_admission_matrix",
        "source_lake": {
            "destination_repository": lake.get("destination_repository"),
            "destination_revision": lake.get("destination_revision"),
            "receipt_sha256": lake.get("receipt_sha256"),
            "materialized_files": total_files,
            "materialized_bytes": total_bytes,
        },
        "rights_inventory": {
            "receipt_sha256": rights.get("receipt_sha256"),
            "license_policy_sha256": rights.get("license_policy_sha256"),
        },
        "rows": matrix_rows,
        "rows_sha256": canonical_sha256(matrix_rows),
        "summary": {
            "sources": len(matrix_rows),
            "materialized_files": total_files,
            "materialized_bytes": total_bytes,
            "rights_work_routes": dict(sorted(route_counts.items())),
            "rights_work_route_bytes": dict(sorted(route_bytes.items())),
            "sources_with_all_admission_gates_complete": 0,
            "training_ready_sources": 0,
        },
        "admission_gate_order": list(ADMISSION_GATES),
        "physical_custody_is_training_admission": False,
        "source_text_persisted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }


def build_matrix(
    lake_receipt_path: Path, rights_receipt_path: Path, output_path: Path
) -> dict[str, Any]:
    """Load exact signed receipts and freeze the joined admission matrix."""

    if output_path.exists() or output_path.is_symlink():
        raise MaterializedSourceAdmissionMatrixError("matrix output exists")
    lake = _load_signed(lake_receipt_path, LAKE_SCHEMA)
    rights = _load_signed(rights_receipt_path, RIGHTS_SCHEMA)
    payload = build_matrix_payload(lake, rights)
    payload["inputs"] = {
        "source_lake_receipt_file_sha256": sha256_file(lake_receipt_path),
        "rights_inventory_receipt_file_sha256": sha256_file(rights_receipt_path),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lake-receipt", type=Path, required=True)
    parser.add_argument("--rights-inventory-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_matrix(
        args.source_lake_receipt,
        args.rights_inventory_receipt,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
