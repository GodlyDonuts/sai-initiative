"""Build a complete, filtered non-training census of Common Pile PEPs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.attribution_manifest import build_manifest as build_attribution_manifest
from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.bounded_near_duplicate_filter import build_filter as build_near_duplicate
from sai.data.common_pile_rights_audit import SCHEMA as RIGHTS_SCHEMA
from sai.data.common_pile_streaming_pilot import (
    audit_exclusions,
    download_parent,
    select_bottom_k,
    select_parent,
    write_raw_population,
)
from sai.data.confirmation_promotion import SCHEMA as PROMOTION_SCHEMA
from sai.data.decontamination import build as build_decontaminated
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-pep-full-parent-census-v1"
SOURCE_ID = "common_pile_python_enhancement_proposals"
MAXIMUM_PARENT_BYTES = 64 * 1024 * 1024
MINIMUM_RETAIN_PPM = 900_000
MINIMUM_REPRESENTATION_PPM = 800_000
MAXIMUM_QUARANTINE_PPM = 100_000


class CommonPilePepCensusError(RuntimeError):
    """The PEP evidence, source parent, or full census differs."""


def _load_signed(path: Path, schema: str, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise CommonPilePepCensusError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CommonPilePepCensusError(f"{label} cannot be decoded") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise CommonPilePepCensusError(f"{label} receipt differs")
    return payload


def validate_recovery_evidence(
    aggregate: dict[str, Any],
    screen: dict[str, Any],
    rights: dict[str, Any],
    promotion: dict[str, Any],
    reservoir_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Authorize only a complete filtered census, never training admission."""

    summary = aggregate.get("summary", {})
    verdicts = summary.get("by_source_verdict", {}).get(SOURCE_ID, {})
    triage = summary.get("by_source_conservative_triage", {}).get(SOURCE_ID, {})
    screen_source = screen.get("summary", {}).get("by_source", {}).get(SOURCE_ID, {})
    rights_source = rights.get("summary", {}).get("by_source", {}).get(SOURCE_ID, {})
    promotion_rows = promotion.get("sources")
    promotion_matches = (
        [row for row in promotion_rows if row.get("source_id") == SOURCE_ID]
        if isinstance(promotion_rows, list)
        else []
    )
    parents = [row for row in reservoir_rows if row.get("source_id") == SOURCE_ID]
    rows = sum(verdicts.values()) if isinstance(verdicts, dict) else 0
    retained = verdicts.get("retain", 0) if isinstance(verdicts, dict) else 0
    representation = (
        triage.get("representation_verification", 0)
        if isinstance(triage, dict)
        else 0
    )
    quarantine = triage.get("quarantine", 0) if isinstance(triage, dict) else 0
    retain_ppm = retained * 1_000_000 // rows if rows else 0
    representation_ppm = representation * 1_000_000 // rows if rows else 0
    quarantine_ppm = quarantine * 1_000_000 // rows if rows else 1_000_000
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete"
        or aggregate.get("training_ready") is not False
        or screen.get("schema") != SCREEN_SCHEMA
        or screen.get("status") != "complete"
        or screen.get("benchmark_contamination_screen_complete") is not True
        or rights.get("schema") != RIGHTS_SCHEMA
        or rights.get("status") != "complete_declaration_audit_not_legal_clearance"
        or rights.get("training_ready") is not False
        or promotion.get("schema") != PROMOTION_SCHEMA
        or promotion.get("status") != "complete"
        or len(promotion_matches) != 1
        or promotion_matches[0].get("bounded_streaming_source_pilot_authorized")
        is not False
        or promotion_matches[0].get("failed_checks") != ["zero_quarantine"]
        or rows != 32
        or retain_ppm < MINIMUM_RETAIN_PPM
        or representation_ppm < MINIMUM_REPRESENTATION_PPM
        or quarantine_ppm > MAXIMUM_QUARANTINE_PPM
        or screen_source.get("rows") != rows
        or screen_source.get("contaminated_rows") != 0
        or rights_source.get("rows") != rows
        or rights_source.get("rights_hold_rows") != 0
        or rights_source.get("recognized_declaration_rows") != rows
        or rights_source.get("canonical_license:LicenseRef-Public-Domain") != rows
        or len(parents) != 1
        or parents[0].get("physical_bytes", MAXIMUM_PARENT_BYTES + 1)
        > MAXIMUM_PARENT_BYTES
    ):
        raise CommonPilePepCensusError("PEP recovery evidence differs")
    return {
        "source_id": SOURCE_ID,
        "confirmation_rows": rows,
        "retained_rows": retained,
        "retain_ppm": retain_ppm,
        "representation_verification_rows": representation,
        "representation_verification_ppm": representation_ppm,
        "quarantine_rows": quarantine,
        "quarantine_ppm": quarantine_ppm,
        "benchmark_contaminated_rows": 0,
        "recognized_public_domain_declarations": rows,
        "rights_hold_rows": 0,
        "parent_files": 1,
        "parent_bytes": parents[0]["physical_bytes"],
        "decision_scope": "complete_filtered_nontraining_parent_census_only",
        "row_level_quarantine_required": True,
        "source_wide_quality_admission": False,
        "bulk_training_admission": False,
        "training_ready": False,
    }


def build_census(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    confirmation_root: Path,
    audit_roots: list[Path],
    boundary_roots: list[Path],
    output_root: Path,
    *,
    token: str,
) -> dict[str, Any]:
    """Scan the one exact PEP parent and seal all mechanically surviving rows."""

    if (
        not token
        or output_root.exists()
        or output_root.is_symlink()
        or not boundary_roots
    ):
        raise CommonPilePepCensusError("PEP census output boundary differs")
    paths = {
        "aggregate": confirmation_root / "aggregate.json",
        "screen": confirmation_root / "benchmark_contamination_screen_v2.json",
        "rights": confirmation_root / "rights_declaration_audit.json",
        "promotion": confirmation_root / "promotion_decision_v2.json",
    }
    aggregate = _load_signed(paths["aggregate"], AGGREGATE_SCHEMA, "aggregate")
    screen = _load_signed(paths["screen"], SCREEN_SCHEMA, "screen")
    rights = _load_signed(paths["rights"], RIGHTS_SCHEMA, "rights")
    promotion = _load_signed(paths["promotion"], PROMOTION_SCHEMA, "promotion")
    reservoir_rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    evidence = validate_recovery_evidence(
        aggregate, screen, rights, promotion, reservoir_rows
    )
    excluded_lines, excluded_content, audit_receipts = audit_exclusions(
        audit_roots, SOURCE_ID
    )
    parent = select_parent(
        reservoir_rows, SOURCE_ID, set(excluded_lines)
    )
    if parent["bytes"] != evidence["parent_bytes"]:
        raise CommonPilePepCensusError("PEP census parent differs")

    output_root.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="sai-pep-census-") as temporary:
            compressed = download_parent(parent, token, Path(temporary))
            selected, scan = select_bottom_k(
                compressed,
                parent,
                maximum_rows=100_000,
                excluded_lines=excluded_lines.get(
                    (parent["repository"], parent["path"]), frozenset()
                ),
                excluded_content_sha256s=excluded_content,
            )
            if scan.get("selected_rows") != scan.get("eligible_rows"):
                raise CommonPilePepCensusError("PEP full-parent census is truncated")
            raw_path = output_root / "raw_candidates.jsonl"
            raw = write_raw_population(compressed, parent, selected, raw_path)

        admitted_path = output_root / "benchmark_disjoint_candidates.jsonl"
        decontamination_path = output_root / "decontamination_receipt.json"
        decontamination = build_decontaminated(
            raw_path,
            [],
            admitted_path,
            decontamination_path,
            boundary_indexes=boundary_roots,
            workers=1,
        )
        deduplicated_path = output_root / "near_deduplicated_candidates.jsonl"
        duplicate_receipt_path = output_root / "near_duplicate_receipt.json"
        duplicate = build_near_duplicate(
            admitted_path,
            deduplicated_path,
            duplicate_receipt_path,
        )
        attribution_path = output_root / "attribution_manifest.jsonl"
        attribution_receipt_path = output_root / "attribution_receipt.json"
        attribution = build_attribution_manifest(
            raw_path,
            deduplicated_path,
            attribution_path,
            attribution_receipt_path,
        )
        inputs = {
            label: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "receipt_sha256": payload["receipt_sha256"],
            }
            for label, path, payload in (
                ("aggregate", paths["aggregate"], aggregate),
                ("screen", paths["screen"], screen),
                ("rights", paths["rights"], rights),
                ("promotion", paths["promotion"], promotion),
            )
        }
        payload = {
            "schema": SCHEMA,
            "status": "complete_filtered_nontraining_parent_census",
            "source_id": SOURCE_ID,
            "recovery_evidence": evidence,
            "inputs": inputs,
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_file_sha256": sha256_file(reservoir_receipt_path),
            },
            "parent": parent,
            "audit_populations": audit_receipts,
            "audit_excluded_content_identities": len(excluded_content),
            "scan": scan,
            "raw_population": raw,
            "decontamination": {
                "receipt_path": decontamination_path.name,
                "receipt_file_sha256": sha256_file(decontamination_path),
                "receipt_sha256": decontamination["receipt_sha256"],
                "scanned": decontamination["scanned"],
                "accepted": decontamination["accepted"],
                "dropped": decontamination["dropped"],
                "output_path": admitted_path.name,
                "output_bytes": admitted_path.stat().st_size,
                "output_sha256": sha256_file(admitted_path),
            },
            "near_duplicate_filter": {
                "receipt_path": duplicate_receipt_path.name,
                "receipt_file_sha256": sha256_file(duplicate_receipt_path),
                "receipt_sha256": duplicate["receipt_sha256"],
                "input_documents": duplicate["input"]["documents"],
                "output_documents": duplicate["output"]["documents"],
                "documents_dropped": duplicate["evidence"]["documents_dropped"],
                "duplicate_groups": duplicate["evidence"]["duplicate_groups"],
                "output_path": deduplicated_path.name,
                "output_bytes": deduplicated_path.stat().st_size,
                "output_sha256": sha256_file(deduplicated_path),
            },
            "attribution_manifest": {
                "receipt_path": attribution_receipt_path.name,
                "receipt_file_sha256": sha256_file(attribution_receipt_path),
                "receipt_sha256": attribution["receipt_sha256"],
                "output_path": attribution_path.name,
                "output_bytes": attribution_path.stat().st_size,
                "output_sha256": sha256_file(attribution_path),
                "records": attribution["output"]["records"],
                "obligation_counts": attribution["obligation_counts"],
                "source_text_persisted_in_manifest": False,
            },
            "complete_parent_census": True,
            "parent_removed_after_census": True,
            "maximum_simultaneous_parent_files": 1,
            "rights_verification_complete": False,
            "representation_verification_complete": False,
            "quality_compilation_complete": False,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservoir-receipt", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, action="append", required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    import os

    result = build_census(
        args.manifest,
        args.reservoir_receipt,
        args.confirmation_root,
        args.audit_root,
        args.boundary_index,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
