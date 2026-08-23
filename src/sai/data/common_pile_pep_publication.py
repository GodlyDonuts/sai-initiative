"""Build a source-safe publication receipt for the complete PEP census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.common_pile_pep_census import SCHEMA as CENSUS_SCHEMA
from sai.data.common_pile_pep_census import SOURCE_ID
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-pep-census-publication-v1"
REPOSITORY = "common-pile/python_enhancement_proposals_filtered"
REVISION = "582170907dd303c207770fceacd38e6abf133edc"
PARENT_PATH = "peps-dolma-0000.json.gz"
PARENT_BYTES = 3_723_467
PARENT_SHA256 = "4bb61eded5168ac7f0059a92ed242577c67e4fced8c0d019c84bfaca5596c791"
EXPECTED_SCANNED_ROWS = 655
EXPECTED_AUDIT_EXCLUDED_ROWS = 36
EXPECTED_SHORT_ROWS = 1
EXPECTED_ELIGIBLE_ROWS = 618
EXPECTED_BENCHMARK_DISJOINT_ROWS = 568
EXPECTED_FINAL_ROWS = 567


class CommonPilePepPublicationError(RuntimeError):
    """The PEP census evidence or publication boundary differs."""


def _load_signed(path: Path, schema: str, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise CommonPilePepPublicationError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CommonPilePepPublicationError(f"{label} cannot be decoded") from error
    if not isinstance(payload, dict):
        raise CommonPilePepPublicationError(f"{label} differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise CommonPilePepPublicationError(f"{label} receipt differs")
    return payload


def _bound_artifact(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    name = descriptor.get("path") or descriptor.get("output_path")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise CommonPilePepPublicationError(f"{label} path differs")
    path = root / name
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size
        != descriptor.get("bytes", descriptor.get("output_bytes"))
        or sha256_file(path)
        != descriptor.get("sha256", descriptor.get("output_sha256"))
    ):
        raise CommonPilePepPublicationError(f"{label} custody differs")
    return path


def summarize_publication(
    census: dict[str, Any], population: dict[str, Any]
) -> dict[str, Any]:
    """Validate the exact census funnel and return its source-free summary."""

    parent = census.get("parent", {})
    scan = census.get("scan", {})
    decontamination = census.get("decontamination", {})
    duplicate = census.get("near_duplicate_filter", {})
    attribution = census.get("attribution_manifest", {})
    compiler_population = population.get("population", {})
    compiler_lineage = population.get("lineage", {})
    recovery = census.get("recovery_evidence", {})
    if (
        census.get("schema") != CENSUS_SCHEMA
        or census.get("status") != "complete_filtered_nontraining_parent_census"
        or census.get("source_id") != SOURCE_ID
        or census.get("complete_parent_census") is not True
        or census.get("parent_removed_after_census") is not True
        or census.get("maximum_simultaneous_parent_files") != 1
        or census.get("training_ready") is not False
        or census.get("quality_compilation_complete") is not False
        or census.get("representation_verification_complete") is not False
        or population.get("schema") != POPULATION_SCHEMA
        or population.get("status") != "complete"
        or population.get("source_census", {}).get("receipt_sha256")
        != census.get("receipt_sha256")
        or population.get("complete_census_survivor_coverage") is not True
        or population.get("benchmark_contamination_screen_complete") is not True
        or population.get("bounded_near_duplicate_filter_complete") is not True
        or population.get("exact_attribution_coverage") is not True
        or population.get("hermes_judgments_complete") is not False
        or population.get("quality_compilation_complete") is not False
        or population.get("representation_verification_complete") is not False
        or population.get("training_ready") is not False
        or parent.get("repository") != REPOSITORY
        or parent.get("revision") != REVISION
        or parent.get("path") != PARENT_PATH
        or parent.get("bytes") != PARENT_BYTES
        or parent.get("sha256") != PARENT_SHA256
        or scan.get("scanned_rows") != EXPECTED_SCANNED_ROWS
        or scan.get("audit_excluded_rows") != EXPECTED_AUDIT_EXCLUDED_ROWS
        or scan.get("short_rows") != EXPECTED_SHORT_ROWS
        or scan.get("eligible_rows") != EXPECTED_ELIGIBLE_ROWS
        or scan.get("selected_rows") != EXPECTED_ELIGIBLE_ROWS
        or decontamination.get("scanned") != EXPECTED_ELIGIBLE_ROWS
        or decontamination.get("accepted") != EXPECTED_BENCHMARK_DISJOINT_ROWS
        or decontamination.get("dropped")
        != EXPECTED_ELIGIBLE_ROWS - EXPECTED_BENCHMARK_DISJOINT_ROWS
        or duplicate.get("input_documents") != EXPECTED_BENCHMARK_DISJOINT_ROWS
        or duplicate.get("output_documents") != EXPECTED_FINAL_ROWS
        or duplicate.get("documents_dropped") != 1
        or duplicate.get("duplicate_groups") != 1
        or attribution.get("records") != EXPECTED_FINAL_ROWS
        or attribution.get("source_text_persisted_in_manifest") is not False
        or compiler_population.get("rows") != EXPECTED_FINAL_ROWS
        or compiler_lineage.get("rows") != EXPECTED_FINAL_ROWS
        or population.get("by_source") != {SOURCE_ID: EXPECTED_FINAL_ROWS}
        or recovery.get("bulk_training_admission") is not False
        or recovery.get("source_wide_quality_admission") is not False
        or recovery.get("training_ready") is not False
    ):
        raise CommonPilePepPublicationError("PEP publication evidence differs")
    return {
        "source_snapshot": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "path": PARENT_PATH,
            "compressed_bytes": PARENT_BYTES,
            "compressed_sha256": PARENT_SHA256,
        },
        "funnel": {
            "scanned_rows": EXPECTED_SCANNED_ROWS,
            "audit_excluded_rows": EXPECTED_AUDIT_EXCLUDED_ROWS,
            "short_rows": EXPECTED_SHORT_ROWS,
            "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
            "benchmark_overlap_drops": (
                EXPECTED_ELIGIBLE_ROWS - EXPECTED_BENCHMARK_DISJOINT_ROWS
            ),
            "benchmark_disjoint_rows": EXPECTED_BENCHMARK_DISJOINT_ROWS,
            "near_duplicate_drops": 1,
            "final_unique_rows": EXPECTED_FINAL_ROWS,
            "attribution_records": EXPECTED_FINAL_ROWS,
            "compiler_population_rows": EXPECTED_FINAL_ROWS,
        },
        "census_receipt_sha256": census["receipt_sha256"],
        "compiler_population_receipt_sha256": population["receipt_sha256"],
        "recovery_evidence": recovery,
    }


def build_publication(
    census_root: Path, population_root: Path, output_path: Path
) -> dict[str, Any]:
    """Create a tamper-evident PEP envelope without publishing source text."""

    if output_path.exists() or output_path.is_symlink():
        raise CommonPilePepPublicationError("publication output already exists")
    census_path = census_root / "receipt.json"
    population_path = population_root / "receipt.json"
    census = _load_signed(census_path, CENSUS_SCHEMA, "census")
    population = _load_signed(population_path, POPULATION_SCHEMA, "population")
    summary = summarize_publication(census, population)

    census_artifacts = {
        "raw_population": _bound_artifact(
            census_root, census["raw_population"], "raw population"
        ),
        "benchmark_disjoint_population": _bound_artifact(
            census_root, census["decontamination"], "benchmark-disjoint population"
        ),
        "near_deduplicated_population": _bound_artifact(
            census_root, census["near_duplicate_filter"], "near-deduplicated population"
        ),
        "attribution_manifest": _bound_artifact(
            census_root, census["attribution_manifest"], "attribution manifest"
        ),
    }
    population_artifacts = {
        "compiler_candidates": _bound_artifact(
            population_root, population["population"], "compiler candidates"
        ),
        "compiler_lineage": _bound_artifact(
            population_root, population["lineage"], "compiler lineage"
        ),
    }
    nested_receipt_hashes: dict[str, str] = {}
    for label, descriptor in (
        ("decontamination", census["decontamination"]),
        ("near_duplicate", census["near_duplicate_filter"]),
        ("attribution", census["attribution_manifest"]),
    ):
        name = descriptor.get("receipt_path")
        if not isinstance(name, str) or Path(name).name != name:
            raise CommonPilePepPublicationError(f"{label} receipt path differs")
        path = census_root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or sha256_file(path) != descriptor.get("receipt_file_sha256")
        ):
            raise CommonPilePepPublicationError(f"{label} receipt custody differs")
        nested_receipt_hashes[f"{label}_receipt_file_sha256"] = sha256_file(path)

    payload = {
        "schema": SCHEMA,
        "status": "complete_pre_hermes_source_safe_evidence",
        **summary,
        "artifact_file_sha256": {
            "census_receipt": sha256_file(census_path),
            "compiler_population_receipt": sha256_file(population_path),
            **nested_receipt_hashes,
            **{label: sha256_file(path) for label, path in census_artifacts.items()},
            **{
                label: sha256_file(path)
                for label, path in population_artifacts.items()
            },
        },
        "source_text_published": False,
        "individual_decontamination_decisions_published": False,
        "absolute_local_paths_persisted": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "representation_verification_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_publication(args.census_root, args.population_root, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
