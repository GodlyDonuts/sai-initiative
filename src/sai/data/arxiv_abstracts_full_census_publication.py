"""Build a source-safe publication receipt for the full arXiv census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.arxiv_abstracts_audit_population import (
    REPOSITORY,
    REVISION,
    SOURCE_ID,
    SOURCE_MEMORY_BYTES,
    SOURCE_ORIGINAL_BYTES,
    SOURCE_ROWS,
)
from sai.data.arxiv_abstracts_full_census import (
    EXPECTED_AUDIT_ROWS,
    EXPECTED_PARENTS,
)
from sai.data.arxiv_abstracts_full_census import (
    SCHEMA as CENSUS_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-arxiv-abstracts-full-census-publication-v1"
EXPECTED_TEXT_BYTES = 2_388_470_891
EXPECTED_PROVENANCE_GAP_POSITIONS = 1_654
EXPECTED_PROVENANCE_PHYSICAL_LINE_DELTA_ROWS = 1_062_885
EXPECTED_SHORT_ROWS = 45_463
EXPECTED_ELIGIBLE_ROWS = 2_458_156
EXPECTED_ELIGIBLE_TEXT_BYTES = 2_380_856_330
EXPECTED_DUPLICATE_NATIVE_ID_ROWS = 1


class ArxivAbstractsFullCensusPublicationError(RuntimeError):
    """The census evidence or source-safe publication boundary differs."""


def _load_census(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise ArxivAbstractsFullCensusPublicationError(
            "full census receipt is missing or unsafe"
        )
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ArxivAbstractsFullCensusPublicationError(
            "full census receipt cannot be decoded"
        ) from error
    if not isinstance(payload, dict):
        raise ArxivAbstractsFullCensusPublicationError("full census receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != CENSUS_SCHEMA or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise ArxivAbstractsFullCensusPublicationError("full census receipt differs")
    return payload


def summarize_publication(census: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact text-free census and return its public summary."""

    snapshot = census.get("source_snapshot", {})
    authorization = census.get("authorization", {})
    totals = census.get("totals", {})
    parents = census.get("parents")
    exclusions = census.get("audit_exclusions")
    expected_parent_paths = set(EXPECTED_PARENTS)
    if (
        census.get("schema") != CENSUS_SCHEMA
        or census.get("status") != "complete_text_free_full_parent_census"
        or census.get("source_id") != SOURCE_ID
        or snapshot.get("repository") != REPOSITORY
        or snapshot.get("revision") != REVISION
        or snapshot.get("parents") != len(EXPECTED_PARENTS)
        or snapshot.get("compressed_bytes") != SOURCE_ORIGINAL_BYTES
        or snapshot.get("reported_memory_bytes") != SOURCE_MEMORY_BYTES
        or snapshot.get("rows") != SOURCE_ROWS
        or authorization.get("decision_scope") != "text_free_full_parent_census_only"
        or authorization.get("bulk_ingestion_authorized") is not False
        or authorization.get("training_ready") is not False
        or not isinstance(parents, list)
        or len(parents) != len(EXPECTED_PARENTS)
        or {parent.get("path") for parent in parents} != expected_parent_paths
        or not isinstance(exclusions, list)
        or [row.get("matched_source_rows") for row in exclusions] != [4, 32, 1_024]
        or census.get("audit_excluded_positions") != EXPECTED_AUDIT_ROWS
        or census.get("audit_excluded_content_identities") != EXPECTED_AUDIT_ROWS
        or totals.get("scanned_rows") != SOURCE_ROWS
        or totals.get("text_rows") != SOURCE_ROWS
        or totals.get("text_bytes") != EXPECTED_TEXT_BYTES
        or totals.get("provenance_valid_rows") != SOURCE_ROWS
        or totals.get("invalid_provenance_rows") != 0
        or totals.get("non_monotonic_provenance_rows") != 0
        or totals.get("source_provenance_gap_positions")
        != EXPECTED_PROVENANCE_GAP_POSITIONS
        or totals.get("provenance_physical_line_delta_rows")
        != EXPECTED_PROVENANCE_PHYSICAL_LINE_DELTA_ROWS
        or totals.get("audit_excluded_rows") != EXPECTED_AUDIT_ROWS
        or totals.get("audit_position_excluded_rows") != EXPECTED_AUDIT_ROWS
        or totals.get("audit_position_excluded_identities") != EXPECTED_AUDIT_ROWS
        or totals.get("audit_content_excluded_rows") != 0
        or totals.get("non_cc0_declaration_rows") != 0
        or totals.get("short_rows") != EXPECTED_SHORT_ROWS
        or totals.get("oversized_rows") != 0
        or totals.get("exact_duplicate_rows") != 0
        or totals.get("duplicate_native_id_rows") != EXPECTED_DUPLICATE_NATIVE_ID_ROWS
        or totals.get("mechanically_eligible_unique_rows") != EXPECTED_ELIGIBLE_ROWS
        or totals.get("mechanically_eligible_unique_text_bytes")
        != EXPECTED_ELIGIBLE_TEXT_BYTES
        or EXPECTED_ELIGIBLE_ROWS + EXPECTED_SHORT_ROWS + EXPECTED_AUDIT_ROWS
        != SOURCE_ROWS
        or census.get("complete_parent_census") is not True
        or census.get("maximum_simultaneous_parent_files") != 1
        or census.get("parents_removed_after_census") is not True
        or census.get("source_text_persisted") is not False
        or census.get("benchmark_contamination_screen_complete") is not False
        or census.get("near_duplicate_filter_complete") is not False
        or census.get("hermes_judgments_complete") is not False
        or census.get("quality_compilation_complete") is not False
        or census.get("full_source_ingestion_authorized") is not False
        or census.get("training_ready") is not False
        or census.get("four_b_training_authorized") is not False
    ):
        raise ArxivAbstractsFullCensusPublicationError(
            "full census publication evidence differs"
        )
    for parent in parents:
        expected = EXPECTED_PARENTS[parent["path"]]
        if (
            parent.get("repository") != REPOSITORY
            or parent.get("revision") != REVISION
            or parent.get("compressed_bytes") != expected["bytes"]
            or parent.get("compressed_sha256") != expected["sha256"]
            or parent.get("source_text_persisted") is not False
        ):
            raise ArxivAbstractsFullCensusPublicationError(
                "full census parent evidence differs"
            )
    return {
        "source_snapshot": snapshot,
        "funnel": {
            "scanned_rows": SOURCE_ROWS,
            "audit_excluded_rows": EXPECTED_AUDIT_ROWS,
            "short_rows": EXPECTED_SHORT_ROWS,
            "mechanically_eligible_unique_rows": EXPECTED_ELIGIBLE_ROWS,
            "mechanically_eligible_unique_text_bytes": EXPECTED_ELIGIBLE_TEXT_BYTES,
        },
        "provenance": {
            "valid_rows": SOURCE_ROWS,
            "gap_positions": EXPECTED_PROVENANCE_GAP_POSITIONS,
            "physical_line_delta_rows": EXPECTED_PROVENANCE_PHYSICAL_LINE_DELTA_ROWS,
            "invalid_rows": 0,
            "non_monotonic_rows": 0,
        },
        "audit_excluded_locator_identities": EXPECTED_AUDIT_ROWS,
        "audit_excluded_full_text_identities": EXPECTED_AUDIT_ROWS,
        "census_receipt_sha256": census["receipt_sha256"],
    }


def build_publication(census_path: Path, output_path: Path) -> dict[str, Any]:
    """Seal the census without publishing source rows or machine-local paths."""

    if output_path.exists() or output_path.is_symlink():
        raise ArxivAbstractsFullCensusPublicationError(
            "full census publication output exists"
        )
    census = _load_census(census_path)
    payload = {
        "schema": SCHEMA,
        "status": "complete_source_safe_full_parent_census_evidence",
        **summarize_publication(census),
        "artifact_file_sha256": {"census_receipt": sha256_file(census_path)},
        "source_text_published": False,
        "individual_source_rows_published": False,
        "absolute_local_paths_persisted": False,
        "benchmark_contamination_screen_complete": False,
        "near_duplicate_filter_complete": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "full_source_ingestion_authorized": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_publication(args.census, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
