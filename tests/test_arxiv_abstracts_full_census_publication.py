import copy

import pytest

from sai.data.arxiv_abstracts_audit_population import (
    REPOSITORY,
    REVISION,
    SOURCE_ID,
    SOURCE_MEMORY_BYTES,
    SOURCE_ORIGINAL_BYTES,
    SOURCE_ROWS,
)
from sai.data.arxiv_abstracts_full_census import (
    EXPECTED_PARENTS,
)
from sai.data.arxiv_abstracts_full_census import (
    SCHEMA as CENSUS_SCHEMA,
)
from sai.data.arxiv_abstracts_full_census_publication import (
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_ELIGIBLE_TEXT_BYTES,
    EXPECTED_PROVENANCE_GAP_POSITIONS,
    EXPECTED_PROVENANCE_PHYSICAL_LINE_DELTA_ROWS,
    EXPECTED_SHORT_ROWS,
    EXPECTED_TEXT_BYTES,
    ArxivAbstractsFullCensusPublicationError,
    summarize_publication,
)
from sai.data.token_stream import canonical_sha256


def _evidence() -> dict:
    payload = {
        "schema": CENSUS_SCHEMA,
        "status": "complete_text_free_full_parent_census",
        "source_id": SOURCE_ID,
        "source_snapshot": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "parents": 2,
            "compressed_bytes": SOURCE_ORIGINAL_BYTES,
            "reported_memory_bytes": SOURCE_MEMORY_BYTES,
            "rows": SOURCE_ROWS,
        },
        "authorization": {
            "decision_scope": "text_free_full_parent_census_only",
            "bulk_ingestion_authorized": False,
            "training_ready": False,
        },
        "audit_exclusions": [
            {"matched_source_rows": 4},
            {"matched_source_rows": 32},
            {"matched_source_rows": 1_024},
        ],
        "audit_excluded_positions": 1_060,
        "audit_excluded_content_identities": 1_060,
        "parents": [
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": path,
                "compressed_bytes": descriptor["bytes"],
                "compressed_sha256": descriptor["sha256"],
                "source_text_persisted": False,
            }
            for path, descriptor in EXPECTED_PARENTS.items()
        ],
        "totals": {
            "scanned_rows": SOURCE_ROWS,
            "text_rows": SOURCE_ROWS,
            "text_bytes": EXPECTED_TEXT_BYTES,
            "provenance_valid_rows": SOURCE_ROWS,
            "invalid_provenance_rows": 0,
            "non_monotonic_provenance_rows": 0,
            "source_provenance_gap_positions": EXPECTED_PROVENANCE_GAP_POSITIONS,
            "provenance_physical_line_delta_rows": (
                EXPECTED_PROVENANCE_PHYSICAL_LINE_DELTA_ROWS
            ),
            "audit_excluded_rows": 1_060,
            "audit_position_excluded_rows": 1_060,
            "audit_position_excluded_identities": 1_060,
            "audit_content_excluded_rows": 0,
            "non_cc0_declaration_rows": 0,
            "short_rows": EXPECTED_SHORT_ROWS,
            "oversized_rows": 0,
            "exact_duplicate_rows": 0,
            "duplicate_native_id_rows": 1,
            "mechanically_eligible_unique_rows": EXPECTED_ELIGIBLE_ROWS,
            "mechanically_eligible_unique_text_bytes": EXPECTED_ELIGIBLE_TEXT_BYTES,
        },
        "complete_parent_census": True,
        "maximum_simultaneous_parent_files": 1,
        "parents_removed_after_census": True,
        "source_text_persisted": False,
        "benchmark_contamination_screen_complete": False,
        "near_duplicate_filter_complete": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "full_source_ingestion_authorized": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def test_summary_seals_exact_full_census_without_source_rows() -> None:
    result = summarize_publication(_evidence())
    assert result["funnel"]["scanned_rows"] == 2_504_679
    assert result["funnel"]["mechanically_eligible_unique_rows"] == 2_458_156
    assert result["provenance"]["gap_positions"] == 1_654
    assert "text" not in result


def test_summary_rejects_audit_locator_undercoverage() -> None:
    census = copy.deepcopy(_evidence())
    census["totals"]["audit_position_excluded_identities"] -= 1
    with pytest.raises(ArxivAbstractsFullCensusPublicationError, match="evidence"):
        summarize_publication(census)
