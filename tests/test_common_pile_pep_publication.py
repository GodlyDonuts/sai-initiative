import copy

import pytest

from sai.data.common_pile_pep_census import SCHEMA as CENSUS_SCHEMA
from sai.data.common_pile_pep_census import SOURCE_ID
from sai.data.common_pile_pep_publication import (
    PARENT_BYTES,
    PARENT_PATH,
    PARENT_SHA256,
    REPOSITORY,
    REVISION,
    CommonPilePepPublicationError,
    summarize_publication,
)
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256


def _signed(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _evidence() -> tuple[dict, dict]:
    census = _signed(
        {
            "schema": CENSUS_SCHEMA,
            "status": "complete_filtered_nontraining_parent_census",
            "source_id": SOURCE_ID,
            "complete_parent_census": True,
            "parent_removed_after_census": True,
            "maximum_simultaneous_parent_files": 1,
            "parent": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": PARENT_PATH,
                "bytes": PARENT_BYTES,
                "sha256": PARENT_SHA256,
            },
            "scan": {
                "scanned_rows": 655,
                "audit_excluded_rows": 36,
                "short_rows": 1,
                "eligible_rows": 618,
                "selected_rows": 618,
            },
            "decontamination": {"scanned": 618, "accepted": 568, "dropped": 50},
            "near_duplicate_filter": {
                "input_documents": 568,
                "output_documents": 567,
                "documents_dropped": 1,
                "duplicate_groups": 1,
            },
            "attribution_manifest": {
                "records": 567,
                "source_text_persisted_in_manifest": False,
            },
            "recovery_evidence": {
                "bulk_training_admission": False,
                "source_wide_quality_admission": False,
                "training_ready": False,
            },
            "quality_compilation_complete": False,
            "representation_verification_complete": False,
            "training_ready": False,
        }
    )
    population = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "source_census": {"receipt_sha256": census["receipt_sha256"]},
            "population": {"rows": 567},
            "lineage": {"rows": 567},
            "by_source": {SOURCE_ID: 567},
            "complete_census_survivor_coverage": True,
            "benchmark_contamination_screen_complete": True,
            "bounded_near_duplicate_filter_complete": True,
            "exact_attribution_coverage": True,
            "hermes_judgments_complete": False,
            "quality_compilation_complete": False,
            "representation_verification_complete": False,
            "training_ready": False,
        }
    )
    return census, population


def test_summary_seals_exact_source_safe_funnel() -> None:
    census, population = _evidence()
    result = summarize_publication(census, population)
    assert result["funnel"]["scanned_rows"] == 655
    assert result["funnel"]["benchmark_overlap_drops"] == 50
    assert result["funnel"]["final_unique_rows"] == 567
    assert "text" not in result


def test_summary_rejects_incomplete_survivor_coverage() -> None:
    census, population = _evidence()
    population["complete_census_survivor_coverage"] = False
    population["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in population.items() if key != "receipt_sha256"}
    )
    with pytest.raises(CommonPilePepPublicationError, match="evidence"):
        summarize_publication(census, population)
