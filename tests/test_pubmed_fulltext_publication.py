import copy

import pytest

from sai.data.common_pile_rights_audit import SCHEMA as RIGHTS_SCHEMA
from sai.data.pubmed_fulltext_audit_population import (
    EXPECTED_ROWS,
    INDEX_STRATA,
    REPOSITORY,
    REVISION,
    WINDOWS_PER_STRATUM,
)
from sai.data.pubmed_fulltext_publication import (
    PubmedFulltextPublicationError,
    summarize_publication,
)
from sai.data.reservoir_audit_duplicates import SCHEMA as DUPLICATE_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256


def _signed(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _evidence() -> tuple[dict, dict, dict, dict]:
    source = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "source_snapshot": {"repository": REPOSITORY, "revision": REVISION},
            "population": {"rows": EXPECTED_ROWS},
            "batch_receipts": {"rows": INDEX_STRATA * WINDOWS_PER_STRATUM},
            "by_stratum": {"index_00": 32},
            "by_license": {"CC-BY-4.0": 972, "CC0-1.0": 52},
            "training_ready": False,
        }
    )
    clean = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "source_population": {"receipt_sha256": source["receipt_sha256"]},
            "input_rows": EXPECTED_ROWS,
            "clean_rows": 1007,
            "contaminated_rows": 17,
            "word_overlap_shingles": 37,
            "code_overlap_shingles": 9,
            "population": {"rows": 1007},
            "by_stratum": {"index_00": 32},
            "benchmark_contamination_screen_complete": True,
            "training_ready": False,
        }
    )
    rights = _signed(
        {
            "schema": RIGHTS_SCHEMA,
            "population": {"receipt_sha256": source["receipt_sha256"]},
            "summary": {
                "rows": EXPECTED_ROWS,
                "by_source": {"common_pile_pubmed": {"rights_hold_rows": 0}},
            },
            "training_ready": False,
        }
    )
    duplicates = _signed(
        {
            "schema": DUPLICATE_SCHEMA,
            "population_receipt_sha256": source["receipt_sha256"],
            "candidate_rows": EXPECTED_ROWS,
            "flagged_pairs": 0,
            "audit_sample_deduplication_complete": True,
            "training_ready": False,
        }
    )
    return source, clean, rights, duplicates


def test_summary_is_source_safe_and_exact() -> None:
    result = summarize_publication(*_evidence())
    assert result["input_rows"] == EXPECTED_ROWS
    assert result["clean_rows"] == 1007
    assert result["rights_hold_rows"] == 0
    assert result["flagged_near_duplicate_pairs"] == 0
    assert "text" not in result


def test_summary_rejects_inconsistent_clean_coverage() -> None:
    source, clean, rights, duplicates = _evidence()
    clean["clean_rows"] = 1006
    clean["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in clean.items() if key != "receipt_sha256"}
    )
    with pytest.raises(PubmedFulltextPublicationError, match="evidence"):
        summarize_publication(source, clean, rights, duplicates)
