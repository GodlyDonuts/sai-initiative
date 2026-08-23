import copy

import pytest

from sai.data.arxiv_abstracts_audit_population import (
    REPOSITORY,
    REVISION,
    SOURCE_ID,
    SOURCE_ROWS,
)
from sai.data.arxiv_abstracts_audit_publication import (
    ArxivAbstractsAuditPublicationError,
    summarize_publication,
)
from sai.data.common_pile_rights_audit import SCHEMA as RIGHTS_SCHEMA
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
            "source_snapshot": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "rows": SOURCE_ROWS,
            },
            "population": {"rows": 1024},
            "dataset_server_batches": 32,
            "source_disjoint_from_audit_populations": True,
            "source_declared_cc0": True,
            "rights_verification_complete": False,
            "benchmark_contamination_screen_complete": False,
            "hermes_judgments_complete": False,
            "training_ready": False,
        }
    )
    clean = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "source_population": {"receipt_sha256": source["receipt_sha256"]},
            "input_rows": 1024,
            "clean_rows": 1023,
            "contaminated_rows": 1,
            "word_overlap_shingles": 0,
            "code_overlap_shingles": 1,
            "population": {"rows": 1023},
            "by_source": {SOURCE_ID: 1023},
            "benchmark_contamination_screen_complete": True,
            "hermes_judgments_complete": False,
            "training_ready": False,
        }
    )
    duplicates = _signed(
        {
            "schema": DUPLICATE_SCHEMA,
            "status": "complete",
            "population_receipt_sha256": clean["receipt_sha256"],
            "candidate_rows": 1023,
            "candidate_pairs_compared": 1023 * 1022 // 2,
            "flagged_pairs": 0,
            "cross_source_flagged_pairs": 0,
            "pairs": [],
            "groups": [],
            "audit_sample_deduplication_complete": True,
            "full_reservoir_deduplication_complete": False,
            "training_ready": False,
        }
    )
    rights = _signed(
        {
            "schema": RIGHTS_SCHEMA,
            "status": "complete_declaration_audit_not_legal_clearance",
            "population": {"receipt_sha256": source["receipt_sha256"]},
            "summary": {
                "rows": 1024,
                "by_source": {
                    SOURCE_ID: {
                        "rows": 1024,
                        "recognized_declaration_rows": 1024,
                        "canonical_license:CC0-1.0": 1024,
                        "rights_hold_rows": 0,
                        "attribution_required_rows": 0,
                        "share_alike_required_rows": 0,
                    }
                },
            },
            "source_provenance_verification_complete": False,
            "source_wide_rights_clearance_established": False,
            "legal_clearance_established": False,
            "training_ready": False,
        }
    )
    return source, clean, duplicates, rights


def test_summary_seals_clean_temporal_screen() -> None:
    result = summarize_publication(*_evidence())
    assert result["input_rows"] == 1024
    assert result["clean_rows"] == 1023
    assert result["candidate_pairs_compared"] == 522_753
    assert result["near_duplicate_pairs"] == 0
    assert "text" not in result


def test_summary_rejects_duplicate_pair() -> None:
    source, clean, duplicates, rights = _evidence()
    duplicates["flagged_pairs"] = 1
    duplicates["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in duplicates.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ArxivAbstractsAuditPublicationError, match="evidence"):
        summarize_publication(source, clean, duplicates, rights)
