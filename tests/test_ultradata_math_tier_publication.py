import copy

import pytest

from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256
from sai.data.ultradata_math_tier_audit_population import REPOSITORY, REVISION
from sai.data.ultradata_math_tier_publication import (
    UltraDataMathTierPublicationError,
    summarize_publication,
)


def _signed(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _evidence() -> tuple[dict, dict, dict]:
    source = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "source_snapshot": {"repository": REPOSITORY, "revision": REVISION},
            "population": {"rows": 160},
            "batch_receipts": {"rows": 20},
            "by_stratum": {"a": 32},
            "hermes_judgments_complete": False,
            "training_ready": False,
        }
    )
    clean = _signed(
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "source_population": {"receipt_sha256": source["receipt_sha256"]},
            "input_rows": 160,
            "clean_rows": 148,
            "contaminated_rows": 12,
            "word_overlap_shingles": 20,
            "code_overlap_shingles": 12,
            "population": {"rows": 148},
            "by_stratum": {"a": 29},
            "benchmark_contamination_screen_complete": True,
            "hermes_judgments_complete": False,
            "training_ready": False,
        }
    )
    screen = _signed(
        {
            "schema": SCREEN_SCHEMA,
            "status": "complete",
            "population": {"receipt_sha256": source["receipt_sha256"]},
            "summary": {
                "rows": 160,
                "clean_rows": 148,
                "contaminated_rows": 12,
                "word_overlap_shingles": 20,
                "code_overlap_shingles": 12,
            },
        }
    )
    return source, clean, screen


def test_summary_is_source_safe_and_exact() -> None:
    source, clean, screen = _evidence()
    result = summarize_publication(source, clean, screen)
    assert result["input_rows"] == 160
    assert result["clean_rows"] == 148
    assert "text" not in result


def test_summary_rejects_inconsistent_contamination() -> None:
    source, clean, screen = _evidence()
    screen["summary"]["clean_rows"] = 149
    screen["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in screen.items() if key != "receipt_sha256"}
    )
    with pytest.raises(UltraDataMathTierPublicationError, match="evidence"):
        summarize_publication(source, clean, screen)
