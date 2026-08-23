import copy

import pytest

from sai.data.reservoir_audit_population import (
    AUDIT_STRATA,
    EXPECTED_ROWS,
    MAX_EXCERPT_BYTES,
    ReservoirAuditError,
    _candidate_and_lineage,
    _excerpt,
    build_plan,
)
from sai.data.source_reservoir import MANIFEST_SCHEMA, SOURCE_SPECS


def _rows() -> list[dict]:
    sources = {spec.source_id: spec for spec in SOURCE_SPECS}
    rows = []
    for stratum_index, stratum in enumerate(AUDIT_STRATA):
        source = sources[stratum.source_id]
        for index in range(stratum.quota):
            suffix = source.suffix
            path = f"{stratum.path_prefix}part-{stratum_index:03d}-{index:03d}{suffix}"
            rows.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "source_id": source.source_id,
                    "repository": source.repository,
                    "revision": source.revision,
                    "license": source.license,
                    "access": source.access,
                    "epistemic_function": source.epistemic_function,
                    "path": path,
                    "bytes": 1_000 + index,
                    "sha256": f"{len(rows) + 1:064x}",
                    "raw_source_is_training_ready": False,
                    "ordinal": len(rows),
                }
            )
    return rows


def test_frozen_audit_plan_covers_exact_geometry() -> None:
    plan = build_plan(_rows())
    assert len(plan) == EXPECTED_ROWS == 128
    assert len({row["selection_key"] for row in plan}) == EXPECTED_ROWS
    assert sum(row["source_id"] == "finepdfs" for row in plan) == 40
    assert sum(row["source_id"] == "dolma3_mix_150b" for row in plan) == 24
    assert sum(row["source_id"] == "fineweb_edu_fill" for row in plan) == 24


def test_plan_rejects_an_underfilled_stratum() -> None:
    rows = _rows()
    rows = [row for row in rows if row["path"] != "data/eng_Latn/part-000-000.parquet"]
    with pytest.raises(ReservoirAuditError, match="underfilled"):
        build_plan(rows)


def test_excerpt_preserves_complete_short_text_and_bounds_long_utf8() -> None:
    short = "A grounded source sentence. " * 20
    assert _excerpt(short) == (short.strip(), "complete")
    long = "数学 and engineering evidence. " * 10_000
    excerpt, method = _excerpt(long)
    assert method == "utf8_beginning_middle_end_32768"
    assert 200 <= len(excerpt.encode()) <= MAX_EXCERPT_BYTES
    assert excerpt.startswith("数学 and engineering evidence.")
    assert excerpt.endswith("数学 and engineering evidence.")


def test_candidate_binds_full_source_and_exact_locator() -> None:
    plan = build_plan(_rows())[0]
    acquired = {
        "text": "Grounded mathematics and science source. " * 20,
        "locator": {
            "format": "parquet",
            "row_group": 3,
            "row_in_group": 4,
            "row_index": 3004,
        },
        "full_file_content_verified": False,
    }
    candidate, lineage = _candidate_and_lineage(plan, acquired)
    assert (
        candidate["candidate_identity_sha256"] == lineage["candidate_identity_sha256"]
    )
    assert lineage["locator"]["row_index"] == 3004
    assert lineage["raw_source_is_training_ready"] is False
    changed = copy.deepcopy(acquired)
    changed["locator"]["row_index"] += 1
    other, _ = _candidate_and_lineage(plan, changed)
    assert other["candidate_identity_sha256"] != candidate["candidate_identity_sha256"]


def test_candidate_lineage_accepts_a_source_specific_text_column() -> None:
    plan = build_plan(_rows())[0]
    plan["text_column"] = "content"
    candidate, lineage = _candidate_and_lineage(
        plan,
        {
            "text": "Grounded source-specific content column. " * 20,
            "locator": {"format": "parquet", "row_index": 7},
            "full_file_content_verified": False,
        },
    )
    assert candidate["text"].startswith("Grounded source-specific")
    assert lineage["locator"]["row_index"] == 7
    assert lineage["text_column"] == "content"
