import copy

import pytest

from sai.data.reservoir_audit_expansion import (
    AUDIT_SOURCE_IDS,
    ReservoirAuditExpansionError,
    allocate_source_quotas,
    build_weighted_plan,
)


def _rows(files_per_source: int = 50) -> list[dict]:
    rows = []
    for source_index, source_id in enumerate(AUDIT_SOURCE_IDS, start=1):
        for file_index in range(files_per_source):
            rows.append(
                {
                    "source_id": source_id,
                    "repository": f"example/{source_id}",
                    "revision": "a" * 40,
                    "license": "odc-by-1.0",
                    "access": "public",
                    "path": (
                        f"data/{source_id}/part-{file_index:05d}.jsonl.zst"
                        if source_id == "dolma3_mix_150b"
                        else f"data/{source_id}/part-{file_index:05d}.parquet"
                    ),
                    "bytes": source_index * 1_000_000 + file_index,
                    "sha256": f"{len(rows) + 1:064x}",
                }
            )
    return rows


def test_quota_has_exact_total_floor_and_byte_weighting() -> None:
    quotas = allocate_source_quotas(_rows(), total_rows=120, minimum_rows_per_source=10)
    assert sum(quotas.values()) == 120
    assert min(quotas.values()) >= 10
    assert quotas[AUDIT_SOURCE_IDS[-1]] > quotas[AUDIT_SOURCE_IDS[0]]


def test_weighted_plan_is_deterministic_unique_and_disjoint() -> None:
    rows = _rows()
    excluded = {(rows[0]["repository"], rows[0]["path"])}
    first, quotas = build_weighted_plan(
        rows, excluded, total_rows=120, minimum_rows_per_source=10
    )
    second, other_quotas = build_weighted_plan(
        copy.deepcopy(rows), excluded, total_rows=120, minimum_rows_per_source=10
    )
    assert first == second
    assert quotas == other_quotas
    assert len(first) == 120
    parent_keys = {(row["repository"], row["path"]) for row in first}
    assert len(parent_keys) == 120
    assert parent_keys.isdisjoint(excluded)
    assert {row["source_id"] for row in first} == set(AUDIT_SOURCE_IDS)


def test_quota_rejects_insufficient_geometry() -> None:
    with pytest.raises(ReservoirAuditExpansionError, match="geometry"):
        allocate_source_quotas(_rows(), total_rows=11, minimum_rows_per_source=2)
