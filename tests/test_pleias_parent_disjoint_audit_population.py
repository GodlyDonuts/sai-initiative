from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sai.data.frontier_source_reservoir import MANIFEST_SCHEMA
from sai.data.pleias_parent_disjoint_audit_population import (
    ACQUISITION_MODES,
    EXPECTED_ROWS,
    PARTITION_QUOTAS,
    PleiasParentDisjointAuditError,
    build_plan,
    prior_parent_identities,
)


def _row(partition: int, index: int) -> dict:
    ordinal = partition * 10_000 + index
    return {
        "schema": MANIFEST_SCHEMA,
        "source_id": "pleias_common_corpus",
        "repository": "PleIAs/common_corpus",
        "revision": "a" * 40,
        "path": f"common_corpus_{partition}/part-{index:05d}.parquet",
        "physical_bytes": 400_000_000 + index,
        "sha256": f"{ordinal:064x}",
        "text_column": "text",
        "license": "source_specific_public_domain_or_open_license",
        "access": "public",
    }


def _rows() -> list[dict]:
    return [
        _row(partition, index)
        for partition, quota in PARTITION_QUOTAS.items()
        for index in range(quota + 2)
    ]


def test_plan_is_exact_partitioned_disjoint_and_order_independent() -> None:
    assert ACQUISITION_MODES == {"range", "full_verified_parent"}
    rows = _rows()
    excluded_row = rows[0]
    excluded = frozenset(
        {
            (
                excluded_row["repository"],
                excluded_row["revision"],
                excluded_row["path"],
            )
        }
    )
    plan = build_plan(rows, excluded)
    assert len(plan) == EXPECTED_ROWS == 1024
    assert len({(row["repository"], row["path"]) for row in plan}) == EXPECTED_ROWS
    assert (excluded_row["repository"], excluded_row["path"]) not in {
        (row["repository"], row["path"]) for row in plan
    }
    assert build_plan(list(reversed(copy.deepcopy(rows))), excluded) == plan
    for partition, quota in PARTITION_QUOTAS.items():
        assert (
            sum(row["stratum"] == f"open_corpus_partition:{partition}" for row in plan)
            == quota
        )


def test_plan_rejects_partition_underfill() -> None:
    rows = [row for row in _rows() if not row["path"].startswith("common_corpus_10/")]
    with pytest.raises(PleiasParentDisjointAuditError, match="underfilled"):
        build_plan(rows, frozenset())


def test_prior_parent_identities_are_source_specific(tmp_path: Path) -> None:
    path = tmp_path / "lineage.jsonl"
    values = [
        {
            "source_id": "other",
            "repository": "x",
            "revision": "b" * 40,
            "path": "ignored.parquet",
        },
        {
            "source_id": "pleias_common_corpus",
            "repository": "PleIAs/common_corpus",
            "revision": "a" * 40,
            "path": "common_corpus_1/a.parquet",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in values))
    assert prior_parent_identities(path) == frozenset(
        {("PleIAs/common_corpus", "a" * 40, "common_corpus_1/a.parquet")}
    )
