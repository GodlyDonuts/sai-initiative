import copy

import pytest

from sai.data.frontier_source_audit_population import (
    AUDIT_STRATA,
    EXPECTED_ROWS,
    FrontierSourceAuditError,
    build_plan,
)
from sai.data.frontier_source_reservoir import MANIFEST_SCHEMA, SOURCE_SPECS


def _rows() -> list[dict]:
    specs = {spec.source_id: spec for spec in SOURCE_SPECS}
    rows = []
    for stratum_index, stratum in enumerate(AUDIT_STRATA):
        spec = specs[stratum.source_id]
        for file_index in range(stratum.quota):
            rows.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "source_id": spec.source_id,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "path": (
                        f"{stratum.path_prefix}part-{stratum_index:03d}-"
                        f"{file_index:04d}.parquet"
                    ),
                    "physical_bytes": 1_000_000 + file_index,
                    "sha256": f"{len(rows) + 1:064x}",
                    "text_column": spec.text_column,
                    "license": spec.license,
                    "access": spec.access,
                    "epistemic_function": spec.epistemic_function,
                    "physical_bytes_are_text_payload_bytes": False,
                    "source_candidate_is_training_ready": False,
                    "ordinal": len(rows),
                }
            )
    return rows


def test_frontier_audit_plan_has_exact_diverse_geometry() -> None:
    plan = build_plan(_rows())
    assert len(plan) == EXPECTED_ROWS == 512
    assert len({(row["repository"], row["path"]) for row in plan}) == 512
    assert sum(row["source_id"] == "fineweb2_hq_multilingual" for row in plan) == 160
    assert (
        sum(row["source_id"] == "nemotron_specialized_reasoning" for row in plan) == 96
    )
    assert {row["text_column"] for row in plan} == {"content", "text"}


def test_frontier_audit_plan_is_order_independent_and_rejects_underfill() -> None:
    rows = _rows()
    expected = build_plan(rows)
    assert build_plan(list(reversed(copy.deepcopy(rows)))) == expected
    rows.pop()
    with pytest.raises(FrontierSourceAuditError, match="underfilled"):
        build_plan(rows)
