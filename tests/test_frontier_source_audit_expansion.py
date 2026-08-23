import copy

import pytest

from sai.data.frontier_source_audit_expansion import (
    AUDIT_STRATA,
    EXPECTED_ROWS,
    FrontierSourceAuditExpansionError,
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


def test_expansion_plan_has_exact_source_screen_geometry() -> None:
    plan = build_plan(_rows())
    assert len(plan) == EXPECTED_ROWS == 91
    assert len({(row["repository"], row["path"]) for row in plan}) == 91
    assert sum(row["source_id"] == "pleias_common_corpus" for row in plan) == 40
    assert sum(row["source_id"] == "nemotron_specialized_v1_2" for row in plan) == 30
    assert sum(row["source_id"] == "nemotron_legal_v1" for row in plan) == 21


def test_expansion_plan_is_order_independent_and_rejects_underfill() -> None:
    rows = _rows()
    expected = build_plan(rows)
    assert build_plan(list(reversed(copy.deepcopy(rows)))) == expected
    rows.pop()
    with pytest.raises(FrontierSourceAuditExpansionError, match="underfilled"):
        build_plan(rows)
