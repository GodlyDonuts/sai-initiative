import copy
import gzip
import json

import pytest

from sai.data.common_pile_audit_population import (
    COMMON_PILE_SOURCE_TYPES,
    EXPECTED_ROWS,
    EXPECTED_SOURCES,
    CommonPileAuditError,
    build_parent_plan,
    sample_verified_gzip_parent,
)
from sai.data.frontier_source_reservoir import (
    COMMON_PILE_FILTERED_SOURCES,
    MANIFEST_SCHEMA,
    SOURCE_SPECS,
)
from sai.data.token_stream import sha256_file


def _rows() -> list[dict]:
    specs = {spec.source_id: spec for spec in SOURCE_SPECS}
    rows = []
    for index, (name, _, _) in enumerate(COMMON_PILE_FILTERED_SOURCES):
        spec = specs[f"common_pile_{name}"]
        for suffix, size in (("large", 2000), ("small", 1000)):
            rows.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "source_id": spec.source_id,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "path": f"{name}-{suffix}.json.gz",
                    "physical_bytes": size,
                    "sha256": f"{index * 2 + (suffix == 'small') + 1:064x}",
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


def test_parent_plan_covers_every_component_and_uses_smallest_parent() -> None:
    rows = _rows()
    plan = build_parent_plan(rows)
    assert EXPECTED_SOURCES == 31
    assert EXPECTED_ROWS == 124
    assert len(plan) == EXPECTED_SOURCES
    assert all(row["path"].endswith("-small.json.gz") for row in plan)
    assert {row["source_type"] for row in plan} <= {
        "textbook",
        "reference",
        "research_paper",
        "documentation",
        "code_repository",
        "forum",
        "general_web",
    }
    assert len(COMMON_PILE_SOURCE_TYPES) == EXPECTED_SOURCES
    assert build_parent_plan(list(reversed(copy.deepcopy(rows)))) == plan


def test_parent_plan_rejects_missing_component() -> None:
    rows = _rows()
    source_id = rows[0]["source_id"]
    rows = [row for row in rows if row["source_id"] != source_id]
    with pytest.raises(CommonPileAuditError, match="absent"):
        build_parent_plan(rows)


def test_gzip_sampling_is_exact_deterministic_and_license_aware(tmp_path) -> None:
    path = tmp_path / "source.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(10):
            handle.write(
                json.dumps(
                    {
                        "id": index,
                        "text": f"row {index} " + "substantive source text " * 20,
                        "metadata": {
                            "license": "CC-BY-4.0",
                            "url": f"https://example.test/{index}",
                        },
                    }
                )
                + "\n"
            )
    parent = {
        "parent_file_bytes": path.stat().st_size,
        "parent_file_sha256": sha256_file(path),
        "parent_selection_key": "a" * 64,
        "license": "fallback",
    }
    first = sample_verified_gzip_parent(path, parent)
    second = sample_verified_gzip_parent(path, copy.deepcopy(parent))
    assert first == second
    assert len(first) == 4
    assert len({row["locator"]["line_number"] for row in first}) == 4
    assert {row["declared_license"] for row in first} == {"CC-BY-4.0"}
    assert all(row["full_file_content_verified"] is True for row in first)

    parent["parent_file_sha256"] = "0" * 64
    with pytest.raises(CommonPileAuditError, match="identity"):
        sample_verified_gzip_parent(path, parent)
