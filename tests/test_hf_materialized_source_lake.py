import copy

import pytest

from sai.data.hf_materialized_source_lake import (
    DESTINATION_REPOSITORY,
    FILE_SCHEMA,
    TARGET_BYTES,
    MaterializedSourceLakeError,
    summarize_lake,
)

REVISION = "a" * 40


def _rows() -> list[dict]:
    return [
        {
            "schema": FILE_SCHEMA,
            "destination_repository": DESTINATION_REPOSITORY,
            "destination_revision": REVISION,
            "destination_path": "sources/example/data/part-000.parquet",
            "source_id": "example",
            "source_repository": "owner/source",
            "source_revision": "b" * 40,
            "source_path": "part-000.parquet",
            "bytes": TARGET_BYTES,
            "sha256": "c" * 64,
            "source_manifest_path": "sources/example/source-manifest.json",
            "raw_source_is_training_ready": False,
        }
    ]


def _components() -> list[dict]:
    return [
        {
            "source_id": "example",
            "materialized_files": 1,
            "materialized_bytes": TARGET_BYTES,
        }
    ]


def test_summary_proves_exact_target_without_ready_claim() -> None:
    result = summarize_lake(_rows(), REVISION, _components())
    assert result["target_met"] is True
    assert result["materialized_bytes"] == TARGET_BYTES
    assert result["by_source"]["example"]["files"] == 1


def test_summary_rejects_tampered_identity() -> None:
    rows = copy.deepcopy(_rows())
    rows[0]["sha256"] = "not-a-hash"
    with pytest.raises(MaterializedSourceLakeError, match="file"):
        summarize_lake(rows, REVISION, _components())


def test_summary_rejects_component_accounting_drift() -> None:
    components = copy.deepcopy(_components())
    components[0]["materialized_bytes"] -= 1
    with pytest.raises(MaterializedSourceLakeError, match="component"):
        summarize_lake(_rows(), REVISION, components)


def test_summary_rejects_sub_target_materialization() -> None:
    rows = copy.deepcopy(_rows())
    rows[0]["bytes"] = TARGET_BYTES - 1
    components = copy.deepcopy(_components())
    components[0]["materialized_bytes"] = TARGET_BYTES - 1
    with pytest.raises(MaterializedSourceLakeError, match="below target"):
        summarize_lake(rows, REVISION, components)
