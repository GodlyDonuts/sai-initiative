import copy

import pytest

from sai.data.hf_materialized_source_lake import RECEIPT_SCHEMA as LAKE_SCHEMA
from sai.data.materialized_source_admission_matrix import (
    ADMISSION_GATES,
    MaterializedSourceAdmissionMatrixError,
    build_matrix_payload,
)
from sai.data.reservoir_rights_inventory import SCHEMA as RIGHTS_SCHEMA


def _evidence() -> tuple[dict, dict]:
    lake = {
        "schema": LAKE_SCHEMA,
        "target_met": True,
        "destination_repository": "Godlydonuts/Sai",
        "destination_revision": "a" * 40,
        "receipt_sha256": "b" * 64,
        "materialized_files": 2,
        "materialized_bytes": 100,
        "all_destination_lfs_identities_replayed_against_pinned_upstream": True,
        "by_source": {"example": {"files": 2, "bytes": 100}},
        "components": [
            {
                "source_id": "example",
                "source_repository": "owner/source",
                "source_revision": "c" * 40,
                "source_manifest_path": "sources/example/source-manifest.json",
                "materialized_files": 2,
                "materialized_bytes": 100,
                "complete_source_snapshot": True,
                "training_ready": False,
            }
        ],
        "training_ready": False,
    }
    rights = {
        "schema": RIGHTS_SCHEMA,
        "receipt_sha256": "d" * 64,
        "license_policy_sha256": "e" * 64,
        "source_rows": [
            {
                "source_id": "upstream_example",
                "repository": "owner/source",
                "revision": "c" * 40,
                "files": 3,
                "bytes": 120,
                "rights_work_route": "recognized_declaration_obligations_required",
                "declared_license": "apache-2.0",
                "card_license_declarations": ["apache-2.0"],
                "source_wide_rights_clearance_established": False,
                "legal_clearance_established": False,
                "training_ready": False,
            }
        ],
        "training_ready": False,
    }
    return lake, rights


def test_matrix_covers_materialized_bytes_without_ready_claim() -> None:
    result = build_matrix_payload(*_evidence())
    row = result["rows"][0]
    assert result["summary"]["materialized_bytes"] == 100
    assert row["materialized_identity_verified"] is True
    assert row["blocking_gates"] == list(ADMISSION_GATES)
    assert all(value is False for value in row["admission_gates"].values())
    assert result["training_ready"] is False


def test_matrix_rejects_missing_rights_identity() -> None:
    lake, rights = _evidence()
    rights["source_rows"][0]["revision"] = "f" * 40
    with pytest.raises(MaterializedSourceAdmissionMatrixError, match="accounting"):
        build_matrix_payload(lake, rights)


def test_matrix_rejects_rights_inventory_smaller_than_custody() -> None:
    lake, rights = _evidence()
    rights["source_rows"][0]["bytes"] = 99
    with pytest.raises(MaterializedSourceAdmissionMatrixError, match="accounting"):
        build_matrix_payload(lake, rights)


def test_matrix_rejects_ready_rights_claim() -> None:
    lake, rights = _evidence()
    tampered = copy.deepcopy(rights)
    tampered["source_rows"][0]["training_ready"] = True
    with pytest.raises(MaterializedSourceAdmissionMatrixError, match="identity"):
        build_matrix_payload(lake, tampered)
