import json
from pathlib import Path

import pytest

from sai.data.bridge_component_admission import SCHEMA as BRIDGE_ADMISSION_SCHEMA
from sai.data.bridge_component_hf_publish import SCHEMA as BRIDGE_PUBLICATION_SCHEMA
from sai.data.common_pile_stack_edu_practical_admission import (
    SCHEMA as CODE_ADMISSION_SCHEMA,
)
from sai.data.common_pile_stack_edu_practical_hf_publish import (
    METADATA_SCHEMA as CODE_PUBLICATION_SCHEMA,
)
from sai.data.final_training_release import (
    FinalTrainingReleaseError,
    build_release,
)
from sai.data.practical_corpus_audit import SCHEMA as FOUNDATION_SCHEMA
from sai.data.token_stream import canonical_sha256


def _write(path: Path, payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    foundation_path = tmp_path / "foundation.json"
    code_path = tmp_path / "code.json"
    code_pub_path = tmp_path / "code-pub.json"
    bridge_path = tmp_path / "bridge.json"
    bridge_pub_path = tmp_path / "bridge-pub.json"
    foundation = _write(
        foundation_path,
        {
            "schema": FOUNDATION_SCHEMA,
            "status": "complete_practical_training_corpus_readiness_audit",
            "totals": {"rows": 100, "text_utf8_bytes": 1_950_000_000_000},
            "bounds": {"combined_byte_bound_satisfied": True},
            "quality": {"english_only": True},
            "practical_training_corpus_ready": True,
            "training_ready": True,
            "four_b_training_authorized": False,
        },
    )
    code = _write(
        code_path,
        {
            "schema": CODE_ADMISSION_SCHEMA,
            "status": "complete_common_pile_stack_edu_practical_admission",
            "counts": {"admitted_rows": 20, "admitted_text_utf8_bytes": 500},
            "practical_pretraining_ready": True,
            "training_ready": True,
            "four_b_training_authorized": False,
        },
    )
    _write(
        code_pub_path,
        {
            "schema": CODE_PUBLICATION_SCHEMA,
            "status": "complete_stack_edu_practical_hf_metadata_publication",
            "admission_receipt_sha256": code["receipt_sha256"],
            "source_text_uploaded": False,
            "training_ready": True,
        },
    )
    bridge = _write(
        bridge_path,
        {
            "schema": BRIDGE_ADMISSION_SCHEMA,
            "status": "complete_bridge_training_component_admission",
            "train": {"text_utf8_bytes": 300},
            "counts": {
                "train_documents": 12,
                "development_documents_excluded": 4,
            },
            "connection_component_admission_authorized": True,
            "transfer_ablation_complete": True,
            "development_rows_physically_excluded": True,
            "training_ready": True,
            "four_b_training_authorized": False,
        },
    )
    _write(
        bridge_pub_path,
        {
            "schema": BRIDGE_PUBLICATION_SCHEMA,
            "status": "complete_bridge_training_component_hf_publication",
            "admission_receipt_sha256": bridge["receipt_sha256"],
            "train_documents": 12,
            "train_text_utf8_bytes": 300,
            "development_rows_uploaded": False,
            "transfer_ablation_complete": True,
            "training_ready": True,
            "four_b_training_authorized": False,
        },
    )
    assert foundation["receipt_sha256"]
    return foundation_path, code_path, code_pub_path, bridge_path, bridge_pub_path


def test_release_requires_and_binds_all_components(tmp_path: Path) -> None:
    result = build_release(*_inputs(tmp_path), tmp_path / "release.json")
    assert result["status"] == "complete_sai_training_data_release"
    assert result["totals"] == {
        "components": 3,
        "rows": 132,
        "logical_text_utf8_bytes": 1_950_000_000_800,
        "foundation_text_utf8_bytes": 1_950_000_000_000,
        "overlay_text_utf8_bytes": 800,
    }
    assert result["verified_cross_domain_connection_overlay_complete"] is True
    assert result["connection_development_rows_physically_excluded"] is True
    assert result["training_data_ready"] is True
    assert result["model_training_started"] is False
    assert result["four_b_training_authorized"] is False


def test_release_rejects_omitted_connection_transfer_gate(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    payload = json.loads(paths[3].read_text())
    payload.pop("receipt_sha256")
    payload["transfer_ablation_complete"] = False
    _write(paths[3], payload)
    with pytest.raises(FinalTrainingReleaseError, match="component evidence"):
        build_release(*paths, tmp_path / "release.json")


def test_release_rejects_underfilled_foundation(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    payload = json.loads(paths[0].read_text())
    payload.pop("receipt_sha256")
    payload["totals"]["text_utf8_bytes"] = 1_899_999_999_999
    _write(paths[0], payload)
    with pytest.raises(FinalTrainingReleaseError, match="byte bound"):
        build_release(*paths, tmp_path / "release.json")
