import json
from pathlib import Path

import pytest

from sai.data.final_training_release import SCHEMA as RELEASE_SCHEMA
from sai.data.final_training_release_hf_publish import (
    DESTINATION_PATH,
    FinalTrainingReleaseHfPublishError,
    publish,
)
from sai.data.token_stream import canonical_sha256


class _Lfs:
    size = 123
    sha256 = "a" * 64


class _Sibling:
    rfilename = DESTINATION_PATH
    size = 123
    lfs = _Lfs()


class _Info:
    sha = "remote-revision"
    siblings = [_Sibling()]


def _release(path: Path) -> dict:
    payload = {
        "schema": RELEASE_SCHEMA,
        "status": "complete_sai_training_data_release",
        "all_required_components_present": True,
        "verified_cross_domain_connection_overlay_complete": True,
        "connection_development_rows_physically_excluded": True,
        "training_data_ready": True,
        "model_training_started": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return payload


def test_publish_replays_final_release_lfs(tmp_path: Path, monkeypatch) -> None:
    release = _release(tmp_path / "release.json")
    monkeypatch.setattr(
        "sai.data.final_training_release_hf_publish.upload_verified",
        lambda *args, **kwargs: {
            "path": DESTINATION_PATH,
            "bytes": 123,
            "sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        "huggingface_hub.HfApi.dataset_info", lambda *args, **kwargs: _Info()
    )
    result = publish(tmp_path / "release.json", tmp_path / "out.json", "token")
    assert result["release_receipt_sha256"] == release["receipt_sha256"]
    assert result["verified_cross_domain_connection_overlay_in_release"] is True
    assert result["all_remote_lfs_identities_verified"] is True


def test_publish_rejects_release_without_connection_component(
    tmp_path: Path,
) -> None:
    path = tmp_path / "release.json"
    payload = _release(path)
    payload.pop("receipt_sha256")
    payload["verified_cross_domain_connection_overlay_complete"] = False
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(FinalTrainingReleaseHfPublishError, match="manifest differs"):
        publish(path, tmp_path / "out.json", "token")
