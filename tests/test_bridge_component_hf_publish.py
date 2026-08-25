from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sai.data.bridge_component_admission import SCHEMA as ADMISSION_SCHEMA
from sai.data.bridge_component_hf_publish import (
    DESTINATION_PREFIX,
    DESTINATION_REPOSITORY,
    publish,
    record_preverified_publication,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _admission(root: Path) -> None:
    root.mkdir()
    train = root / "train.jsonl.gz"
    train.write_bytes(b"deterministic-gzip-fixture")
    payload = {
        "schema": ADMISSION_SCHEMA,
        "status": "complete_bridge_training_component_admission",
        "train": {
            "path": train.name,
            "rows": 2,
            "bytes": train.stat().st_size,
            "sha256": sha256_file(train),
            "text_utf8_bytes": 1234,
        },
        "counts": {"train_documents": 2},
        "development_rows_physically_excluded": True,
        "transfer_ablation_complete": True,
        "connection_component_admission_authorized": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    (root / "receipt.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_publishes_and_replays_only_train_component(
    tmp_path: Path, monkeypatch
) -> None:
    admission = tmp_path / "admission"
    _admission(admission)
    remotes: list[dict] = []

    def fake_upload(path, remote_path, token, *, repository):
        assert token == "token"
        assert repository == DESTINATION_REPOSITORY
        value = {
            "repository": repository,
            "path": remote_path,
            "bytes": Path(path).stat().st_size,
            "sha256": sha256_file(Path(path)),
        }
        remotes.append(value)
        return value

    class FakeApi:
        def __init__(self, token):
            assert token == "token"

        def dataset_info(self, repository, files_metadata):
            assert repository == DESTINATION_REPOSITORY
            assert files_metadata is True
            siblings = [
                SimpleNamespace(
                    rfilename=row["path"],
                    size=row["bytes"],
                    lfs=SimpleNamespace(size=row["bytes"], sha256=row["sha256"]),
                )
                for row in remotes
            ]
            return SimpleNamespace(sha="f" * 40, siblings=siblings)

    monkeypatch.setattr(
        "sai.data.bridge_component_hf_publish.upload_verified", fake_upload
    )
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    result = publish(admission, tmp_path / "publication.json", "token")
    assert result["training_ready"] is True
    assert result["development_rows_uploaded"] is False
    assert result["four_b_training_authorized"] is False
    assert [row["path"] for row in remotes] == [
        f"{DESTINATION_PREFIX}/train.jsonl.gz",
        f"{DESTINATION_PREFIX}/receipt.lfs.json",
    ]


def test_records_one_preverified_bridge_commit(tmp_path: Path) -> None:
    admission = tmp_path / "admission"
    _admission(admission)
    revision = "a" * 40
    paths = [
        (admission / "train.jsonl.gz", f"{DESTINATION_PREFIX}/train.jsonl.gz"),
        (admission / "receipt.json", f"{DESTINATION_PREFIX}/receipt.lfs.json"),
    ]
    siblings = [
        SimpleNamespace(
            rfilename=remote,
            size=path.stat().st_size,
            lfs=SimpleNamespace(size=path.stat().st_size, sha256=sha256_file(path)),
        )
        for path, remote in paths
    ]
    api = SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(
            sha=revision, siblings=siblings
        )
    )
    result = record_preverified_publication(
        admission, tmp_path / "publication.json", revision, "token", api=api
    )
    assert result["remote_revision_verified"] == revision
    assert {row["commit"] for row in result["remote_outputs"]} == {revision}
    assert result["development_rows_uploaded"] is False
