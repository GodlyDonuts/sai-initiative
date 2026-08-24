from types import SimpleNamespace

import pytest

from sai.data.pleias_production_materializer import (
    PleiasProductionMaterializerError,
    replay_selected_row,
    upload_verified,
)
from sai.data.pleias_semantic_sample import _token_band
from sai.data.token_stream import canonical_sha256, sha256_file


def _row(text):
    return {
        "identifier": "book-one",
        "collection": "Books",
        "open_type": "Open Culture",
        "license": "Public Domain",
        "language": "English",
        "word_count": len(text.split()),
        "token_count": len(text.split()) * 2,
        "text": text,
    }


def test_replays_exact_selected_identity_and_rejects_mutation():
    import hashlib

    text = "astronomy observation evidence measurement telescope orbit " * 20
    row = _row(text)
    parent = {
        "source_path": "data/p.parquet",
        "sha256": "a" * 64,
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "b" * 40,
    }
    content_sha256 = hashlib.sha256(text.encode()).hexdigest()
    identity = canonical_sha256(
        {
            "source_path": parent["source_path"],
            "row_index": 3,
            "identifier": row["identifier"],
            "content_sha256": content_sha256,
        }
    )
    selected = (
        identity,
        parent["sha256"],
        content_sha256,
        "::".join(
            (row["collection"], row["open_type"], _token_band(row["token_count"]))
        ),
        len(text.encode()),
        row["token_count"],
        7_500,
        8_000,
    )
    result = replay_selected_row(row, parent, 3, selected)
    assert result["source_row_identity_sha256"] == identity
    assert result["text"] == text
    assert result["semantic_quality_floor_milli"] == 7_500
    assert result["semantic_quality_mean_milli"] == 8_000
    assert result["training_ready"] is False
    changed = list(selected)
    changed[2] = "f" * 64
    with pytest.raises(PleiasProductionMaterializerError, match="identity"):
        replay_selected_row(row, parent, 3, tuple(changed))


def test_upload_replays_remote_lfs_identity(tmp_path, monkeypatch):
    path = tmp_path / "shard.parquet"
    path.write_bytes(b"verified payload")
    digest = sha256_file(path)

    class FakeApi:
        def __init__(self, token):
            assert token == "token"

        def upload_file(self, **kwargs):
            assert kwargs["path_or_fileobj"] == path
            return SimpleNamespace(oid="c" * 40)

        def dataset_info(self, repository, revision, files_metadata):
            assert revision == "c" * 40
            sibling = SimpleNamespace(
                rfilename="candidate/shard.parquet",
                size=path.stat().st_size,
                lfs=SimpleNamespace(size=path.stat().st_size, sha256=digest),
            )
            return SimpleNamespace(sha=revision, siblings=[sibling])

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi),
    )
    result = upload_verified(
        path, "candidate/shard.parquet", "token", repository="test/repo"
    )
    assert result["sha256"] == digest
    assert result["bytes"] == path.stat().st_size
