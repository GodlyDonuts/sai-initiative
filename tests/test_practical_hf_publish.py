import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data import practical_hf_publish as publish
from sai.data.pleias_practical_admission import SCHEMA as ADMISSION_SCHEMA
from sai.data.pleias_practical_locator_scan import LOCATOR_SCHEMA, _schema
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _admission(root: Path) -> tuple[Path, dict]:
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    row = {
        "schema": LOCATOR_SCHEMA,
        "source_id": "pleias_common_corpus",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": "data/part.parquet",
        "source_parent_sha256": "1" * 64,
        "source_row_index": 1,
        "source_row_identity_sha256": "2" * 64,
        "identifier": "doc-1",
        "collection": "books",
        "open_type": "open",
        "license": "public domain",
        "language": "English",
        "word_count": 100,
        "source_token_count": 150,
        "text_utf8_bytes": 600,
        "content_sha256": "3" * 64,
    }
    path = shard / "locators.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=_schema()), path)
    descriptor = {
        "shard_index": 0,
        "path": str(path.relative_to(root)),
        "rows": 1,
        "text_utf8_bytes": 600,
        "source_token_count": 150,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    receipt = _signed(
        {
            "schema": ADMISSION_SCHEMA,
            "status": "complete_practical_pleias_pretraining_admission",
            "counts": {
                "admitted_rows": 1,
                "admitted_text_utf8_bytes": 600,
            },
            "source": {
                "scan_logical_shards": 1,
                "quarantine_registry": {
                    "receipt_sha256": "a" * 64,
                    "registry_sha256": "b" * 64,
                    "rows": 1_548,
                    "unique_content_hashes": 1_548,
                }
            },
            "policy": {
                "byte_cap_selection_policy": "canonical_content_sha256_order",
                "output_partition_policy": "canonical_source_path_sha256_modulo",
            },
            "outputs": {"descriptors": [descriptor]},
            "global_exact_content_deduplication_complete": True,
            "known_quarantine_exclusions_applied": True,
            "complete_source_identity_partition_coverage": True,
            "source_text_copied": False,
            "practical_pretraining_ready": True,
            "training_ready": True,
        }
    )
    (root / "receipt.json").write_text(json.dumps(receipt))
    return path, receipt


def test_publish_shard_uploads_only_text_free_locator(
    tmp_path: Path, monkeypatch
) -> None:
    admission = tmp_path / "admission"
    path, receipt = _admission(admission)

    def fake_upload(local, remote, token, repository):
        assert local == path
        assert token == "token"
        assert repository == publish.DESTINATION_REPOSITORY
        return {
            "repository": repository,
            "commit": "4" * 40,
            "path": remote,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    monkeypatch.setattr(publish, "upload_verified", fake_upload)
    result = publish.publish_shard(admission, tmp_path / "published", 0, 1, "token")
    assert result["status"] == "complete_practical_hf_publish_shard"
    assert result["admission_receipt_sha256"] == receipt["receipt_sha256"]
    assert result["source_text_uploaded"] is False
    assert result["remote_output"]["path"].endswith("locators.parquet")


def test_publish_empty_logical_shard_does_not_upload(
    tmp_path: Path, monkeypatch
) -> None:
    admission = tmp_path / "admission"
    _admission(admission)

    def fail_upload(*args, **kwargs):
        raise AssertionError("empty logical shard must not upload")

    monkeypatch.setattr(publish, "upload_verified", fail_upload)
    result = publish.publish_shard(admission, tmp_path / "published", 1, 2, "token")
    assert result["status"] == "complete_practical_hf_publish_empty_shard"
    assert result["remote_output"] is None
