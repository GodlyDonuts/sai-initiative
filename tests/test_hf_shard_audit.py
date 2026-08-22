from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.hf_dataset_inventory import build_inventory
from sai.data.hf_shard_audit import HFShardAuditError, audit_shard, audit_to_file
from sai.data.token_stream import sha256_file


def fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    zstandard = pytest.importorskip("zstandard")
    tmp_path.mkdir(parents=True, exist_ok=True)
    member = "data/code-Python/shard_00000.jsonl.zst"
    rows = [
        {
            "id": "a",
            "text": "print('hello')",
            "source": "repo-a",
            "metadata": {
                "license_type": "permissive",
                "int_score": 4,
                "nested_source_metadata": {"key": "value"},
            },
        },
        {
            "id": "a",
            "text": "print('hello')",
            "source": "repo-a",
            "metadata": {"license_type": "permissive", "int_score": 4},
        },
        {
            "id": "b",
            "text": "",
            "source": "repo-b",
            "metadata": {"license_type": "no_license", "int_score": 5},
        },
    ]
    plain = "".join(json.dumps(row) + "\n" for row in rows).encode()
    compressed = tmp_path / "shard.jsonl.zst"
    compressed.write_bytes(zstandard.ZstdCompressor().compress(plain))
    api = {
        "id": "owner/dataset",
        "sha": "a" * 40,
        "private": False,
        "gated": False,
        "siblings": [
            {"rfilename": "README.md", "blobId": "1" * 40, "size": 10},
            {
                "rfilename": member,
                "blobId": "2" * 40,
                "size": compressed.stat().st_size,
                "lfs": {
                    "sha256": sha256_file(compressed),
                    "size": compressed.stat().st_size,
                    "pointerSize": 130,
                },
            },
        ],
    }
    inventory = build_inventory(
        api,
        dataset="owner/dataset",
        revision="a" * 40,
        api_response_sha256="3" * 64,
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    return inventory_path, compressed, member


def test_measures_duplicates_and_per_row_license_metadata(tmp_path: Path) -> None:
    inventory, compressed, member = fixture(tmp_path)
    result = audit_shard(inventory, compressed, member_path=member)
    assert result["population"]["rows"] == 3
    assert result["population"]["unique_document_ids"] == 2
    assert result["population"]["duplicate_document_id_rows"] == 1
    assert result["population"]["empty_text_rows"] == 1
    assert result["population"]["document_id_multiplicity_histogram"] == {
        "1": 1,
        "2": 1,
    }
    assert result["metadata"]["license_type_counts"] == {
        "no_license": 1,
        "permissive": 2,
    }
    assert not result["source_admitted"]
    assert not result["training_authorized"]


def test_compressed_tamper_and_identity_collision_fail(tmp_path: Path) -> None:
    inventory, compressed, member = fixture(tmp_path)
    compressed.write_bytes(compressed.read_bytes() + b"x")
    with pytest.raises(HFShardAuditError, match="bytes differ"):
        audit_shard(inventory, compressed, member_path=member)

    inventory, compressed, member = fixture(tmp_path / "collision")
    zstandard = pytest.importorskip("zstandard")
    bad = (
        json.dumps({"id": "same", "text": "first", "metadata": {}})
        + "\n"
        + json.dumps({"id": "same", "text": "second", "metadata": {}})
        + "\n"
    ).encode()
    compressed.write_bytes(zstandard.ZstdCompressor().compress(bad))
    payload = json.loads(inventory.read_text())
    payload["files"][1]["bytes"] = compressed.stat().st_size
    payload["files"][1]["sha256"] = sha256_file(compressed)
    payload["data_compressed_bytes"] = compressed.stat().st_size
    from sai.data.token_stream import canonical_sha256

    payload["files_sha256"] = canonical_sha256(payload["files"])
    payload["component_partitions"][0]["compressed_bytes"] = compressed.stat().st_size
    payload["component_partitions_sha256"] = canonical_sha256(
        payload["component_partitions"]
    )
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    inventory.write_text(json.dumps(payload))
    with pytest.raises(HFShardAuditError, match="multiple texts"):
        audit_shard(inventory, compressed, member_path=member)


def test_output_is_create_only(tmp_path: Path) -> None:
    inventory, compressed, member = fixture(tmp_path)
    output = tmp_path / "audit.json"
    audit_to_file(inventory, compressed, output, member_path=member)
    with pytest.raises(HFShardAuditError, match="already exists"):
        audit_to_file(inventory, compressed, output, member_path=member)
