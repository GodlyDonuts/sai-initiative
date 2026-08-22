from __future__ import annotations

import copy
import json

import pytest

from sai.data.hf_dataset_inventory import (
    HFDatasetInventoryError,
    build_inventory,
    inventory_bytes,
    validate_inventory,
)

DATASET = "owner/dataset"
REVISION = "a" * 40


def response() -> dict:
    return {
        "id": DATASET,
        "sha": REVISION,
        "private": False,
        "gated": False,
        "siblings": [
            {"rfilename": "README.md", "blobId": "1" * 40, "size": 10},
            {
                "rfilename": "data/web-0001/shard_00000.jsonl.zst",
                "blobId": "2" * 40,
                "size": 100,
                "lfs": {"sha256": "3" * 64, "size": 100, "pointerSize": 130},
            },
            {
                "rfilename": "data/code-Python/shard_00000.jsonl.zst",
                "blobId": "4" * 40,
                "size": 200,
                "lfs": {"sha256": "5" * 64, "size": 200, "pointerSize": 130},
            },
        ],
    }


def inventory(payload: dict | None = None) -> dict:
    return build_inventory(
        payload or response(),
        dataset=DATASET,
        revision=REVISION,
        api_response_sha256="6" * 64,
    )


def test_freezes_exact_members_and_component_partitions_without_content() -> None:
    result = inventory()
    assert result["status"].endswith("content_not_acquired")
    assert not result["content_downloaded"]
    assert not result["source_admitted"]
    assert not result["training_authorized"]
    assert result["file_count"] == 3
    assert result["data_file_count"] == 2
    assert result["data_compressed_bytes"] == 300
    assert result["component_partition_count"] == 2
    assert [row["component"] for row in result["component_partitions"]] == [
        "code-Python",
        "web-0001",
    ]
    assert len(result["receipt_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    ["revision", "duplicate", "unsafe", "missing_lfs", "wrong_size", "wrong_ext"],
)
def test_identity_path_and_lfs_tamper_fail_closed(mutation: str) -> None:
    payload = copy.deepcopy(response())
    if mutation == "revision":
        payload["sha"] = "b" * 40
    elif mutation == "duplicate":
        payload["siblings"].append(copy.deepcopy(payload["siblings"][1]))
    elif mutation == "unsafe":
        payload["siblings"][1]["rfilename"] = "data/../secret.jsonl.zst"
    elif mutation == "missing_lfs":
        del payload["siblings"][1]["lfs"]
    elif mutation == "wrong_size":
        payload["siblings"][1]["lfs"]["size"] += 1
    else:
        payload["siblings"][1]["rfilename"] += ".tmp"
    with pytest.raises(HFDatasetInventoryError):
        inventory(payload)


def test_raw_response_hash_is_bound() -> None:
    first = json.dumps(response(), sort_keys=True).encode()
    second = json.dumps(response(), indent=2).encode()
    a = inventory_bytes(first, dataset=DATASET, revision=REVISION)
    b = inventory_bytes(second, dataset=DATASET, revision=REVISION)
    assert a["files_sha256"] == b["files_sha256"]
    assert a["api_response_sha256"] != b["api_response_sha256"]
    assert a["receipt_sha256"] != b["receipt_sha256"]


@pytest.mark.parametrize("mutation", ["member", "count", "component", "receipt"])
def test_generated_inventory_tamper_fails_replay(mutation: str) -> None:
    payload = inventory()
    if mutation == "member":
        payload["files"][1]["bytes"] += 1
    elif mutation == "count":
        payload["data_file_count"] += 1
    elif mutation == "component":
        payload["component_partitions"][0]["compressed_bytes"] += 1
    else:
        payload["receipt_sha256"] = "f" * 64
    with pytest.raises(HFDatasetInventoryError):
        validate_inventory(payload)
