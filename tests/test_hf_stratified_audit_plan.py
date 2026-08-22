from __future__ import annotations

import copy

import pytest

from sai.data.hf_dataset_inventory import build_inventory
from sai.data.hf_stratified_audit_plan import (
    HFStratifiedAuditPlanError,
    plan_audit,
)
from sai.data.token_stream import canonical_sha256


def inventory() -> dict:
    siblings = [{"rfilename": "README.md", "blobId": "1" * 40, "size": 10}]
    index = 2
    for component in (
        "web-science-0019",
        "web-code-0019",
        "stack-Python",
        "stack-Rust",
    ):
        for shard in range(3):
            size = 100 + index
            siblings.append(
                {
                    "rfilename": f"data/{component}/shard_{shard:05d}.jsonl.zst",
                    "blobId": f"{index:040x}",
                    "size": size,
                    "lfs": {
                        "sha256": f"{index:064x}",
                        "size": size,
                        "pointerSize": 130,
                    },
                }
            )
            index += 1
    return build_inventory(
        {
            "id": "owner/data",
            "sha": "a" * 40,
            "private": False,
            "gated": False,
            "siblings": siblings,
        },
        dataset="owner/data",
        revision="a" * 40,
        api_response_sha256="b" * 64,
    )


def spec() -> dict:
    result = {
        "schema": "sai-hf-stratified-audit-spec-v1",
        "status": "prospective_no_download",
        "training_authorized": False,
        "source_admitted": False,
        "content_download_authorized": False,
        "dataset": "owner/data",
        "revision": "a" * 40,
        "inventory_receipt_sha256": inventory()["receipt_sha256"],
        "selection_seed": "fixed-seed",
        "rules": [
            {
                "name": "web",
                "component_regex": r"^web-(?P<topic>[a-z]+)-0019$",
                "group_keys": ["topic"],
                "samples_per_group": 1,
            },
            {
                "name": "stack",
                "component_regex": r"^stack-(?P<language>[A-Za-z]+)$",
                "group_keys": ["language"],
                "samples_per_group": 1,
            },
        ],
    }
    result["spec_sha256"] = canonical_sha256(result)
    return result


def planned(inventory_payload=None, spec_payload=None):
    return plan_audit(
        inventory_payload or inventory(),
        spec_payload or spec(),
        inventory_file_sha256="c" * 64,
        spec_file_sha256="d" * 64,
    )


def test_selects_one_hash_ranked_member_per_declared_group() -> None:
    result = planned()
    assert result["selected_shards"] == 4
    assert len({row["path"] for row in result["selections"]}) == 4
    assert {row["stratum"] for row in result["selections"]} == {"web", "stack"}
    assert not result["content_downloaded"]
    assert not result["source_admitted"]
    assert not result["training_authorized"]
    assert result["checks"]["hash_ranked_not_size_selected"]
    assert result == planned()


def test_resigned_spec_overlap_and_empty_rule_fail_closed() -> None:
    changed = spec()
    changed["selection_seed"] = "observed-after-looking"
    with pytest.raises(HFStratifiedAuditPlanError, match="specification differs"):
        planned(spec_payload=changed)

    overlap = spec()
    overlap["rules"].append(
        {
            "name": "all_web",
            "component_regex": r"^web-(?P<topic>[a-z]+)-0019$",
            "group_keys": ["topic"],
            "samples_per_group": 1,
        }
    )
    overlap["spec_sha256"] = canonical_sha256(
        {key: value for key, value in overlap.items() if key != "spec_sha256"}
    )
    with pytest.raises(HFStratifiedAuditPlanError, match="multiple audit rules"):
        planned(spec_payload=overlap)

    empty = copy.deepcopy(spec())
    empty["rules"][0]["component_regex"] = r"^books-(?P<topic>[a-z]+)$"
    empty["spec_sha256"] = canonical_sha256(
        {key: value for key, value in empty.items() if key != "spec_sha256"}
    )
    with pytest.raises(HFStratifiedAuditPlanError, match="matched no components"):
        planned(spec_payload=empty)
