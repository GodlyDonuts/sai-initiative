"""Plan a deterministic stratified content audit from an HF inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sai.data.hf_dataset_inventory import (
    HFDatasetInventoryError,
    validate_inventory,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SPEC_SCHEMA = "sai-hf-stratified-audit-spec-v1"
PLAN_SCHEMA = "sai-hf-stratified-audit-plan-v1"


class HFStratifiedAuditPlanError(RuntimeError):
    """The inventory, audit specification, or deterministic selection differs."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HFStratifiedAuditPlanError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise HFStratifiedAuditPlanError(f"{field} differs") from error
    return value


def _git_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise HFStratifiedAuditPlanError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise HFStratifiedAuditPlanError(f"{field} differs") from error
    return value


def validate_spec(payload: object) -> dict[str, Any]:
    expected = {
        "schema",
        "status",
        "training_authorized",
        "source_admitted",
        "content_download_authorized",
        "dataset",
        "revision",
        "inventory_receipt_sha256",
        "selection_seed",
        "rules",
        "spec_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema") != SPEC_SCHEMA
        or payload.get("status") != "prospective_no_download"
        or payload.get("training_authorized") is not False
        or payload.get("source_admitted") is not False
        or payload.get("content_download_authorized") is not False
        or not isinstance(payload.get("dataset"), str)
        or payload["dataset"].count("/") != 1
        or not isinstance(payload.get("selection_seed"), str)
        or not payload["selection_seed"]
        or payload.get("spec_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "spec_sha256"}
        )
    ):
        raise HFStratifiedAuditPlanError("stratified audit specification differs")
    _git_sha(payload["revision"], "dataset revision")
    _sha256(payload["inventory_receipt_sha256"], "inventory receipt")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not rules:
        raise HFStratifiedAuditPlanError("stratified audit rules differ")
    names = set()
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule)
            != {"name", "component_regex", "group_keys", "samples_per_group"}
            or not isinstance(rule["name"], str)
            or not rule["name"]
            or rule["name"] in names
            or not isinstance(rule["component_regex"], str)
            or not isinstance(rule["group_keys"], list)
            or not rule["group_keys"]
            or len(set(rule["group_keys"])) != len(rule["group_keys"])
            or any(not isinstance(key, str) or not key for key in rule["group_keys"])
            or isinstance(rule["samples_per_group"], bool)
            or not isinstance(rule["samples_per_group"], int)
            or rule["samples_per_group"] <= 0
        ):
            raise HFStratifiedAuditPlanError("stratified audit rule differs")
        names.add(rule["name"])
        try:
            expression = re.compile(rule["component_regex"])
        except re.error as error:
            raise HFStratifiedAuditPlanError("component regex differs") from error
        if set(rule["group_keys"]) - set(expression.groupindex):
            raise HFStratifiedAuditPlanError("rule group keys differ")
    return payload


def plan_audit(
    inventory: dict[str, Any],
    spec: dict[str, Any],
    *,
    inventory_file_sha256: str,
    spec_file_sha256: str,
) -> dict[str, Any]:
    try:
        inventory = validate_inventory(inventory)
    except HFDatasetInventoryError as error:
        raise HFStratifiedAuditPlanError("dataset inventory differs") from error
    spec = validate_spec(spec)
    inventory_file_sha256 = _sha256(inventory_file_sha256, "inventory file")
    spec_file_sha256 = _sha256(spec_file_sha256, "specification file")
    if (
        spec["dataset"] != inventory["dataset"]
        or spec["revision"] != inventory["revision"]
        or spec["inventory_receipt_sha256"] != inventory["receipt_sha256"]
    ):
        raise HFStratifiedAuditPlanError("inventory/specification identity differs")
    files_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory["files"]:
        if row["path"].startswith("data/"):
            component = row["path"].split("/", 2)[1]
            files_by_component[component].append(row)

    matched_components = set()
    selected_paths = set()
    selections = []
    summaries = []
    for rule in spec["rules"]:
        expression = re.compile(rule["component_regex"])
        groups: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(
            list
        )
        rule_components = set()
        for component, rows in files_by_component.items():
            match = expression.fullmatch(component)
            if match is None:
                continue
            if component in matched_components:
                raise HFStratifiedAuditPlanError(
                    "one component matches multiple audit rules"
                )
            matched_components.add(component)
            rule_components.add(component)
            group = tuple(match.group(key) for key in rule["group_keys"])
            for row in rows:
                rank = hashlib.sha256(
                    "\0".join(
                        (
                            spec["selection_seed"],
                            rule["name"],
                            *group,
                            row["path"],
                            row["sha256"],
                        )
                    ).encode()
                ).hexdigest()
                groups[group].append((rank, row))
        if not groups:
            raise HFStratifiedAuditPlanError(
                f"audit rule matched no components: {rule['name']}"
            )
        rule_selected = []
        for group, candidates in sorted(groups.items()):
            candidates.sort(key=lambda item: (item[0], item[1]["path"]))
            if len(candidates) < rule["samples_per_group"]:
                raise HFStratifiedAuditPlanError(
                    f"audit group has too few shards: {rule['name']}"
                )
            group_values = dict(zip(rule["group_keys"], group, strict=True))
            for rank, row in candidates[: rule["samples_per_group"]]:
                if row["path"] in selected_paths:
                    raise HFStratifiedAuditPlanError("audit member selected twice")
                selected_paths.add(row["path"])
                selected = {
                    "stratum": rule["name"],
                    "group": group_values,
                    "component": row["path"].split("/", 2)[1],
                    "path": row["path"],
                    "compressed_bytes": row["bytes"],
                    "compressed_sha256": row["sha256"],
                    "selection_rank_sha256": rank,
                }
                selections.append(selected)
                rule_selected.append(selected)
        summaries.append(
            {
                "stratum": rule["name"],
                "matched_component_partitions": len(rule_components),
                "groups": len(groups),
                "selected_shards": len(rule_selected),
                "selected_compressed_bytes": sum(
                    row["compressed_bytes"] for row in rule_selected
                ),
            }
        )
    selections.sort(
        key=lambda row: (
            row["stratum"],
            tuple(sorted(row["group"].items())),
            row["selection_rank_sha256"],
        )
    )
    result: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "prospective_stratified_members_selected_no_download",
        "training_authorized": False,
        "source_admitted": False,
        "content_downloaded": False,
        "dataset_inventory": {
            "file_sha256": inventory_file_sha256,
            "receipt_sha256": inventory["receipt_sha256"],
            "dataset": inventory["dataset"],
            "revision": inventory["revision"],
        },
        "specification": {
            "file_sha256": spec_file_sha256,
            "spec_sha256": spec["spec_sha256"],
            "selection_seed": spec["selection_seed"],
        },
        "selected_shards": len(selections),
        "selected_compressed_bytes": sum(row["compressed_bytes"] for row in selections),
        "stratum_summaries": summaries,
        "selections_sha256": canonical_sha256(selections),
        "selections": selections,
        "checks": {
            "prospective_specification_replayed": True,
            "hash_ranked_not_size_selected": True,
            "groups_complete": True,
            "members_unique": True,
            "content_not_downloaded": True,
            "no_source_admission_or_training": True,
        },
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def plan_to_file(inventory_path: Path, spec_path: Path, output_path: Path) -> dict:
    for path, field in (
        (inventory_path, "dataset inventory"),
        (spec_path, "audit specification"),
    ):
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise HFStratifiedAuditPlanError(f"{field} is missing or unsafe")
    if output_path.exists() or output_path.is_symlink():
        raise HFStratifiedAuditPlanError("stratified audit plan already exists")
    inventory_hash = sha256_file(inventory_path)
    spec_hash = sha256_file(spec_path)
    try:
        inventory = json.loads(inventory_path.read_bytes())
        spec = json.loads(spec_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFStratifiedAuditPlanError("audit planning input is malformed") from error
    result = plan_audit(
        inventory,
        spec,
        inventory_file_sha256=inventory_hash,
        spec_file_sha256=spec_hash,
    )
    if (
        sha256_file(inventory_path) != inventory_hash
        or sha256_file(spec_path) != spec_hash
    ):
        raise HFStratifiedAuditPlanError("audit planning input changed while reading")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan_to_file(args.inventory, args.specification, args.output)
    print(
        json.dumps(
            {"receipt_sha256": result["receipt_sha256"], "status": result["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
