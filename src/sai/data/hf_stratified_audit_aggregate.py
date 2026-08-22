"""Aggregate a complete prospective Hugging Face stratified shard audit."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.hf_shard_audit import SCHEMA as SHARD_AUDIT_SCHEMA
from sai.data.hf_stratified_audit_plan import PLAN_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-hf-stratified-audit-aggregate-v1"


class HFStratifiedAuditAggregateError(RuntimeError):
    """The plan or its complete shard-audit population differs."""


def _regular(path: Path, field: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise HFStratifiedAuditAggregateError(f"{field} is missing or unsafe")


def _self_hash(payload: dict[str, Any], field: str) -> None:
    receipt = payload.get("receipt_sha256")
    if (
        not isinstance(receipt, str)
        or len(receipt) != 64
        or receipt
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "receipt_sha256"}
        )
    ):
        raise HFStratifiedAuditAggregateError(f"{field} receipt differs")


def _validate_plan(payload: object) -> dict[str, Any]:
    expected = {
        "schema",
        "status",
        "training_authorized",
        "source_admitted",
        "content_downloaded",
        "dataset_inventory",
        "specification",
        "selected_shards",
        "selected_compressed_bytes",
        "stratum_summaries",
        "selections_sha256",
        "selections",
        "checks",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema") != PLAN_SCHEMA
        or payload.get("status")
        != "prospective_stratified_members_selected_no_download"
        or payload.get("training_authorized") is not False
        or payload.get("source_admitted") is not False
        or payload.get("content_downloaded") is not False
        or not isinstance(payload.get("selections"), list)
        or not payload["selections"]
        or payload.get("selected_shards") != len(payload["selections"])
        or payload.get("selected_compressed_bytes")
        != sum(row.get("compressed_bytes", -1) for row in payload["selections"])
        or payload.get("selections_sha256") != canonical_sha256(payload["selections"])
    ):
        raise HFStratifiedAuditAggregateError("stratified audit plan differs")
    _self_hash(payload, "stratified audit plan")
    paths = []
    for row in payload["selections"]:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "stratum",
                "group",
                "component",
                "path",
                "compressed_bytes",
                "compressed_sha256",
                "selection_rank_sha256",
            }
            or not isinstance(row["stratum"], str)
            or not row["stratum"]
            or not isinstance(row["path"], str)
            or not row["path"].startswith("data/")
            or isinstance(row["compressed_bytes"], bool)
            or not isinstance(row["compressed_bytes"], int)
            or row["compressed_bytes"] <= 0
        ):
            raise HFStratifiedAuditAggregateError("audit selection differs")
        paths.append(row["path"])
    if len(set(paths)) != len(paths):
        raise HFStratifiedAuditAggregateError("audit selection paths differ")
    return payload


def _validate_audit(payload: object) -> dict[str, Any]:
    expected = {
        "schema",
        "status",
        "training_authorized",
        "source_admitted",
        "rows_selected_for_training",
        "dataset_inventory",
        "member",
        "population",
        "metadata",
        "checks",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema") != SHARD_AUDIT_SCHEMA
        or payload.get("status") != "diagnostic_complete_source_not_admitted"
        or payload.get("training_authorized") is not False
        or payload.get("source_admitted") is not False
        or payload.get("rows_selected_for_training") != 0
        or not isinstance(payload.get("member"), dict)
        or not isinstance(payload.get("population"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise HFStratifiedAuditAggregateError("compressed shard audit differs")
    _self_hash(payload, "compressed shard audit")
    population = payload["population"]
    metadata = payload["metadata"]
    for field in (
        "rows",
        "unique_document_ids",
        "unique_texts",
        "duplicate_document_id_rows",
        "duplicate_text_rows",
        "empty_text_rows",
    ):
        value = population.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HFStratifiedAuditAggregateError("audit population differs")
    if (
        population["rows"] <= 0
        or population["unique_document_ids"] + population["duplicate_document_id_rows"]
        != population["rows"]
        or population["unique_texts"] + population["duplicate_text_rows"]
        != population["rows"]
        or not isinstance(metadata.get("license_type_counts"), dict)
        or sum(metadata["license_type_counts"].values()) != population["rows"]
    ):
        raise HFStratifiedAuditAggregateError("audit counts differ")
    return payload


def aggregate_audits(plan_path: Path, audit_paths: list[Path]) -> dict[str, Any]:
    """Validate and aggregate exactly one audit for every selected shard."""

    _regular(plan_path, "stratified audit plan")
    plan_file_sha256 = sha256_file(plan_path)
    try:
        plan = _validate_plan(json.loads(plan_path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFStratifiedAuditAggregateError("audit plan is malformed") from error
    if len(audit_paths) != plan["selected_shards"]:
        raise HFStratifiedAuditAggregateError("audit file count differs")

    selected = {row["path"]: row for row in plan["selections"]}
    seen_paths = set()
    seen_inodes = set()
    audit_files = []
    by_stratum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "shards": 0,
            "physical_rows": 0,
            "within_shard_unique_document_ids_sum": 0,
            "within_shard_unique_texts_sum": 0,
            "within_shard_duplicate_document_id_rows": 0,
            "within_shard_duplicate_text_rows": 0,
            "empty_text_rows": 0,
            "license_type_counts": Counter(),
        }
    )
    for audit_path in sorted(audit_paths, key=lambda path: path.name):
        _regular(audit_path, "compressed shard audit")
        inode = (audit_path.stat().st_dev, audit_path.stat().st_ino)
        if inode in seen_inodes:
            raise HFStratifiedAuditAggregateError("audit file inode is duplicated")
        seen_inodes.add(inode)
        file_sha256 = sha256_file(audit_path)
        try:
            audit = _validate_audit(json.loads(audit_path.read_bytes()))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HFStratifiedAuditAggregateError("shard audit is malformed") from error
        member_path = audit["member"].get("path")
        if member_path not in selected or member_path in seen_paths:
            raise HFStratifiedAuditAggregateError("audit member population differs")
        selection = selected[member_path]
        if (
            audit["member"].get("compressed_bytes") != selection["compressed_bytes"]
            or audit["member"].get("compressed_sha256")
            != selection["compressed_sha256"]
            or audit["dataset_inventory"] != plan["dataset_inventory"]
            or sha256_file(audit_path) != file_sha256
        ):
            raise HFStratifiedAuditAggregateError("audit lineage differs")
        seen_paths.add(member_path)
        population = audit["population"]
        bucket = by_stratum[selection["stratum"]]
        bucket["shards"] += 1
        bucket["physical_rows"] += population["rows"]
        bucket["within_shard_unique_document_ids_sum"] += population[
            "unique_document_ids"
        ]
        bucket["within_shard_unique_texts_sum"] += population["unique_texts"]
        bucket["within_shard_duplicate_document_id_rows"] += population[
            "duplicate_document_id_rows"
        ]
        bucket["within_shard_duplicate_text_rows"] += population["duplicate_text_rows"]
        bucket["empty_text_rows"] += population["empty_text_rows"]
        bucket["license_type_counts"].update(audit["metadata"]["license_type_counts"])
        audit_files.append(
            {
                "member_path": member_path,
                "stratum": selection["stratum"],
                "file_sha256": file_sha256,
                "receipt_sha256": audit["receipt_sha256"],
            }
        )
    if seen_paths != set(selected):
        raise HFStratifiedAuditAggregateError("audit member population is incomplete")

    strata = []
    for stratum, bucket in sorted(by_stratum.items()):
        bucket["license_type_counts"] = dict(
            sorted(bucket["license_type_counts"].items())
        )
        bucket["within_shard_duplicate_document_id_fraction"] = (
            bucket["within_shard_duplicate_document_id_rows"] / bucket["physical_rows"]
        )
        strata.append({"stratum": stratum, **bucket})
    totals = {
        key: sum(row[key] for row in strata)
        for key in (
            "shards",
            "physical_rows",
            "within_shard_unique_document_ids_sum",
            "within_shard_unique_texts_sum",
            "within_shard_duplicate_document_id_rows",
            "within_shard_duplicate_text_rows",
            "empty_text_rows",
        )
    }
    total_licenses: Counter[str] = Counter()
    for row in strata:
        total_licenses.update(row["license_type_counts"])
    totals["license_type_counts"] = dict(sorted(total_licenses.items()))
    totals["within_shard_duplicate_document_id_fraction"] = (
        totals["within_shard_duplicate_document_id_rows"] / totals["physical_rows"]
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_source_not_admitted",
        "training_authorized": False,
        "source_admitted": False,
        "rows_selected_for_training": 0,
        "plan": {
            "file_sha256": plan_file_sha256,
            "receipt_sha256": plan["receipt_sha256"],
            "selected_shards": plan["selected_shards"],
            "selections_sha256": plan["selections_sha256"],
        },
        "audit_files_sha256": canonical_sha256(audit_files),
        "audit_files": audit_files,
        "strata": strata,
        "totals": totals,
        "checks": {
            "plan_replayed": True,
            "every_selected_member_audited_once": True,
            "compressed_member_identities_match": True,
            "audit_receipts_replayed": True,
            "within_shard_duplicates_measured": True,
            "cross_shard_duplicate_identity_not_measured": True,
            "license_metadata_reported_not_inferred": True,
            "no_source_admission_or_training": True,
        },
    }
    if sha256_file(plan_path) != plan_file_sha256:
        raise HFStratifiedAuditAggregateError("audit plan changed while reading")
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def aggregate_to_file(
    plan_path: Path, audit_directory: Path, output_path: Path
) -> dict[str, Any]:
    if not audit_directory.is_dir() or audit_directory.is_symlink():
        raise HFStratifiedAuditAggregateError("audit directory is missing or unsafe")
    if output_path.exists() or output_path.is_symlink():
        raise HFStratifiedAuditAggregateError("aggregate output already exists")
    audit_paths = sorted(audit_directory.glob("*.json"))
    payload = aggregate_audits(plan_path, audit_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_to_file(args.plan, args.audit_directory, args.output)
    print(
        json.dumps(
            {"receipt_sha256": payload["receipt_sha256"], "status": payload["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
