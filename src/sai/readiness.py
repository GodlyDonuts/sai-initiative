"""Validate the provisional Qwen replay-control rehearsal without authorizing Sai."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA = "sai-4b-pretraining-readiness-v1"
BENCHMARK_ROWS = {
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "ifeval": 541,
    "musr": 756,
    "correctbench": 739,
}
DATA_ROLES = {
    "skill_direct",
    "skill_deliberate",
    "behavior_replay",
    "rl_prompts",
}
REQUIRED_CHECKS = {
    "parent_manifest_verified",
    "data_decontamination_verified",
    "replay_disjoint_from_benchmarks",
    "tokenizer_roundtrip_verified",
    "evaluator_official_scoring_verified",
    "candidate_control_compute_matched",
    "runtime_tests_passed",
    "zero_gpu_jobs_submitted",
    "no_training_performed",
}
ALLOWED_CONFIG_DIFFERENCES = {"role", "replay_weight", "output"}


class ReadinessError(RuntimeError):
    """Sai preparation is incomplete, mismatched, or unsafe to authorize."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReadinessError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ReadinessError(f"{field} differs") from error
    return value


def validate_artifact(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ReadinessError("artifact receipt must be an object")
    path_value = receipt.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ReadinessError("artifact path differs")
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        raise ReadinessError(f"artifact is missing or unsafe: {path}")
    expected_hash = _hex_sha256(receipt.get("sha256"), "artifact sha256")
    size = receipt.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ReadinessError("artifact byte count differs")
    if path.stat().st_size != size or sha256_file(path) != expected_hash:
        raise ReadinessError(f"artifact content differs: {path}")
    return {"path": str(path.resolve()), "sha256": expected_hash, "bytes": size}


def _validate_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != DATA_ROLES:
        raise ReadinessError("exact Sai data roles are required")
    validated = {}
    identities: set[str] = set()
    for role, receipt in data.items():
        artifact = validate_artifact(receipt)
        rows = receipt.get("rows")
        identity = _hex_sha256(receipt.get("identity_sha256"), f"{role} identity")
        contamination = _hex_sha256(
            receipt.get("contamination_report_sha256"),
            f"{role} contamination report",
        )
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ReadinessError(f"{role} row count differs")
        if identity in identities:
            raise ReadinessError("data role identities must be distinct")
        identities.add(identity)
        validated[role] = {
            **artifact,
            "rows": rows,
            "identity_sha256": identity,
            "contamination_report_sha256": contamination,
        }
    return validated


def _validate_benchmarks(benchmarks: Any) -> dict[str, Any]:
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(BENCHMARK_ROWS):
        raise ReadinessError("exact five public benchmarks are required")
    validated = {}
    for name, expected_rows in BENCHMARK_ROWS.items():
        receipt = benchmarks[name]
        artifact = validate_artifact(receipt)
        if (
            receipt.get("official_scoring") is not True
            or receipt.get("rows") != expected_rows
            or not isinstance(receipt.get("version"), str)
            or not receipt["version"]
        ):
            raise ReadinessError(f"{name} official benchmark contract differs")
        validated[name] = {
            **artifact,
            "rows": expected_rows,
            "version": receipt["version"],
            "official_scoring": True,
        }
    return validated


def _validate_matched_training(training: Any) -> dict[str, Any]:
    if not isinstance(training, dict):
        raise ReadinessError("training contract must be an object")
    candidate = training.get("candidate")
    control = training.get("equal_compute_control")
    if not isinstance(candidate, dict) or not isinstance(control, dict):
        raise ReadinessError("candidate and equal-compute configs are required")
    if set(candidate) != set(control):
        raise ReadinessError("candidate/control config fields differ")
    mismatches = {key for key in candidate if candidate[key] != control[key]}
    if mismatches != ALLOWED_CONFIG_DIFFERENCES:
        raise ReadinessError(
            f"candidate/control differences are not exact: {sorted(mismatches)}"
        )
    if (
        candidate.get("role") != "sai_candidate"
        or control.get("role") != "equal_compute_control"
        or isinstance(candidate.get("replay_weight"), bool)
        or not isinstance(candidate.get("replay_weight"), (int, float))
        or not math.isfinite(float(candidate["replay_weight"]))
        or candidate["replay_weight"] <= 0
        or control.get("replay_weight") != 0
        or candidate.get("output") == control.get("output")
    ):
        raise ReadinessError("matched replay roles or weights differ")
    return {"candidate": candidate, "equal_compute_control": control}


def validate(payload: Any) -> dict[str, Any]:
    """Return a rehearsal receipt while retaining the explicit 4B training hold.

    This legacy manifest exercises Qwen-parent replay matching.  It cannot
    establish final Sai pretraining readiness because it does not bind the
    evidence-selected 300M/1B architecture, tokenizer, or base-pretraining
    stream.
    """

    if not isinstance(payload, dict):
        raise ReadinessError("readiness manifest must be an object")
    if payload.get("schema") != SCHEMA or payload.get("status") != "prepared":
        raise ReadinessError("readiness schema/status differs")
    if (
        payload.get("training_hold") is not True
        or payload.get("official_training_order_received") is not False
        or payload.get("gpu_jobs_submitted") != 0
        or payload.get("training_updates_completed") != 0
    ):
        raise ReadinessError("training hold or zero-execution evidence differs")

    parent = payload.get("parent")
    if not isinstance(parent, dict):
        raise ReadinessError("parent contract is missing")
    if (
        parent.get("model_id") != "Qwen/Qwen3.5-4B"
        or parent.get("revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or parent.get("license") != "Apache-2.0"
    ):
        raise ReadinessError("provisional parent identity differs")
    parent_manifest = validate_artifact(parent.get("manifest"))
    runtime = validate_artifact(payload.get("runtime_manifest"))
    environment = validate_artifact(payload.get("environment_receipt"))
    tokenizer = validate_artifact(payload.get("tokenizer_audit"))
    data = _validate_data(payload.get("data"))
    benchmarks = _validate_benchmarks(payload.get("benchmarks"))
    training = _validate_matched_training(payload.get("training"))

    checks = payload.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != REQUIRED_CHECKS
        or any(checks[key] is not True for key in REQUIRED_CHECKS)
    ):
        raise ReadinessError("pretraining readiness checks are incomplete")
    resources = payload.get("resource_plan")
    if (
        not isinstance(resources, dict)
        or resources.get("independent_single_h100_jobs") is not True
        or isinstance(resources.get("estimated_h100_hours"), bool)
        or not isinstance(resources.get("estimated_h100_hours"), (int, float))
        or not 0 < float(resources["estimated_h100_hours"]) < 1000
    ):
        raise ReadinessError("resource plan differs")

    return {
        "schema": "sai-4b-qwen-replay-rehearsal-receipt-v1",
        "status": "provisional_qwen_replay_rehearsal_complete",
        "sai_4b_pretraining_ready": False,
        "selected_300m_1b_architecture_receipts_required": True,
        "selected_tokenizer_and_base_stream_receipts_required": True,
        "training_authorized": False,
        "official_training_order_required": True,
        "parent_manifest": parent_manifest,
        "runtime_manifest": runtime,
        "environment_receipt": environment,
        "tokenizer_audit": tokenizer,
        "data": data,
        "benchmarks": benchmarks,
        "training": training,
        "resource_plan": resources,
        "checks": checks,
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    return validate(json.loads(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ReadinessError("readiness receipt already exists")
    receipt = load_and_validate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": receipt["status"]}, sort_keys=True))
    return 0
