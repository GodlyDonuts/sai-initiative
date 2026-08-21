"""Sai's benchmark-first promotion gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "sai-4b-public-benchmark-score-v1"
SCHEMA = "sai-4b-public-benchmark-gate-v1"
BENCHMARKS = ("humaneval_plus", "mbpp_plus", "ifeval", "musr", "correctbench")
SHA256_KEYS = (
    "benchmark_source_sha256",
    "identity_order_sha256",
    "prompt_contract_sha256",
    "decoding_contract_sha256",
    "original_checkpoint_sha256",
    "equal_compute_checkpoint_sha256",
    "candidate_checkpoint_sha256",
)


class PublicGateError(RuntimeError):
    """A score report or matched-comparison binding is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicGateError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise PublicGateError(f"{field} is outside [0, 100]")
    return result


def validate_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise PublicGateError("score report must be an object")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise PublicGateError("score report schema/status differs")
    if report.get("benchmark") not in BENCHMARKS:
        raise PublicGateError("benchmark identity differs")
    rows = report.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise PublicGateError("benchmark row count differs")
    if (
        not isinstance(report.get("benchmark_version"), str)
        or not report["benchmark_version"]
    ):
        raise PublicGateError("benchmark version is missing")
    for key in SHA256_KEYS:
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise PublicGateError(f"{key} differs")
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise PublicGateError(f"{key} differs") from error
    return {
        **report,
        "original_score": score(report.get("original_score"), "original_score"),
        "equal_compute_score": score(
            report.get("equal_compute_score"), "equal_compute_score"
        ),
        "candidate_score": score(report.get("candidate_score"), "candidate_score"),
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    """Validate five reports and return the conjunctive promotion decision."""

    if len(paths) != len(BENCHMARKS):
        raise PublicGateError("exactly five score reports are required")
    reports = [validate_report(json.loads(path.read_text())) for path in paths]
    if {report["benchmark"] for report in reports} != set(BENCHMARKS):
        raise PublicGateError("one report per required benchmark is required")
    first = reports[0]
    checkpoints = (
        "original_checkpoint_sha256",
        "equal_compute_checkpoint_sha256",
        "candidate_checkpoint_sha256",
    )
    if any(report[key] != first[key] for report in reports for key in checkpoints):
        raise PublicGateError("cross-benchmark checkpoint binding differs")

    ordered = sorted(reports, key=lambda report: BENCHMARKS.index(report["benchmark"]))
    benchmarks: dict[str, dict[str, Any]] = {}
    for report in ordered:
        candidate = report["candidate_score"]
        original = report["original_score"]
        control = report["equal_compute_score"]
        benchmarks[report["benchmark"]] = {
            "rows": report["rows"],
            "benchmark_version": report["benchmark_version"],
            "original_score": original,
            "equal_compute_score": control,
            "candidate_score": candidate,
            "candidate_vs_original_points": candidate - original,
            "candidate_vs_equal_compute_points": candidate - control,
        }
    macro = {
        name: sum(item[name] for item in benchmarks.values()) / len(BENCHMARKS)
        for name in ("original_score", "equal_compute_score", "candidate_score")
    }
    macro.update(
        {
            "candidate_vs_original_points": macro["candidate_score"]
            - macro["original_score"],
            "candidate_vs_equal_compute_points": macro["candidate_score"]
            - macro["equal_compute_score"],
        }
    )
    checks = {
        "macro_beats_original_by_at_least_1_point": macro["candidate_score"]
        >= macro["original_score"] + 1.0,
        "macro_beats_equal_compute_by_at_least_1_point": macro["candidate_score"]
        >= macro["equal_compute_score"] + 1.0,
        "no_benchmark_regresses_over_1_point_vs_original": all(
            item["candidate_vs_original_points"] >= -1.0 for item in benchmarks.values()
        ),
        "no_benchmark_regresses_over_1_point_vs_equal_compute": all(
            item["candidate_vs_equal_compute_points"] >= -1.0
            for item in benchmarks.values()
        ),
        "beats_original_on_at_least_four_benchmarks": sum(
            item["candidate_vs_original_points"] > 0 for item in benchmarks.values()
        )
        >= 4,
        "beats_equal_compute_on_at_least_four_benchmarks": sum(
            item["candidate_vs_equal_compute_points"] > 0
            for item in benchmarks.values()
        )
        >= 4,
        "musr_nonnegative_vs_both": min(
            benchmarks["musr"]["candidate_vs_original_points"],
            benchmarks["musr"]["candidate_vs_equal_compute_points"],
        )
        >= 0,
        "correctbench_nonnegative_vs_both": min(
            benchmarks["correctbench"]["candidate_vs_original_points"],
            benchmarks["correctbench"]["candidate_vs_equal_compute_points"],
        )
        >= 0,
    }
    promote = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": "promote_sai_candidate" if promote else "reject_sai_candidate",
        "architecture_locked": False,
        "promote_to_full_confirmation": promote,
        "stop_candidate": not promote,
        "reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
        "checkpoints": {key: first[key] for key in checkpoints},
        "macro": macro,
        "benchmarks": benchmarks,
        "checks": checks,
    }


def write_analysis(paths: list[Path], output: Path) -> dict[str, Any]:
    payload = analyze(paths)
    if output.exists():
        raise PublicGateError("gate output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return payload
