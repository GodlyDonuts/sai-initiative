"""Analyze paired fast/slow/control rows without training or routing deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from sai.training.lineage import (
    CompletedRunLineageError,
    load_and_validate_receipt,
)

MANIFEST_SCHEMA = "sai-slow-path-row-manifest-v2"
SCHEMA = "sai-oracle-slow-path-evaluation-v1"
MODES = ("forced_fast", "forced_slow", "equal_flop_fast_control")
GATE_SLOTS = (
    "code_primary",
    "code_secondary",
    "instruction_following",
    "multi_step_reasoning",
    "self_correction",
)
SHARED_HASHES = (
    "benchmark_source_sha256",
    "identity_order_sha256",
    "prompt_contract_sha256",
    "decoding_contract_sha256",
    "official_scorer_sha256",
    "environment_sha256",
)
LINEAGE_HASHES = (
    "system_checkpoint_tree_sha256",
    "fast_path_state_sha256",
    "system_config_sha256",
    "completed_run_lineage_sha256",
    "comparison_group_sha256",
)


class OracleEvaluationError(RuntimeError):
    """A row manifest, pairing, FLOP match, or receipt differs."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise OracleEvaluationError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise OracleEvaluationError(f"{field} differs") from error
    return value


def _finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleEvaluationError(f"{field} differs")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise OracleEvaluationError(f"{field} differs")
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OracleEvaluationError(f"{field} differs")
    return value


def _validate_row(row: Any, mode: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise OracleEvaluationError("row must be an object")
    if row.get("infrastructure_status") != "complete":
        raise OracleEvaluationError("row infrastructure is incomplete")
    identity = _sha256(row.get("row_identity_sha256"), "row identity")
    prompt = _sha256(row.get("prompt_sha256"), "row prompt")
    output = _sha256(row.get("output_sha256"), "row output")
    score = _finite(row.get("official_score"), "official row score", minimum=0.0)
    if score > 1.0:
        raise OracleEvaluationError("official row score differs")
    weight = _finite(row.get("score_weight"), "row score weight", minimum=0.0)
    if weight <= 0:
        raise OracleEvaluationError("row score weight differs")
    modeled_flops = _positive_integer(
        row.get("modeled_inference_flops"), "modeled inference FLOPs"
    )
    executed_flops = _positive_integer(
        row.get("executed_inference_flops"), "executed inference FLOPs"
    )
    output_tokens = row.get("output_tokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens < 0
    ):
        raise OracleEvaluationError("row output-token count differs")
    diagnostics = row.get("workspace_diagnostics")
    if mode == "forced_slow":
        if not isinstance(diagnostics, dict):
            raise OracleEvaluationError("slow row workspace diagnostics are missing")
        normalized_diagnostics = {
            "iterations": _positive_integer(
                diagnostics.get("iterations"), "workspace iterations"
            ),
            "workspace_plan_sha256": _sha256(
                diagnostics.get("workspace_plan_sha256"), "workspace plan"
            ),
            "workspace_candidate_identity_sha256": _sha256(
                diagnostics.get("workspace_candidate_identity_sha256"),
                "workspace candidate identity",
            ),
            "last_update_rms": _finite(
                diagnostics.get("last_update_rms"),
                "workspace last-update RMS",
                minimum=0.0,
            ),
            "output_delta_rms": _finite(
                diagnostics.get("output_delta_rms"),
                "workspace output-delta RMS",
                minimum=0.0,
            ),
        }
    elif diagnostics is not None:
        raise OracleEvaluationError("non-slow row has workspace diagnostics")
    else:
        normalized_diagnostics = None
    return {
        "row_identity_sha256": identity,
        "prompt_sha256": prompt,
        "output_sha256": output,
        "official_score": score,
        "score_weight": weight,
        "modeled_inference_flops": modeled_flops,
        "executed_inference_flops": executed_flops,
        "output_tokens": output_tokens,
        "infrastructure_status": "complete",
        "workspace_diagnostics": normalized_diagnostics,
    }


def validate_manifest(payload: Any, expected_mode: str) -> dict[str, Any]:
    if expected_mode not in MODES or not isinstance(payload, dict):
        raise OracleEvaluationError("manifest mode or object differs")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if (
        payload.get("schema") != MANIFEST_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("mode") != expected_mode
        or payload.get("gate_slot") not in GATE_SLOTS
        or payload.get("source_disjoint") is not True
        or payload.get("terminal_public_board_accessed") is not False
        or payload.get("training_authorized") is not False
        or payload.get("manifest_sha256") != canonical_sha256(unsigned)
    ):
        raise OracleEvaluationError("manifest identity or boundary differs")
    if (
        not isinstance(payload.get("benchmark_name"), str)
        or not payload["benchmark_name"]
        or not isinstance(payload.get("benchmark_version"), str)
        or not payload["benchmark_version"]
    ):
        raise OracleEvaluationError("benchmark name/version differs")
    hashes = {
        key: _sha256(payload.get(key), key) for key in (*SHARED_HASHES, *LINEAGE_HASHES)
    }
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise OracleEvaluationError("manifest rows are missing")
    normalized_rows = [_validate_row(row, expected_mode) for row in rows]
    identities = [row["row_identity_sha256"] for row in normalized_rows]
    if len(identities) != len(set(identities)):
        raise OracleEvaluationError("manifest row identities are duplicated")
    if hashes["identity_order_sha256"] != canonical_sha256(identities):
        raise OracleEvaluationError("manifest row identity order differs")
    if payload.get("rows_sha256") != canonical_sha256(normalized_rows):
        raise OracleEvaluationError("manifest row receipt differs")
    if expected_mode == "forced_slow":
        contracts = {
            (
                row["workspace_diagnostics"]["iterations"],
                row["workspace_diagnostics"]["workspace_plan_sha256"],
                row["workspace_diagnostics"]["workspace_candidate_identity_sha256"],
            )
            for row in normalized_rows
        }
        if len(contracts) != 1:
            raise OracleEvaluationError("slow-row workspace contracts differ")
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "mode": expected_mode,
        "gate_slot": payload["gate_slot"],
        "benchmark_name": payload["benchmark_name"],
        "benchmark_version": payload["benchmark_version"],
        **hashes,
        "source_disjoint": True,
        "terminal_public_board_accessed": False,
        "training_authorized": False,
        "rows": normalized_rows,
        "rows_sha256": payload["rows_sha256"],
        "manifest_sha256": payload["manifest_sha256"],
    }


def _load_manifests(paths: list[Path], mode: str) -> dict[str, dict[str, Any]]:
    if len(paths) != len(GATE_SLOTS):
        raise OracleEvaluationError(f"exactly five {mode} manifests are required")
    results = {}
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise OracleEvaluationError(f"manifest is missing or unsafe: {path}")
        try:
            payload = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OracleEvaluationError("manifest is unreadable") from error
        manifest = validate_manifest(payload, mode)
        slot = manifest["gate_slot"]
        if slot in results:
            raise OracleEvaluationError(f"duplicate {mode} gate slot")
        results[slot] = manifest
    if set(results) != set(GATE_SLOTS):
        raise OracleEvaluationError(f"{mode} gate slots are incomplete")
    return results


def _weighted_score(rows: list[dict[str, Any]], field: str) -> float:
    denominator = sum(row["score_weight"] for row in rows)
    if denominator <= 0:
        raise OracleEvaluationError("benchmark score weight is empty")
    return 100.0 * sum(row[field] * row["score_weight"] for row in rows) / denominator


def _pair_slot(
    fast: dict[str, Any], slow: dict[str, Any], control: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for other in (slow, control):
        if (
            other["benchmark_name"] != fast["benchmark_name"]
            or other["benchmark_version"] != fast["benchmark_version"]
            or any(other[key] != fast[key] for key in SHARED_HASHES)
        ):
            raise OracleEvaluationError("cross-mode benchmark binding differs")
    if (
        slow["system_checkpoint_tree_sha256"] != fast["system_checkpoint_tree_sha256"]
        or slow["fast_path_state_sha256"] != fast["fast_path_state_sha256"]
        or slow["system_config_sha256"] != fast["system_config_sha256"]
        or slow["completed_run_lineage_sha256"] != fast["completed_run_lineage_sha256"]
        or slow["comparison_group_sha256"] != fast["comparison_group_sha256"]
    ):
        raise OracleEvaluationError("fast/slow system lineage differs")
    if (
        control["system_checkpoint_tree_sha256"]
        == fast["system_checkpoint_tree_sha256"]
    ):
        raise OracleEvaluationError("equal-FLOP control checkpoint is not independent")
    if len(fast["rows"]) != len(slow["rows"]) or len(fast["rows"]) != len(
        control["rows"]
    ):
        raise OracleEvaluationError("cross-mode row counts differ")

    paired = []
    for fast_row, slow_row, control_row in zip(
        fast["rows"], slow["rows"], control["rows"], strict=True
    ):
        for other in (slow_row, control_row):
            if (
                other["row_identity_sha256"] != fast_row["row_identity_sha256"]
                or other["prompt_sha256"] != fast_row["prompt_sha256"]
                or other["score_weight"] != fast_row["score_weight"]
            ):
                raise OracleEvaluationError("cross-mode row pairing differs")
        if (
            slow_row["modeled_inference_flops"]
            != control_row["modeled_inference_flops"]
            or slow_row["executed_inference_flops"]
            != control_row["executed_inference_flops"]
        ):
            raise OracleEvaluationError("slow/control inference FLOPs differ")
        if (
            fast_row["modeled_inference_flops"] >= slow_row["modeled_inference_flops"]
            or fast_row["executed_inference_flops"]
            >= slow_row["executed_inference_flops"]
        ):
            raise OracleEvaluationError("slow path does not add measured compute")
        route_slow = slow_row["official_score"] > fast_row["official_score"]
        paired.append(
            {
                "row_identity_sha256": fast_row["row_identity_sha256"],
                "score_weight": fast_row["score_weight"],
                "fast_score": fast_row["official_score"],
                "slow_score": slow_row["official_score"],
                "control_score": control_row["official_score"],
                "oracle_score": (
                    slow_row["official_score"]
                    if route_slow
                    else fast_row["official_score"]
                ),
                "mask_control_score": (
                    control_row["official_score"]
                    if route_slow
                    else fast_row["official_score"]
                ),
                "route_slow": route_slow,
                "fast_modeled_flops": fast_row["modeled_inference_flops"],
                "slow_modeled_flops": slow_row["modeled_inference_flops"],
                "fast_executed_flops": fast_row["executed_inference_flops"],
                "slow_executed_flops": slow_row["executed_inference_flops"],
            }
        )

    summary = {
        "benchmark_name": fast["benchmark_name"],
        "benchmark_version": fast["benchmark_version"],
        "rows": len(paired),
        "forced_fast_score": _weighted_score(paired, "fast_score"),
        "forced_slow_score": _weighted_score(paired, "slow_score"),
        "forced_equal_flop_control_score": _weighted_score(paired, "control_score"),
        "oracle_score": _weighted_score(paired, "oracle_score"),
        "mask_matched_control_score": _weighted_score(paired, "mask_control_score"),
        "slow_route_rate": sum(row["route_slow"] for row in paired) / len(paired),
        "forced_fast_modeled_flops": sum(row["fast_modeled_flops"] for row in paired),
        "forced_slow_modeled_flops": sum(row["slow_modeled_flops"] for row in paired),
        "forced_equal_flop_control_modeled_flops": sum(
            row["slow_modeled_flops"] for row in paired
        ),
        "forced_fast_executed_flops": sum(row["fast_executed_flops"] for row in paired),
        "forced_slow_executed_flops": sum(row["slow_executed_flops"] for row in paired),
        "forced_equal_flop_control_executed_flops": sum(
            row["slow_executed_flops"] for row in paired
        ),
        "oracle_modeled_flops": sum(
            (
                row["slow_modeled_flops"]
                if row["route_slow"]
                else row["fast_modeled_flops"]
            )
            for row in paired
        ),
        "mask_matched_control_modeled_flops": sum(
            (
                row["slow_modeled_flops"]
                if row["route_slow"]
                else row["fast_modeled_flops"]
            )
            for row in paired
        ),
        "oracle_executed_flops": sum(
            (
                row["slow_executed_flops"]
                if row["route_slow"]
                else row["fast_executed_flops"]
            )
            for row in paired
        ),
        "mask_matched_control_executed_flops": sum(
            (
                row["slow_executed_flops"]
                if row["route_slow"]
                else row["fast_executed_flops"]
            )
            for row in paired
        ),
    }
    return summary, paired


def _macro(benchmarks: dict[str, dict[str, Any]], field: str) -> float:
    return sum(row[field] for row in benchmarks.values()) / len(benchmarks)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.floor(probability * len(ordered))))
    return ordered[index]


def _paired_bootstrap(
    rows: dict[str, list[dict[str, Any]]], seed: int, replicates: int
) -> dict[str, Any]:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 100
    ):
        raise OracleEvaluationError("bootstrap replicates must be at least 100")
    generator = random.Random(seed)
    versus_fast = []
    versus_control = []
    for _ in range(replicates):
        fast_slot_scores = []
        oracle_slot_scores = []
        control_slot_scores = []
        for slot in GATE_SLOTS:
            source = rows[slot]
            sample = [source[generator.randrange(len(source))] for _ in source]
            fast_slot_scores.append(_weighted_score(sample, "fast_score"))
            oracle_slot_scores.append(_weighted_score(sample, "oracle_score"))
            control_slot_scores.append(_weighted_score(sample, "mask_control_score"))
        fast_macro = sum(fast_slot_scores) / len(GATE_SLOTS)
        oracle_macro = sum(oracle_slot_scores) / len(GATE_SLOTS)
        control_macro = sum(control_slot_scores) / len(GATE_SLOTS)
        versus_fast.append(oracle_macro - fast_macro)
        versus_control.append(oracle_macro - control_macro)
    return {
        "method": "deterministic_benchmark_stratified_paired_percentile_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "oracle_vs_fast_points": {
            "lower_95": _percentile(versus_fast, 0.025),
            "median": _percentile(versus_fast, 0.5),
            "upper_95": _percentile(versus_fast, 0.975),
        },
        "oracle_vs_mask_matched_control_points": {
            "lower_95": _percentile(versus_control, 0.025),
            "median": _percentile(versus_control, 0.5),
            "upper_95": _percentile(versus_control, 0.975),
        },
    }


def _validate_lineages(
    adaptive_path: Path,
    adaptive_root: Path,
    control_path: Path,
    control_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        adaptive = load_and_validate_receipt(adaptive_path, adaptive_root)
        control = load_and_validate_receipt(control_path, control_root)
    except CompletedRunLineageError as error:
        raise OracleEvaluationError("completed-run lineage differs") from error

    def planned_run(receipt: dict[str, Any], root: Path) -> dict[str, Any]:
        path = root / receipt["plan"]["path"]
        try:
            plan = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OracleEvaluationError("completed-run plan is unreadable") from error
        matches = [
            row
            for row in plan.get("runs", [])
            if row.get("run_identity_sha256") == receipt["run_identity_sha256"]
        ]
        if len(matches) != 1:
            raise OracleEvaluationError("completed-run plan identity differs")
        return matches[0]

    adaptive_planned = planned_run(adaptive, adaptive_root)
    control_planned = planned_run(control, control_root)
    if (
        adaptive["role"] != "workspace_treatment"
        or control["role"] != "equal_flop_fast_control"
        or adaptive["receipt_sha256"] == control["receipt_sha256"]
        or adaptive["checkpoint_tree"]["tree_sha256"]
        == control["checkpoint_tree"]["tree_sha256"]
        or adaptive["comparison_group_sha256"] != control["comparison_group_sha256"]
        or adaptive["parent"] != control["parent"]
        or adaptive["execution"]["optimizer_steps"]
        != control["execution"]["optimizer_steps"]
        or adaptive["execution"]["sequences"] != control["execution"]["sequences"]
        or adaptive["execution"]["valid_tokens"] != control["execution"]["valid_tokens"]
        or adaptive["execution"]["admitted_utf8_bytes"]
        != control["execution"]["admitted_utf8_bytes"]
        or adaptive["execution"]["modeled_training_flops"]
        != control["execution"]["modeled_training_flops"]
        or adaptive["immutable_inputs"]["tokenizer_sha256"]
        != control["immutable_inputs"]["tokenizer_sha256"]
        or adaptive["immutable_inputs"]["ordered_stream_sha256"]
        != control["immutable_inputs"]["ordered_stream_sha256"]
        or adaptive_planned["seed"] != control_planned["seed"]
        or adaptive_planned["mixer_family"] != control_planned["mixer_family"]
        or adaptive_planned["contrast"] != control_planned["contrast"]
    ):
        raise OracleEvaluationError("adaptive/control lineage comparison differs")
    artifacts = {
        "adaptive": {
            "path": str(adaptive_path.resolve()),
            "bytes": adaptive_path.stat().st_size,
            "sha256": sha256_file(adaptive_path),
        },
        "control": {
            "path": str(control_path.resolve()),
            "bytes": control_path.stat().st_size,
            "sha256": sha256_file(control_path),
        },
    }
    return adaptive, control, artifacts


def analyze(
    fast_paths: list[Path],
    slow_paths: list[Path],
    control_paths: list[Path],
    *,
    adaptive_lineage: Path,
    adaptive_artifact_root: Path,
    control_lineage: Path,
    control_artifact_root: Path,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    adaptive_run, control_run, lineage_artifacts = _validate_lineages(
        adaptive_lineage,
        adaptive_artifact_root,
        control_lineage,
        control_artifact_root,
    )
    fast = _load_manifests(fast_paths, "forced_fast")
    slow = _load_manifests(slow_paths, "forced_slow")
    control = _load_manifests(control_paths, "equal_flop_fast_control")
    for mode, manifests in (("fast", fast), ("slow", slow), ("control", control)):
        system_contracts = {
            (
                manifest["system_checkpoint_tree_sha256"],
                manifest["fast_path_state_sha256"],
                manifest["system_config_sha256"],
                manifest["completed_run_lineage_sha256"],
                manifest["comparison_group_sha256"],
                manifest["environment_sha256"],
                manifest["decoding_contract_sha256"],
            )
            for manifest in manifests.values()
        }
        if len(system_contracts) != 1:
            raise OracleEvaluationError(f"cross-slot {mode} system contract differs")
    expected_adaptive = (
        adaptive_run["checkpoint_tree"]["tree_sha256"],
        adaptive_run["state_projections"]["fast_path_state_sha256"]["state_sha256"],
        adaptive_run["immutable_inputs"]["system_config_sha256"],
        lineage_artifacts["adaptive"]["sha256"],
        adaptive_run["comparison_group_sha256"],
    )
    expected_control = (
        control_run["checkpoint_tree"]["tree_sha256"],
        control_run["state_projections"]["fast_path_state_sha256"]["state_sha256"],
        control_run["immutable_inputs"]["system_config_sha256"],
        lineage_artifacts["control"]["sha256"],
        control_run["comparison_group_sha256"],
    )
    for manifests, expected, mode in (
        (fast, expected_adaptive, "forced_fast"),
        (slow, expected_adaptive, "forced_slow"),
        (control, expected_control, "equal_flop_fast_control"),
    ):
        for manifest in manifests.values():
            observed = tuple(manifest[key] for key in LINEAGE_HASHES)
            if observed != expected:
                raise OracleEvaluationError(f"{mode} completed-run lineage differs")
    workspace_contracts = {
        (
            manifest["rows"][0]["workspace_diagnostics"]["iterations"],
            manifest["rows"][0]["workspace_diagnostics"]["workspace_plan_sha256"],
            manifest["rows"][0]["workspace_diagnostics"][
                "workspace_candidate_identity_sha256"
            ],
        )
        for manifest in slow.values()
    }
    if len(workspace_contracts) != 1:
        raise OracleEvaluationError("cross-slot slow workspace contract differs")
    workspace_contract = next(iter(workspace_contracts))
    if workspace_contract[1:] != (
        adaptive_run["immutable_inputs"]["workspace_plan_sha256"],
        adaptive_run["immutable_inputs"]["workspace_candidate_identity_sha256"],
    ):
        raise OracleEvaluationError("slow workspace lineage differs")
    benchmarks = {}
    paired_rows = {}
    for slot in GATE_SLOTS:
        benchmarks[slot], paired_rows[slot] = _pair_slot(
            fast[slot], slow[slot], control[slot]
        )

    seed_material = [
        manifest[slot]["manifest_sha256"]
        for manifest in (fast, slow, control)
        for slot in GATE_SLOTS
    ]
    seed = int(canonical_sha256(seed_material)[:16], 16)
    bootstrap = _paired_bootstrap(paired_rows, seed, bootstrap_replicates)
    macro = {
        field: _macro(benchmarks, field)
        for field in (
            "forced_fast_score",
            "forced_slow_score",
            "forced_equal_flop_control_score",
            "oracle_score",
            "mask_matched_control_score",
        )
    }
    macro["oracle_vs_fast_points"] = macro["oracle_score"] - macro["forced_fast_score"]
    macro["oracle_vs_mask_matched_control_points"] = (
        macro["oracle_score"] - macro["mask_matched_control_score"]
    )
    versus_fast = {
        slot: row["oracle_score"] - row["forced_fast_score"]
        for slot, row in benchmarks.items()
    }
    versus_control = {
        slot: row["oracle_score"] - row["mask_matched_control_score"]
        for slot, row in benchmarks.items()
    }
    checks = {
        "oracle_vs_fast_positive_paired_95ci": bootstrap["oracle_vs_fast_points"][
            "lower_95"
        ]
        > 0,
        "oracle_vs_mask_control_positive_paired_95ci": bootstrap[
            "oracle_vs_mask_matched_control_points"
        ]["lower_95"]
        > 0,
        "oracle_macro_beats_fast_by_at_least_1_point": macro["oracle_vs_fast_points"]
        >= 1.0,
        "oracle_macro_beats_mask_control_by_at_least_1_point": macro[
            "oracle_vs_mask_matched_control_points"
        ]
        >= 1.0,
        "no_gate_slot_regresses_over_1_point_vs_fast": all(
            value >= -1.0 for value in versus_fast.values()
        ),
        "no_gate_slot_regresses_over_1_point_vs_mask_control": all(
            value >= -1.0 for value in versus_control.values()
        ),
        "oracle_beats_fast_on_at_least_four_slots": sum(
            value > 0 for value in versus_fast.values()
        )
        >= 4,
        "oracle_beats_mask_control_on_at_least_four_slots": sum(
            value > 0 for value in versus_control.values()
        )
        >= 4,
        "multi_step_reasoning_nonnegative_vs_both": min(
            versus_fast["multi_step_reasoning"],
            versus_control["multi_step_reasoning"],
        )
        >= 0,
        "self_correction_nonnegative_vs_both": min(
            versus_fast["self_correction"], versus_control["self_correction"]
        )
        >= 0,
        "slow_and_control_exactly_flop_matched": all(
            row["oracle_modeled_flops"] == row["mask_matched_control_modeled_flops"]
            and row["oracle_executed_flops"]
            == row["mask_matched_control_executed_flops"]
            for row in benchmarks.values()
        ),
        "terminal_public_boards_unopened": True,
    }
    supported = all(checks.values())
    artifacts = {
        mode: [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
        for mode, paths in zip(
            MODES, (fast_paths, slow_paths, control_paths), strict=True
        )
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "decision": (
            "oracle_slow_path_supported" if supported else "oracle_slow_path_rejected"
        ),
        "architecture_locked": False,
        "training_authorized": False,
        "terminal_public_board_accessed": False,
        "next_falsification_gate_authorized": supported,
        "manifest_artifacts": artifacts,
        "completed_run_lineage_artifacts": lineage_artifacts,
        "benchmarks": benchmarks,
        "macro": macro,
        "bootstrap": bootstrap,
        "checks": checks,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


def validate_analysis(
    payload: Any,
    fast_paths: list[Path],
    slow_paths: list[Path],
    control_paths: list[Path],
    *,
    adaptive_lineage: Path,
    adaptive_artifact_root: Path,
    control_lineage: Path,
    control_artifact_root: Path,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Reopen all manifests and replay one oracle analysis exactly."""

    if not isinstance(payload, dict):
        raise OracleEvaluationError("oracle analysis must be an object")
    analysis_hash = payload.get("analysis_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "analysis_sha256"
    }
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "complete"
        or payload.get("architecture_locked") is not False
        or payload.get("training_authorized") is not False
        or payload.get("terminal_public_board_accessed") is not False
        or analysis_hash != canonical_sha256(unsigned)
    ):
        raise OracleEvaluationError("oracle analysis identity or boundary differs")
    expected = analyze(
        fast_paths,
        slow_paths,
        control_paths,
        adaptive_lineage=adaptive_lineage,
        adaptive_artifact_root=adaptive_artifact_root,
        control_lineage=control_lineage,
        control_artifact_root=control_artifact_root,
        bootstrap_replicates=bootstrap_replicates,
    )
    if payload != expected:
        raise OracleEvaluationError("oracle analysis differs from exact replay")
    return payload


def write_analysis(
    fast_paths: list[Path],
    slow_paths: list[Path],
    control_paths: list[Path],
    output: Path,
    *,
    adaptive_lineage: Path,
    adaptive_artifact_root: Path,
    control_lineage: Path,
    control_artifact_root: Path,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    if output.exists():
        raise OracleEvaluationError("oracle output already exists")
    payload = analyze(
        fast_paths,
        slow_paths,
        control_paths,
        adaptive_lineage=adaptive_lineage,
        adaptive_artifact_root=adaptive_artifact_root,
        control_lineage=control_lineage,
        control_artifact_root=control_artifact_root,
        bootstrap_replicates=bootstrap_replicates,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", type=Path, action="append", required=True)
    parser.add_argument("--slow", type=Path, action="append", required=True)
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--adaptive-lineage", type=Path, required=True)
    parser.add_argument("--adaptive-artifact-root", type=Path, required=True)
    parser.add_argument("--control-lineage", type=Path, required=True)
    parser.add_argument("--control-artifact-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_analysis(
        args.fast,
        args.slow,
        args.control,
        args.output,
        adaptive_lineage=args.adaptive_lineage,
        adaptive_artifact_root=args.adaptive_artifact_root,
        control_lineage=args.control_lineage,
        control_artifact_root=args.control_artifact_root,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(
        json.dumps(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "decision": payload["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
