"""Emit the deterministic no-training Sai 16-slot workspace mechanics plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.adaptive.config import (
    default_workspace_config,
    workspace_activation_ledger,
    workspace_forward_flop_ledger,
    workspace_parameter_ledger,
)
from sai.model.planner import validate_plan as validate_geometry_plan

SCHEMA = "sai-16-slot-workspace-plan-v1"
HORIZONS = (1, 2, 4, 8, 16)
SEQUENCE_LENGTH = 2048
FAMILIES = ("gated_gqa", "gdn_hybrid", "kda_mla_hybrid")


class WorkspacePlanError(RuntimeError):
    """The base geometry, workspace mechanics, or no-training hold differs."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_plan(geometry_payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_geometry_plan(geometry_payload)
    if validated["vocab_size"] != 48_000:
        raise WorkspacePlanError("workspace Gate 0 requires the frozen 48K geometry")
    base_rows = {
        row["mixer_family"]: row
        for row in validated["geometries"]
        if row["scale"] == "300m"
    }
    if set(base_rows) != set(FAMILIES):
        raise WorkspacePlanError("exact 300M base families are required")

    candidates = []
    for family in FAMILIES:
        base = base_rows[family]
        config = base["config"]
        if config.get("hidden_size") != 768:
            raise WorkspacePlanError("300M base hidden width differs")
        workspace = default_workspace_config(config["hidden_size"])
        parameters = workspace_parameter_ledger(workspace)
        base_parameters = base["parameter_ledger"]["total"]
        flops = {
            str(horizon): workspace_forward_flop_ledger(
                workspace, SEQUENCE_LENGTH, horizon
            )
            for horizon in HORIZONS
        }
        row: dict[str, Any] = {
            "base_mixer_family": family,
            "base_geometry_identity_sha256": canonical_sha256(base),
            "base_parameters": base_parameters,
            "workspace_config": workspace.as_dict(),
            "workspace_parameter_ledger": parameters,
            "combined_parameters_without_base_reallocation": base_parameters
            + parameters["total"],
            "workspace_fraction_of_combined_parameters": parameters["total"]
            / (base_parameters + parameters["total"]),
            "workspace_flop_ledgers_by_iterations": flops,
            "workspace_activation_ledger": workspace_activation_ledger(
                workspace, SEQUENCE_LENGTH
            ),
        }
        row["candidate_identity_sha256"] = canonical_sha256(row)
        candidates.append(row)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "prospective_cpu_accounting",
        "training_hold": True,
        "training_authorized": False,
        "official_training_order_received": False,
        "gpu_jobs_submitted": 0,
        "training_updates_completed": 0,
        "selected_backbone_family": None,
        "primary_100m_screen_unchanged": True,
        "stage": "300m_post_mixer_factor_screen",
        "sequence_length": SEQUENCE_LENGTH,
        "recurrence_horizons": list(HORIZONS),
        "geometry_plan_sha256": validated["plan_sha256"],
        "workspace_factor": {
            "num_slots": 16,
            "query_conditioned_compiler": True,
            "reinject_initial_slots_each_reactor_block": True,
            "bidirectional_workspace_attention": True,
            "reader_applies_only_to_final_generation_position": True,
            "reader_output_initialized_exactly_zero": True,
            "forced_fast_is_direct_base_bypass": True,
            "fixed_point_loss_included": False,
            "learned_regret_controller_included": False,
            "semantic_memory_included": False,
            "typed_side_channels_included": False,
            "learned_exact_anchors_included": False,
        },
        "accounting_scope": {
            "parameter_count_exact": True,
            "flop_convention": (
                "matmul_attention_one_multiply_add_equals_two_"
                "normalization_softmax_nonlinearity_excluded"
            ),
            "activation_scope": (
                "analytical_incremental_tensor_geometry_not_framework_peak"
            ),
            "production_allocator_peak_still_required": True,
            "memory_traffic_microbenchmark_still_required": True,
        },
        "candidates": candidates,
        "checks": {
            "one_new_factor_only": True,
            "base_backbone_not_shrunk_or_reallocated": True,
            "workspace_parameters_independent_of_recurrence_horizon": True,
            "oracle_evidence_not_yet_available": True,
            "architecture_not_locked": True,
            "training_remains_unauthorized": True,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(payload: Any, geometry_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise WorkspacePlanError("workspace plan must be an object")
    plan_hash = payload.get("plan_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "prospective_cpu_accounting"
        or payload.get("training_hold") is not True
        or payload.get("training_authorized") is not False
        or payload.get("official_training_order_received") is not False
        or payload.get("gpu_jobs_submitted") != 0
        or payload.get("training_updates_completed") != 0
        or payload.get("selected_backbone_family") is not None
        or payload.get("primary_100m_screen_unchanged") is not True
        or plan_hash != canonical_sha256(unsigned)
    ):
        raise WorkspacePlanError("workspace plan identity or no-training hold differs")
    try:
        expected = build_plan(geometry_payload)
    except (ValueError, RuntimeError) as error:
        if isinstance(error, WorkspacePlanError):
            raise
        raise WorkspacePlanError("base geometry validation failed") from error
    if payload != expected:
        raise WorkspacePlanError("workspace plan differs from deterministic planner")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-plan", type=Path, required=True)
    args = parser.parse_args()
    try:
        geometry_payload = json.loads(args.geometry_plan.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkspacePlanError("geometry plan is unreadable") from error
    if not isinstance(geometry_payload, dict):
        raise WorkspacePlanError("geometry plan must be an object")
    print(json.dumps(build_plan(geometry_payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
