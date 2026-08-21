"""Emit deterministic, no-training Sai scale geometries and parameter ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from sai.model.config import frozen_scale_geometries

SCHEMA = "sai-model-geometry-plan-v1"
ALLOWED_VOCAB_SIZES = {32_000, 48_000, 64_000}


class SaiModelPlanError(ValueError):
    """The requested geometry plan differs from the frozen tokenizer tournament."""


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_plan(vocab_size: int) -> dict[str, Any]:
    if vocab_size not in ALLOWED_VOCAB_SIZES:
        raise SaiModelPlanError("vocabulary size is outside the primary tournament")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "prospective_cpu_reference",
        "training_hold": True,
        "training_authorized": False,
        "official_training_order_received": False,
        "gpu_jobs_submitted": 0,
        "training_updates_completed": 0,
        "vocab_size": vocab_size,
        "geometry_method": (
            "nearest_64_ffn_width_with_exact_analytical_parameter_ledger"
        ),
        "reference_scope": [
            "causal_forward",
            "backward_gradients",
            "kda_gdn_recurrent_state",
            "gqa_partial_rope",
            "nope_gated_mla",
            "tied_embeddings",
        ],
        "geometries": frozen_scale_geometries(vocab_size),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SaiModelPlanError("model plan must be an object")
    plan_hash = payload.get("plan_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "prospective_cpu_reference"
        or payload.get("training_hold") is not True
        or payload.get("training_authorized") is not False
        or payload.get("official_training_order_received") is not False
        or payload.get("gpu_jobs_submitted") != 0
        or payload.get("training_updates_completed") != 0
        or payload.get("vocab_size") not in ALLOWED_VOCAB_SIZES
        or plan_hash != canonical_sha256(unsigned)
    ):
        raise SaiModelPlanError("model plan identity or no-training boundary differs")
    expected = build_plan(payload["vocab_size"])
    if payload != expected:
        raise SaiModelPlanError("model geometries differ from deterministic planner")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-size", type=int, choices=sorted(ALLOWED_VOCAB_SIZES), required=True
    )
    args = parser.parse_args()
    print(json.dumps(build_plan(args.vocab_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
