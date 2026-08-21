"""Build an exact, no-training Sai 100M architecture-tournament plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from sai.architecture.tournament import validate as validate_architecture_tournament
from sai.model.config import SaiModelConfig, forward_flop_ledger
from sai.model.planner import validate_plan as validate_geometry_plan

SCHEMA = "sai-100m-experiment-plan-v1"
TOKENIZER_SCHEMA = "sai-tokenizer-qualification-receipt-v1"
STREAM_SCHEMA = "sai-ordered-token-stream-receipt-v1"
ENVIRONMENT_SCHEMA = "sai-training-environment-receipt-v1"
FAMILIES = ("gated_gqa", "gdn_hybrid", "kda_mla_hybrid")
SEEDS = (20260821, 20260822, 20260823)
CONTRASTS = ("iso_data", "iso_flop")
EXPECTED_TEMPLATE: dict[str, Any] = {
    "schema": "sai-100m-tournament-template-v1",
    "status": "prospective",
    "training_hold": True,
    "training_authorized": False,
    "official_training_order_received": False,
    "gpu_jobs_submitted": 0,
    "training_updates_completed": 0,
    "scale": "100m",
    "vocab_size": 48_000,
    "mixer_families": list(FAMILIES),
    "seeds": list(SEEDS),
    "sequence_length": 2_048,
    "sequences_per_full_update": 256,
    "iso_data_full_updates": 4_096,
    "objective": {
        "name": "causal_next_token_prediction",
        "cross_document_targets_masked": True,
        "padding_targets_masked": True,
        "training_reduction": "mean_per_valid_token",
        "validation_reduction": "negative_log_likelihood_per_utf8_byte",
    },
    "optimizer": {
        "name": "adamw",
        "learning_rate": 0.0006,
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "decay_matrices_only": True,
    },
    "schedule": {
        "name": "linear_warmup_cosine_decay",
        "warmup_fraction": 0.01,
        "minimum_learning_rate_ratio": 0.1,
    },
    "precision": {
        "parameters": "bfloat16",
        "activations": "bfloat16",
        "optimizer_states": "float32",
        "recurrent_state": "float32",
    },
    "comparison": {
        "iso_data": "same_ordered_token_stream_prefix_and_utf8_bytes",
        "iso_flop": "exact_common_model_flops_via_integer_sequence_lcm",
        "hardware_counters_reported_separately": True,
        "one_changed_factor": "mixer_family",
    },
}


class ExperimentPlanError(RuntimeError):
    """An experiment input, comparison budget, or no-training boundary differs."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ExperimentPlanError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ExperimentPlanError(f"{field} differs") from error
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentPlanError(f"{field} differs")
    return value


def _load_json(path: Path, field: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ExperimentPlanError(f"{field} is missing or unsafe: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExperimentPlanError(f"{field} is unreadable") from error
    if not isinstance(payload, dict):
        raise ExperimentPlanError(f"{field} must be an object")
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved != path.resolve() or not path.is_file() or path.is_symlink():
        raise ExperimentPlanError(f"artifact is missing or unsafe: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ExperimentPlanError(f"artifact is empty: {path}")
    return {"path": str(resolved), "sha256": sha256_file(path), "bytes": size}


def validate_template(payload: Any) -> dict[str, Any]:
    if payload != EXPECTED_TEMPLATE:
        raise ExperimentPlanError("100M tournament template differs")
    return payload


def _validate_tokenizer(payload: dict[str, Any]) -> dict[str, Any]:
    if (
        payload.get("schema") != TOKENIZER_SCHEMA
        or payload.get("status") != "qualified"
        or payload.get("training_authorized") is not False
        or payload.get("vocab_size") != EXPECTED_TEMPLATE["vocab_size"]
        or payload.get("byte_fallback") is not True
        or payload.get("roundtrip_failures") != 0
        or payload.get("special_tokens_preserved") is not True
    ):
        raise ExperimentPlanError("tokenizer qualification differs")
    return {
        "schema": TOKENIZER_SCHEMA,
        "status": "qualified",
        "vocab_size": EXPECTED_TEMPLATE["vocab_size"],
        "tokenizer_identity_sha256": _sha256(
            payload.get("tokenizer_identity_sha256"), "tokenizer identity"
        ),
        "corpus_identity_sha256": _sha256(
            payload.get("corpus_identity_sha256"), "tokenizer corpus identity"
        ),
        "byte_fallback": True,
        "roundtrip_failures": 0,
        "special_tokens_preserved": True,
        "training_authorized": False,
    }


def _validate_stream(
    payload: dict[str, Any], tokenizer: dict[str, Any]
) -> dict[str, Any]:
    if (
        payload.get("schema") != STREAM_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("training_authorized") is not False
        or payload.get("tokenizer_identity_sha256")
        != tokenizer["tokenizer_identity_sha256"]
        or payload.get("sequence_length") != EXPECTED_TEMPLATE["sequence_length"]
        or payload.get("benchmark_disjoint") is not True
        or payload.get("cross_document_targets_masked") is not True
    ):
        raise ExperimentPlanError("ordered token stream differs")
    sequences = _positive_integer(payload.get("sequences"), "stream sequences")
    valid_tokens = _positive_integer(payload.get("valid_tokens"), "valid tokens")
    admitted_bytes = _positive_integer(
        payload.get("admitted_utf8_bytes"), "admitted UTF-8 bytes"
    )
    if valid_tokens > sequences * EXPECTED_TEMPLATE["sequence_length"]:
        raise ExperimentPlanError("valid tokens exceed the packed stream geometry")
    prefixes = payload.get("prefix_utf8_bytes")
    if not isinstance(prefixes, dict) or not prefixes:
        raise ExperimentPlanError("stream prefix-byte ledger is missing")
    normalized_prefixes: dict[str, int] = {}
    previous_count = previous_bytes = 0
    try:
        ordered_prefixes = sorted((int(key), value) for key, value in prefixes.items())
    except (TypeError, ValueError) as error:
        raise ExperimentPlanError("stream prefix-byte keys differ") from error
    for count, byte_count in ordered_prefixes:
        byte_count = _positive_integer(byte_count, "stream prefix UTF-8 bytes")
        if count <= previous_count or count > sequences or byte_count < previous_bytes:
            raise ExperimentPlanError("stream prefix-byte ledger is not monotonic")
        normalized_prefixes[str(count)] = byte_count
        previous_count, previous_bytes = count, byte_count
    if normalized_prefixes.get(str(sequences)) != admitted_bytes:
        raise ExperimentPlanError("full stream byte count differs")
    return {
        "schema": STREAM_SCHEMA,
        "status": "complete",
        "tokenizer_identity_sha256": tokenizer["tokenizer_identity_sha256"],
        "ordered_stream_identity_sha256": _sha256(
            payload.get("ordered_stream_identity_sha256"), "ordered stream identity"
        ),
        "source_manifest_sha256": _sha256(
            payload.get("source_manifest_sha256"), "stream source manifest"
        ),
        "sequence_length": EXPECTED_TEMPLATE["sequence_length"],
        "sequences": sequences,
        "valid_tokens": valid_tokens,
        "admitted_utf8_bytes": admitted_bytes,
        "prefix_utf8_bytes": normalized_prefixes,
        "benchmark_disjoint": True,
        "cross_document_targets_masked": True,
        "training_authorized": False,
    }


def _validate_environment(payload: dict[str, Any]) -> dict[str, Any]:
    if (
        payload.get("schema") != ENVIRONMENT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("training_authorized") is not False
    ):
        raise ExperimentPlanError("training environment receipt differs")
    versions = payload.get("versions")
    if (
        not isinstance(versions, dict)
        or set(versions) != {"python", "torch", "cuda", "triton"}
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise ExperimentPlanError("training environment versions differ")
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "status": "complete",
        "environment_identity_sha256": _sha256(
            payload.get("environment_identity_sha256"), "environment identity"
        ),
        "versions": versions,
        "training_authorized": False,
    }


def _geometry_rows(geometry_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validated = validate_geometry_plan(geometry_payload)
    if validated["vocab_size"] != EXPECTED_TEMPLATE["vocab_size"]:
        raise ExperimentPlanError("geometry vocabulary differs")
    rows = {
        row["mixer_family"]: row
        for row in validated["geometries"]
        if row["scale"] == EXPECTED_TEMPLATE["scale"]
    }
    if set(rows) != set(FAMILIES):
        raise ExperimentPlanError("exact 100M family geometries are required")
    return rows


def _exact_flop_geometry(
    rows: dict[str, dict[str, Any]], iso_data_sequences: int
) -> tuple[dict[str, int], int, dict[str, int]]:
    per_sequence = {}
    for family in FAMILIES:
        config = SaiModelConfig(**rows[family]["config"])
        per_sequence[family] = forward_flop_ledger(
            config, EXPECTED_TEMPLATE["sequence_length"]
        )["forward_plus_backward_approximation"]
    quantum = math.lcm(*per_sequence.values())
    maximum = min(per_sequence.values()) * iso_data_sequences
    common_budget = (maximum // quantum) * quantum
    if common_budget <= 0:
        raise ExperimentPlanError("iso-FLOP budget has no exact common quantum")
    iso_flop_sequences = {
        family: common_budget // per_sequence[family] for family in FAMILIES
    }
    if any(count <= 0 for count in iso_flop_sequences.values()):
        raise ExperimentPlanError("iso-FLOP sequence geometry differs")
    return per_sequence, common_budget, iso_flop_sequences


def build_plan(
    template_path: Path,
    architecture_path: Path,
    geometry_path: Path,
    tokenizer_path: Path,
    stream_path: Path,
    environment_path: Path,
) -> dict[str, Any]:
    template = validate_template(_load_json(template_path, "tournament template"))
    architecture_payload = _load_json(architecture_path, "architecture tournament")
    architecture_receipt = validate_architecture_tournament(architecture_payload)
    geometry_payload = _load_json(geometry_path, "geometry plan")
    rows = _geometry_rows(geometry_payload)
    tokenizer = _validate_tokenizer(_load_json(tokenizer_path, "tokenizer receipt"))
    stream = _validate_stream(
        _load_json(stream_path, "ordered token stream receipt"), tokenizer
    )
    environment = _validate_environment(
        _load_json(environment_path, "environment receipt")
    )

    sequences_per_update = template["sequences_per_full_update"]
    iso_data_sequences = template["iso_data_full_updates"] * sequences_per_update
    per_sequence, common_flops, iso_flop_sequences = _exact_flop_geometry(
        rows, iso_data_sequences
    )
    required_prefixes = {iso_data_sequences, *iso_flop_sequences.values()}
    if stream["sequences"] < max(required_prefixes):
        raise ExperimentPlanError("ordered stream is too short for the tournament")
    missing_prefixes = required_prefixes - {
        int(value) for value in stream["prefix_utf8_bytes"]
    }
    if missing_prefixes:
        raise ExperimentPlanError(
            f"ordered stream lacks required prefixes: {sorted(missing_prefixes)}"
        )

    runs = []
    for contrast in CONTRASTS:
        for family in FAMILIES:
            sequence_count = (
                iso_data_sequences
                if contrast == "iso_data"
                else iso_flop_sequences[family]
            )
            modeled_flops = per_sequence[family] * sequence_count
            if contrast == "iso_flop" and modeled_flops != common_flops:
                raise ExperimentPlanError("iso-FLOP arithmetic is not exact")
            for seed in SEEDS:
                run = {
                    "contrast": contrast,
                    "mixer_family": family,
                    "seed": seed,
                    "geometry_identity_sha256": canonical_sha256(rows[family]),
                    "ordered_stream_identity_sha256": stream[
                        "ordered_stream_identity_sha256"
                    ],
                    "prefix_sequences": sequence_count,
                    "prefix_valid_token_capacity": sequence_count
                    * template["sequence_length"],
                    "prefix_utf8_bytes": stream["prefix_utf8_bytes"][
                        str(sequence_count)
                    ],
                    "modeled_training_flops": modeled_flops,
                    "full_updates": sequence_count // sequences_per_update,
                    "final_partial_update_sequences": sequence_count
                    % sequences_per_update,
                }
                run["run_identity_sha256"] = canonical_sha256(run)
                runs.append(run)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "prospective_exact_plan",
        "training_hold": True,
        "training_authorized": False,
        "official_training_order_received": False,
        "gpu_jobs_submitted": 0,
        "training_updates_completed": 0,
        "inputs": {
            "template": _artifact(template_path),
            "architecture_tournament": _artifact(architecture_path),
            "geometry_plan": _artifact(geometry_path),
            "tokenizer_receipt": _artifact(tokenizer_path),
            "ordered_token_stream_receipt": _artifact(stream_path),
            "environment_receipt": _artifact(environment_path),
        },
        "architecture_plan_sha256": architecture_receipt["plan_sha256"],
        "geometry_plan_sha256": geometry_payload["plan_sha256"],
        "tokenizer": tokenizer,
        "ordered_token_stream": stream,
        "environment": environment,
        "fixed_training_contract": template,
        "budget_geometry": {
            "iso_data_sequences": iso_data_sequences,
            "iso_data_valid_token_capacity": iso_data_sequences
            * template["sequence_length"],
            "exact_iso_flop_common_budget": common_flops,
            "per_sequence_forward_backward_flops": per_sequence,
            "iso_flop_sequences": iso_flop_sequences,
            "integer_flop_quantum_lcm": math.lcm(*per_sequence.values()),
        },
        "runs": runs,
        "checks": {
            "three_families": True,
            "three_seeds": True,
            "separate_iso_data_and_iso_flop": True,
            "iso_data_same_ordered_prefix_and_bytes": True,
            "iso_flop_exact_integer_equality": True,
            "training_not_authorized": True,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ExperimentPlanError("experiment plan must be an object")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "prospective_exact_plan"
        or payload.get("training_hold") is not True
        or payload.get("training_authorized") is not False
        or payload.get("official_training_order_received") is not False
        or payload.get("gpu_jobs_submitted") != 0
        or payload.get("training_updates_completed") != 0
    ):
        raise ExperimentPlanError("experiment plan no-training boundary differs")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "template",
        "architecture_tournament",
        "geometry_plan",
        "tokenizer_receipt",
        "ordered_token_stream_receipt",
        "environment_receipt",
    }:
        raise ExperimentPlanError("experiment input receipts differ")
    paths = []
    for name in (
        "template",
        "architecture_tournament",
        "geometry_plan",
        "tokenizer_receipt",
        "ordered_token_stream_receipt",
        "environment_receipt",
    ):
        receipt = inputs[name]
        if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
            raise ExperimentPlanError(f"{name} input receipt differs")
        paths.append(Path(receipt["path"]))
    expected = build_plan(*paths)
    if payload != expected:
        raise ExperimentPlanError("experiment plan or bound inputs differ")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--architecture-tournament", type=Path, required=True)
    parser.add_argument("--geometry-plan", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--ordered-token-stream-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ExperimentPlanError("experiment plan output already exists")
    plan = build_plan(
        args.template,
        args.architecture_tournament,
        args.geometry_plan,
        args.tokenizer_receipt,
        args.ordered_token_stream_receipt,
        args.environment_receipt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "status": plan["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
