"""Aggregate fresh-seed confirmation of verified connection-data transfer."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_transfer_screen import ARMS
from sai.data.bridge_transfer_screen_aggregate import (
    ARM_SCHEMA,
    MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION,
    MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT,
    _metric,
)
from sai.data.bridge_transfer_screen_aggregate import (
    SCHEMA as SCREEN_SCHEMA,
)
from sai.data.bridge_transfer_screen_aggregate import (
    _load as _load_arm,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bridge-transfer-proxy-confirmation-v1"
CONFIRMATION_SEEDS = (20_260_827, 20_260_828, 20_260_829)


class BridgeTransferConfirmationError(RuntimeError):
    """A screen, fresh-seed arm, matched identity, or score differs."""


def _load_screen(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BridgeTransferConfirmationError("screen receipt is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeTransferConfirmationError("screen receipt differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SCREEN_SCHEMA
        or payload.get("status") != "complete_bridge_transfer_proxy_screen"
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("screen_pass") is not True
        or payload.get("multi_seed_confirmation_authorized") is not True
        or payload.get("connection_component_admission_authorized") is not False
        or payload.get("transfer_ablation_complete") is not False
        or payload.get("training_ready") is not False
        or payload.get("four_b_training_authorized") is not False
    ):
        raise BridgeTransferConfirmationError("screen did not authorize confirmation")
    return payload


def _assert_matched_seed(
    arms: dict[str, dict[str, Any]], seed: int
) -> tuple[dict[str, Any], int]:
    if any(payload.get("arm") != arm for arm, payload in arms.items()):
        raise BridgeTransferConfirmationError("arm identity differs")
    common = arms["unchanged"]
    budget = common.get("training", {}).get("matched_token_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise BridgeTransferConfirmationError("matched budget differs")
    for arm, payload in arms.items():
        training = payload.get("training", {})
        if (
            payload.get("schema") != ARM_SCHEMA
            or training.get("seed") != seed
            or payload.get("code_commit") != common.get("code_commit")
            or payload.get("lineage") != common.get("lineage")
            or payload.get("model", {}).get("repository")
            != common.get("model", {}).get("repository")
            or payload.get("model", {}).get("revision")
            != common.get("model", {}).get("revision")
            or payload.get("model", {}).get("ordered_files_sha256")
            != common.get("model", {}).get("ordered_files_sha256")
            or payload.get("model", {}).get("initial_state_sha256")
            != common.get("model", {}).get("initial_state_sha256")
            or payload.get("tokenizer") != common.get("tokenizer")
            or payload.get("evaluation_streams") != common.get("evaluation_streams")
            or training.get("matched_token_budget") != budget
            or training.get("block_size")
            != common.get("training", {}).get("block_size")
            or training.get("micro_batch_size")
            != common.get("training", {}).get("micro_batch_size")
            or training.get("gradient_accumulation")
            != common.get("training", {}).get("gradient_accumulation")
            or training.get("learning_rate")
            != common.get("training", {}).get("learning_rate")
            or training.get("weight_decay")
            != common.get("training", {}).get("weight_decay")
            or training.get("warmup_fraction")
            != common.get("training", {}).get("warmup_fraction")
        ):
            raise BridgeTransferConfirmationError(f"seed {seed} {arm} identity differs")
    initial = common["model"]["initial_state_sha256"]
    if (
        common["training"].get("used_train_tokens") != 0
        or common["training"].get("optimizer_steps") != 0
        or common["model"].get("final_state_sha256") != initial
    ):
        raise BridgeTransferConfirmationError("unchanged control differs")
    for arm in ("source_control", "connections"):
        if (
            arms[arm]["training"].get("used_train_tokens") != budget
            or arms[arm]["training"].get("optimizer_steps", 0) <= 0
            or arms[arm]["model"].get("final_state_sha256") == initial
            or not isinstance(arms[arm]["training"].get("selected_stream_sha256"), str)
        ):
            raise BridgeTransferConfirmationError(f"{arm} training differs")
    return common, budget


def aggregate(screen_path: Path, arm_root: Path, output: Path) -> dict[str, Any]:
    """Require the connection treatment to pass every fresh matched seed."""

    if output.exists() or output.is_symlink():
        raise BridgeTransferConfirmationError("confirmation output exists")
    screen = _load_screen(screen_path)
    seed_payloads: dict[int, dict[str, dict[str, Any]]] = {}
    input_descriptors: dict[str, dict[str, dict[str, Any]]] = {}
    per_seed: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    reference_budget: int | None = None
    reference_streams: dict[str, str] = {}
    for seed in CONFIRMATION_SEEDS:
        paths = {arm: arm_root / f"seed_{seed}" / f"{arm}.json" for arm in ARMS}
        arms = {arm: _load_arm(path) for arm, path in paths.items()}
        common, budget = _assert_matched_seed(arms, seed)
        if reference is None:
            reference = common
            reference_budget = budget
            reference_streams = {
                arm: str(arms[arm]["training"].get("selected_stream_sha256"))
                for arm in ("source_control", "connections")
            }
        elif (
            common.get("code_commit") != reference.get("code_commit")
            or common.get("lineage") != reference.get("lineage")
            or common.get("model", {}).get("repository")
            != reference.get("model", {}).get("repository")
            or common.get("model", {}).get("revision")
            != reference.get("model", {}).get("revision")
            or common.get("model", {}).get("ordered_files_sha256")
            != reference.get("model", {}).get("ordered_files_sha256")
            or common.get("model", {}).get("initial_state_sha256")
            != reference.get("model", {}).get("initial_state_sha256")
            or common.get("tokenizer") != reference.get("tokenizer")
            or common.get("evaluation_streams") != reference.get("evaluation_streams")
            or budget != reference_budget
            or any(
                arms[arm]["training"].get("selected_stream_sha256")
                != reference_streams[arm]
                for arm in ("source_control", "connections")
            )
        ):
            raise BridgeTransferConfirmationError("cross-seed matched identity differs")
        unchanged_connection = _metric(arms["unchanged"], "connection_development")
        control_connection = _metric(arms["source_control"], "connection_development")
        treatment_connection = _metric(arms["connections"], "connection_development")
        unchanged_source = _metric(arms["unchanged"], "source_development")
        control_source = _metric(arms["source_control"], "source_development")
        treatment_source = _metric(arms["connections"], "source_development")
        effects = {
            "relative_connection_improvement_vs_source_control": (
                control_connection - treatment_connection
            )
            / control_connection,
            "relative_connection_improvement_vs_unchanged": (
                unchanged_connection - treatment_connection
            )
            / unchanged_connection,
            "relative_source_regression_vs_source_control": (
                treatment_source - control_source
            )
            / control_source,
            "relative_source_regression_vs_unchanged": (
                treatment_source - unchanged_source
            )
            / unchanged_source,
        }
        seed_pass = (
            effects["relative_connection_improvement_vs_source_control"]
            >= MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT
            and effects["relative_connection_improvement_vs_unchanged"]
            >= MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT
            and effects["relative_source_regression_vs_source_control"]
            <= MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION
            and effects["relative_source_regression_vs_unchanged"]
            <= MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION
        )
        per_seed[str(seed)] = {
            "scores": {
                "connection_development_mean_nll": {
                    "unchanged": unchanged_connection,
                    "source_control": control_connection,
                    "connections": treatment_connection,
                },
                "source_development_mean_nll": {
                    "unchanged": unchanged_source,
                    "source_control": control_source,
                    "connections": treatment_source,
                },
            },
            "effects": effects,
            "pass": seed_pass,
        }
        input_descriptors[str(seed)] = {
            arm: {
                "path": str(path.relative_to(arm_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "receipt_sha256": arms[arm]["receipt_sha256"],
            }
            for arm, path in paths.items()
        }
        seed_payloads[seed] = arms
    assert reference is not None and reference_budget is not None
    effect_names = tuple(next(iter(per_seed.values()))["effects"])
    medians = {
        name: statistics.median(
            per_seed[str(seed)]["effects"][name] for seed in CONFIRMATION_SEEDS
        )
        for name in effect_names
    }
    confirmation_pass = all(value["pass"] for value in per_seed.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete_bridge_transfer_proxy_confirmation",
        "screen": {
            "path": screen_path.name,
            "bytes": screen_path.stat().st_size,
            "sha256": sha256_file(screen_path),
            "receipt_sha256": screen["receipt_sha256"],
            "code_commit": screen["code_commit"],
            "seed": screen["seed"],
        },
        "confirmation_code_commit": reference["code_commit"],
        "lineage": reference["lineage"],
        "inputs": input_descriptors,
        "seeds": list(CONFIRMATION_SEEDS),
        "model_repository": reference["model"]["repository"],
        "model_revision": reference["model"]["revision"],
        "matched_token_budget": reference_budget,
        "per_seed": per_seed,
        "median_effects": medians,
        "thresholds": {
            "minimum_relative_connection_improvement": (
                MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT
            ),
            "maximum_relative_source_retention_regression": (
                MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION
            ),
            "all_fresh_seeds_must_pass": True,
        },
        "confirmation_pass": confirmation_pass,
        "connection_component_admission_authorized": confirmation_pass,
        "screen_is_proxy_not_4b_capability_claim": True,
        "transfer_ablation_complete": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.screen, args.arm_root, args.output)
    print(
        json.dumps(
            {
                "confirmation_pass": result["confirmation_pass"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
