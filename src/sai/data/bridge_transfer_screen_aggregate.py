"""Aggregate matched unchanged/control/connection proxy-transfer arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_transfer_screen import ARMS
from sai.data.bridge_transfer_screen import SCHEMA as ARM_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bridge-transfer-proxy-screen-aggregate-v1"
MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT = 0.005
MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION = 0.01


class BridgeTransferScreenAggregateError(RuntimeError):
    """An arm identity, compute budget, score stream, or receipt differs."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BridgeTransferScreenAggregateError("arm receipt is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeTransferScreenAggregateError("arm receipt differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != ARM_SCHEMA
        or payload.get("status") != "complete_bridge_transfer_proxy_arm"
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("screen_is_proxy_not_4b_capability_claim") is not True
        or payload.get("transfer_ablation_complete") is not False
        or payload.get("training_ready") is not False
    ):
        raise BridgeTransferScreenAggregateError("arm receipt differs")
    return payload


def _metric(payload: dict[str, Any], stream: str) -> float:
    value = payload.get("evaluations", {}).get(stream, {}).get("mean_nll")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise BridgeTransferScreenAggregateError("arm metric differs")
    return float(value)


def aggregate(arm_root: Path, output: Path) -> dict[str, Any]:
    """Verify equal compute and decide whether multi-seed confirmation is warranted."""

    if output.exists() or output.is_symlink():
        raise BridgeTransferScreenAggregateError("aggregate output exists")
    paths = {arm: arm_root / f"{arm}.json" for arm in ARMS}
    arms = {arm: _load(path) for arm, path in paths.items()}
    if any(payload.get("arm") != arm for arm, payload in arms.items()):
        raise BridgeTransferScreenAggregateError("arm identity differs")
    common = arms["unchanged"]
    for arm, payload in arms.items():
        if (
            payload.get("code_commit") != common.get("code_commit")
            or not isinstance(payload.get("code_commit"), str)
            or len(payload["code_commit"]) != 40
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
            or payload.get("training", {}).get("seed")
            != common.get("training", {}).get("seed")
            or payload.get("training", {}).get("matched_token_budget")
            != common.get("training", {}).get("matched_token_budget")
        ):
            raise BridgeTransferScreenAggregateError(f"{arm} matched identity differs")
    if (
        arms["unchanged"]["training"].get("used_train_tokens") != 0
        or arms["unchanged"]["training"].get("optimizer_steps") != 0
        or arms["unchanged"]["model"].get("final_state_sha256")
        != arms["unchanged"]["model"].get("initial_state_sha256")
    ):
        raise BridgeTransferScreenAggregateError("unchanged control differs")
    budget = common["training"]["matched_token_budget"]
    for arm in ("source_control", "connections"):
        if (
            arms[arm]["training"].get("used_train_tokens") != budget
            or arms[arm]["training"].get("optimizer_steps", 0) <= 0
            or arms[arm]["model"].get("final_state_sha256")
            == arms[arm]["model"].get("initial_state_sha256")
        ):
            raise BridgeTransferScreenAggregateError(f"{arm} training differs")
    unchanged_connection = _metric(arms["unchanged"], "connection_development")
    control_connection = _metric(arms["source_control"], "connection_development")
    treatment_connection = _metric(arms["connections"], "connection_development")
    unchanged_source = _metric(arms["unchanged"], "source_development")
    control_source = _metric(arms["source_control"], "source_development")
    treatment_source = _metric(arms["connections"], "source_development")
    treatment_vs_control = (
        control_connection - treatment_connection
    ) / control_connection
    treatment_vs_unchanged = (
        unchanged_connection - treatment_connection
    ) / unchanged_connection
    source_vs_control = (treatment_source - control_source) / control_source
    source_vs_unchanged = (treatment_source - unchanged_source) / unchanged_source
    screen_pass = (
        treatment_vs_control >= MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT
        and treatment_vs_unchanged >= MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT
        and source_vs_control <= MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION
        and source_vs_unchanged <= MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete_bridge_transfer_proxy_screen",
        "code_commit": common["code_commit"],
        "inputs": {
            arm: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "receipt_sha256": arms[arm]["receipt_sha256"],
            }
            for arm, path in paths.items()
        },
        "model_repository": common["model"]["repository"],
        "model_revision": common["model"]["revision"],
        "seed": common["training"]["seed"],
        "matched_token_budget": budget,
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
        "effects": {
            "relative_connection_improvement_vs_source_control": treatment_vs_control,
            "relative_connection_improvement_vs_unchanged": treatment_vs_unchanged,
            "relative_source_regression_vs_source_control": source_vs_control,
            "relative_source_regression_vs_unchanged": source_vs_unchanged,
        },
        "thresholds": {
            "minimum_relative_connection_improvement": (
                MINIMUM_RELATIVE_CONNECTION_IMPROVEMENT
            ),
            "maximum_relative_source_retention_regression": (
                MAXIMUM_RELATIVE_SOURCE_RETENTION_REGRESSION
            ),
        },
        "screen_pass": screen_pass,
        "multi_seed_confirmation_authorized": screen_pass,
        "connection_component_admission_authorized": False,
        "screen_is_proxy_not_4b_capability_claim": True,
        "transfer_ablation_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.arm_root, args.output)
    print(
        json.dumps(
            {
                "screen_pass": result["screen_pass"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
