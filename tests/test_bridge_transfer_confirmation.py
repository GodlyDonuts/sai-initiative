from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.bridge_transfer_confirmation import (
    CONFIRMATION_SEEDS,
    BridgeTransferConfirmationError,
    aggregate,
)
from sai.data.bridge_transfer_screen import SCHEMA as ARM_SCHEMA
from sai.data.bridge_transfer_screen_aggregate import SCHEMA as SCREEN_SCHEMA
from sai.data.token_stream import canonical_sha256


def _signed(payload: dict) -> dict:
    value = dict(payload)
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _screen() -> dict:
    return _signed(
        {
            "schema": SCREEN_SCHEMA,
            "status": "complete_bridge_transfer_proxy_screen",
            "code_commit": "0" * 40,
            "seed": 20_260_826,
            "screen_pass": True,
            "multi_seed_confirmation_authorized": True,
            "connection_component_admission_authorized": False,
            "transfer_ablation_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
    )


def _arm(name: str, seed: int, connection_nll: float, source_nll: float) -> dict:
    initial = "a" * 64
    payload = {
        "schema": ARM_SCHEMA,
        "status": "complete_bridge_transfer_proxy_arm",
        "arm": name,
        "code_commit": "1" * 40,
        "lineage": {"reconciliation_receipt_sha256": "b" * 64},
        "model": {
            "repository": "HuggingFaceTB/SmolLM2-360M",
            "revision": "c" * 40,
            "ordered_files_sha256": "d" * 64,
            "initial_state_sha256": initial,
            "final_state_sha256": initial if name == "unchanged" else name[0] * 64,
        },
        "tokenizer": {"vocabulary_size": 49_152, "eos_token_id": 0},
        "training": {
            "seed": seed,
            "block_size": 512,
            "matched_token_budget": 1_000_000,
            "used_train_tokens": 0 if name == "unchanged" else 1_000_000,
            "selected_stream_sha256": None if name == "unchanged" else name[0] * 64,
            "micro_batch_size": 8,
            "gradient_accumulation": 8,
            "optimizer_steps": 0 if name == "unchanged" else 31,
            "learning_rate": 2e-5,
            "weight_decay": 0.1,
            "warmup_fraction": 0.1,
        },
        "evaluation_streams": {
            "connection_development": {"tokens": 10_000, "sha256": "e" * 64},
            "source_development": {"tokens": 20_000, "sha256": "f" * 64},
        },
        "evaluations": {
            "connection_development": {"mean_nll": connection_nll},
            "source_development": {"mean_nll": source_nll},
        },
        "screen_is_proxy_not_4b_capability_claim": True,
        "transfer_ablation_complete": False,
        "training_ready": False,
    }
    return _signed(payload)


def _fixture(root: Path, failing_seed: int | None = None) -> tuple[Path, Path]:
    screen = root / "screen.json"
    screen.write_text(json.dumps(_screen(), sort_keys=True) + "\n")
    arms = root / "arms"
    for seed in CONFIRMATION_SEEDS:
        seed_root = arms / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        values = (
            ("unchanged", 4.0, 3.0),
            ("source_control", 3.9, 2.99),
            ("connections", 4.1 if seed == failing_seed else 3.8, 3.0),
        )
        for name, connection_nll, source_nll in values:
            (seed_root / f"{name}.json").write_text(
                json.dumps(_arm(name, seed, connection_nll, source_nll), sort_keys=True)
                + "\n"
            )
    return screen, arms


def test_all_fresh_seeds_must_pass_before_admission(tmp_path: Path) -> None:
    screen, arms = _fixture(tmp_path)
    result = aggregate(screen, arms, tmp_path / "aggregate.json")
    assert result["confirmation_pass"] is True
    assert result["connection_component_admission_authorized"] is True
    assert result["transfer_ablation_complete"] is True
    assert result["training_ready"] is False
    assert result["four_b_training_authorized"] is False
    assert all(value["pass"] for value in result["per_seed"].values())


def test_one_failed_seed_blocks_connection_admission(tmp_path: Path) -> None:
    screen, arms = _fixture(tmp_path, failing_seed=CONFIRMATION_SEEDS[-1])
    result = aggregate(screen, arms, tmp_path / "aggregate.json")
    assert result["confirmation_pass"] is False
    assert result["connection_component_admission_authorized"] is False


def test_seed_identity_tamper_fails_closed(tmp_path: Path) -> None:
    screen, arms = _fixture(tmp_path)
    path = arms / f"seed_{CONFIRMATION_SEEDS[0]}" / "connections.json"
    payload = json.loads(path.read_text())
    payload["training"]["seed"] += 1
    payload.pop("receipt_sha256")
    path.write_text(json.dumps(_signed(payload), sort_keys=True) + "\n")
    with pytest.raises(BridgeTransferConfirmationError, match="identity differs"):
        aggregate(screen, arms, tmp_path / "aggregate.json")
