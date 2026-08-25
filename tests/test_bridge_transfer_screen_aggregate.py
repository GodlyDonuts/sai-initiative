from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.bridge_transfer_screen import SCHEMA as ARM_SCHEMA
from sai.data.bridge_transfer_screen_aggregate import (
    BridgeTransferScreenAggregateError,
    aggregate,
)
from sai.data.token_stream import canonical_sha256


def _arm(name: str, connection_nll: float, source_nll: float) -> dict:
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
            "seed": 20_260_826,
            "matched_token_budget": 1_000_000,
            "used_train_tokens": 0 if name == "unchanged" else 1_000_000,
            "optimizer_steps": 0 if name == "unchanged" else 61,
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
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write(root: Path, name: str, payload: dict) -> None:
    (root / f"{name}.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_positive_screen_authorizes_only_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "arms"
    root.mkdir()
    _write(root, "unchanged", _arm("unchanged", 4.0, 3.0))
    _write(root, "source_control", _arm("source_control", 3.9, 2.99))
    _write(root, "connections", _arm("connections", 3.8, 3.0))
    result = aggregate(root, tmp_path / "aggregate.json")
    assert result["screen_pass"] is True
    assert result["multi_seed_confirmation_authorized"] is True
    assert result["connection_component_admission_authorized"] is False
    assert result["transfer_ablation_complete"] is False


def test_regression_fails_screen(tmp_path: Path) -> None:
    root = tmp_path / "arms"
    root.mkdir()
    _write(root, "unchanged", _arm("unchanged", 4.0, 3.0))
    _write(root, "source_control", _arm("source_control", 3.9, 2.9))
    _write(root, "connections", _arm("connections", 4.1, 3.2))
    result = aggregate(root, tmp_path / "aggregate.json")
    assert result["screen_pass"] is False
    assert result["multi_seed_confirmation_authorized"] is False


def test_rejects_unmatched_evaluation_stream(tmp_path: Path) -> None:
    root = tmp_path / "arms"
    root.mkdir()
    for name, connection, source in (
        ("unchanged", 4.0, 3.0),
        ("source_control", 3.9, 2.9),
        ("connections", 3.8, 3.0),
    ):
        payload = _arm(name, connection, source)
        if name == "connections":
            payload["evaluation_streams"]["connection_development"]["sha256"] = "0" * 64
            payload.pop("receipt_sha256")
            payload["receipt_sha256"] = canonical_sha256(payload)
        _write(root, name, payload)
    with pytest.raises(BridgeTransferScreenAggregateError, match="matched identity"):
        aggregate(root, tmp_path / "aggregate.json")
