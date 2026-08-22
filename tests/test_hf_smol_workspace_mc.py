from __future__ import annotations

import hashlib
import json
import math

import pytest

from sai.data.token_stream import canonical_sha256
from sai.evaluation.hf_smol_workspace_mc import (
    HFWorkspaceEvaluationError,
    _load_result,
)
from sai.training.hf_smol_workspace_screen import (
    EXPECTED_WORKSPACE_PARAMETERS,
    SEQUENCES_PER_UPDATE,
    make_bindings,
)
from sai.training.runner import TrainingRunConfig


def _result() -> dict:
    sequences = 61_035
    optimizer = TrainingRunConfig(
        optimizer_steps=math.ceil(sequences / SEQUENCES_PER_UPDATE),
        learning_rate=3e-4,
        warmup_steps=100,
        minimum_learning_rate_ratio=0.1,
        weight_decay=0.1,
        gradient_clip_norm=1.0,
    )
    _, specification = make_bindings(
        state_mode="recurrent",
        snapshot_tree_sha256="1" * 64,
        mechanics_file_sha256="2" * 64,
        stream_identity_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        training_sequences=sequences,
        training_utf8_bytes=123,
        optimizer=optimizer,
        code_sha256="5" * 64,
        environment_sha256="6" * 64,
    )
    payload = {
        **specification,
        "status": "complete",
        "parent_state_unchanged": True,
        "architecture_improvement_demonstrated": False,
        "workspace_initial_state_sha256": "7" * 64,
        "workspace_final_state_sha256": "8" * 64,
        "counters": {
            "optimizer_steps": optimizer.optimizer_steps,
            "sequences": sequences,
            "targets": 1,
        },
        "stream_cursor": {
            "ordered_stream_identity_sha256": "3" * 64,
            "next_sequence": sequences,
        },
        "checkpoint": {"bytes": 1, "sha256": "9" * 64},
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    assert payload["workspace_parameter_count"] == EXPECTED_WORKSPACE_PARAMETERS
    return payload


def test_smol_training_result_reopens_exact_schema(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result(), sort_keys=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    observed = _load_result(path, digest)
    assert observed["training_sequences"] == 61_035
    assert observed["four_b_training_executed"] is False


def test_smol_training_result_rejects_resigned_geometry(tmp_path) -> None:
    payload = _result()
    payload["workspace_parameter_count"] += 1
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(HFWorkspaceEvaluationError, match="evidence differs"):
        _load_result(path, digest)
