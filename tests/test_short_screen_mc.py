from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sai.data.token_stream import canonical_sha256
from sai.evaluation.short_screen_mc import (
    ShortScreenMCError,
    _state_sha256,
    load_validated_model_state,
    validate_short_screen_result,
)
from sai.model.config import SaiModelConfig, parameter_ledger
from sai.training.checkpoint import (
    CheckpointBindings,
    TrainingCounters,
    checkpoint_manifest_path,
    save_mechanics_checkpoint,
)
from sai.training.runner import TrainingRunConfig
from sai.training.short_screen import make_bindings
from sai.training.stream import StreamCursor


def _bindings() -> CheckpointBindings:
    return CheckpointBindings(
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        ordered_stream_identity_sha256="3" * 64,
        code_sha256="4" * 64,
        environment_sha256="5" * 64,
        run_sha256="6" * 64,
    )


def _screen_config() -> SaiModelConfig:
    return SaiModelConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        mixer_family="gated_gqa",
        mla_kv_rank=4,
        mla_qk_head_dim=4,
        mla_value_head_dim=4,
    )


def _checkpoint(tmp_path: Path) -> tuple[Path, nn.Module, dict, CheckpointBindings]:
    bindings = _bindings()
    source = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters())
    checkpoint = tmp_path / "screen.pt"
    manifest = save_mechanics_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        bindings=bindings,
        counters=TrainingCounters(optimizer_steps=1, sequences=2, targets=3),
        cursor=StreamCursor(bindings.ordered_stream_identity_sha256, 2),
    )
    return checkpoint, source, manifest, bindings


def test_read_only_loader_restores_only_validated_model_state(tmp_path: Path) -> None:
    checkpoint, source, manifest, bindings = _checkpoint(tmp_path)
    target = nn.Linear(3, 2)
    observation = load_validated_model_state(
        checkpoint,
        checkpoint_manifest_path(checkpoint),
        model=target,
        expected_bindings=bindings,
        expected_descriptor=manifest["checkpoint"],
        expected_counters=manifest["counters"],
        expected_cursor=manifest["cursor"],
        expected_final_state_sha256=_state_sha256(source),
    )
    assert observation["checkpoint_sha256"] == manifest["checkpoint"]["sha256"]
    assert observation["final_state_sha256"] == _state_sha256(source)
    assert all(
        torch.equal(target.state_dict()[name], tensor)
        for name, tensor in source.state_dict().items()
    )


def test_terminal_result_reconstructs_exact_run_and_model_identities(
    tmp_path: Path,
) -> None:
    config = _screen_config()
    optimizer = TrainingRunConfig(optimizer_steps=1)
    bindings, specification = make_bindings(
        config=config,
        family="gated_gqa",
        seed=17,
        train_identity_sha256="1" * 64,
        development_identity_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        optimizer=optimizer,
        micro_batch_size=1,
        sequences_per_update=1,
        training_sequences=1,
        training_utf8_bytes=8,
        development_sequences=1,
    )
    result = {
        **specification,
        "status": "complete",
        "parameter_count": parameter_ledger(config)["total"],
    }
    result["receipt_sha256"] = canonical_sha256(result)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result, sort_keys=True) + "\n")
    observed, observed_bindings = validate_short_screen_result(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config=config,
        family="gated_gqa",
        geometry_parameter_count=parameter_ledger(config)["total"],
    )
    assert observed["run_sha256"] == specification["run_sha256"]
    assert observed_bindings == bindings


def test_loader_rejects_manifest_or_final_state_tampering(tmp_path: Path) -> None:
    checkpoint, source, manifest, bindings = _checkpoint(tmp_path)
    sidecar = checkpoint_manifest_path(checkpoint)
    payload = json.loads(sidecar.read_text())
    payload["bindings"]["run_sha256"] = "7" * 64
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ShortScreenMCError, match="lineage"):
        load_validated_model_state(
            checkpoint,
            sidecar,
            model=nn.Linear(3, 2),
            expected_bindings=bindings,
            expected_descriptor=manifest["checkpoint"],
            expected_counters=manifest["counters"],
            expected_cursor=manifest["cursor"],
            expected_final_state_sha256=_state_sha256(source),
        )

    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ShortScreenMCError, match="final model state"):
        load_validated_model_state(
            checkpoint,
            sidecar,
            model=nn.Linear(3, 2),
            expected_bindings=bindings,
            expected_descriptor=manifest["checkpoint"],
            expected_counters=manifest["counters"],
            expected_cursor=manifest["cursor"],
            expected_final_state_sha256="8" * 64,
        )


def test_single_h100_job_is_read_only_and_requires_absolute_submit_logs() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "jobs" / "sai-short-screen-development-mc-single-h100.sbatch"
    ).read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "--no-requeue" in job
    assert 'gpu_count" = 1' in job
    assert 'gpu_name" = "NVIDIA H100 PCIe"' in job
    assert "StdOut=" in job and '"$LOG_ROOT"/*' in job
    assert "sai.evaluation.short_screen_mc" in job
    assert "backward" not in job
    assert "optimizer" not in job.lower()
