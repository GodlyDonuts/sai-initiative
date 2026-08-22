from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.evaluation.scale_checkpoint import (
    EVALUATION_SCALES,
    FAMILIES,
    ScaleCheckpointError,
    load_evaluation_config,
)
from sai.evaluation.short_screen_mc import validate_short_screen_result
from sai.model.config import SaiModelConfig, parameter_ledger
from sai.model.planner import build_plan
from sai.training.runner import TrainingRunConfig
from sai.training.short_screen import make_bindings


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(build_plan(48_000), sort_keys=True) + "\n")
    return path


@pytest.mark.parametrize("scale", EVALUATION_SCALES)
@pytest.mark.parametrize("family", FAMILIES)
def test_resolves_exact_non_4b_scale_geometry(
    tmp_path: Path, scale: str, family: str
) -> None:
    config, row = load_evaluation_config(_plan(tmp_path), family, scale)
    assert row["scale"] == scale
    assert row["mixer_family"] == family
    assert config.as_dict() == row["config"]
    assert abs(row["relative_error"]) <= 0.01


@pytest.mark.parametrize("scale", ["4b", "250m", "", "1B"])
def test_refuses_4b_and_noncanonical_scale_labels(tmp_path: Path, scale: str) -> None:
    with pytest.raises(ScaleCheckpointError, match="100m, 300m, or 1b"):
        load_evaluation_config(_plan(tmp_path), "gated_gqa", scale)


def test_rejects_resigned_or_byte_modified_geometry_plan(tmp_path: Path) -> None:
    path = _plan(tmp_path)
    payload = json.loads(path.read_text())
    payload["geometries"][3]["config"]["intermediate_size"] += 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ScaleCheckpointError, match="deterministic geometry plan"):
        load_evaluation_config(path, "gated_gqa", "300m")


def test_300m_terminal_receipt_reconstructs_scale_promotion_binding(
    tmp_path: Path,
) -> None:
    config = SaiModelConfig(
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
    bindings, specification = make_bindings(
        config=config,
        family="gated_gqa",
        seed=17,
        train_identity_sha256="1" * 64,
        development_identity_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        optimizer=TrainingRunConfig(optimizer_steps=1),
        micro_batch_size=1,
        sequences_per_update=1,
        training_sequences=1,
        training_utf8_bytes=8,
        development_sequences=1,
        scale="300m",
        promotion_receipt_sha256="5" * 64,
    )
    result = {
        **specification,
        "status": "complete",
        "parameter_count": parameter_ledger(config)["total"],
    }
    result["receipt_sha256"] = canonical_sha256(result)
    path = tmp_path / "300m-result.json"
    path.write_text(json.dumps(result, sort_keys=True) + "\n")
    observed, observed_bindings = validate_short_screen_result(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config=config,
        family="gated_gqa",
        geometry_parameter_count=parameter_ledger(config)["total"],
        scale="300m",
    )
    assert observed["promotion_receipt_sha256"] == "5" * 64
    assert observed_bindings == bindings


def test_single_h100_adapter_defaults_to_100m_and_allows_only_non_4b() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "jobs" / "sai-short-screen-development-mc-single-h100.sbatch"
    ).read_text()
    assert 'SCALE="${SCALE:-100m}"' in job
    assert "100m|300m|1b" in job
    assert '--scale "$SCALE"' in job
    assert "4b)" not in job.lower()
