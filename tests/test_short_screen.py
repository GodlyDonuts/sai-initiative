from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.model.config import SaiModelConfig, parameter_ledger
from sai.training.runner import TrainingRunConfig
from sai.training.short_screen import (
    ShortScreenError,
    load_bounded_config,
    make_bindings,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(*, hidden_size: int = 16, vocab_size: int = 64) -> SaiModelConfig:
    return SaiModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        mixer_family="kda_mla_hybrid",
        mla_kv_rank=8,
        mla_qk_head_dim=8,
        mla_value_head_dim=8,
    )


def _geometry(path: Path, config: SaiModelConfig) -> Path:
    path.write_text(
        json.dumps(
            {
                "geometries": [
                    {
                        "scale": "100m",
                        "mixer_family": config.mixer_family,
                        "target_parameters": 100_000_000,
                        "config": config.as_dict(),
                        "parameter_ledger": parameter_ledger(config),
                    }
                ]
            }
        )
    )
    return path


def test_loads_exact_frozen_100m_geometry_and_rejects_large_relabel(
    tmp_path: Path,
) -> None:
    config, row = load_bounded_config(
        _geometry(tmp_path / "small.json", _config()), "kda_mla_hybrid"
    )
    assert config == _config()
    assert row["parameter_ledger"]["total"] < 101_000_000

    oversized = _config(hidden_size=8_000, vocab_size=48_000)
    with pytest.raises(ShortScreenError, match="frozen 100M geometry"):
        load_bounded_config(
            _geometry(tmp_path / "large.json", oversized), "kda_mla_hybrid"
        )


def test_all_three_checked_in_100m_comparison_rows_are_admitted() -> None:
    geometry = ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json"
    expected = {
        "gated_gqa": 100_481_024,
        "gdn_hybrid": 100_019_648,
        "kda_mla_hybrid": 99_594_248,
    }
    for family, parameter_count in expected.items():
        _, row = load_bounded_config(geometry, family)
        assert row["parameter_ledger"]["total"] == parameter_count


def test_run_binding_changes_with_every_scientific_identity() -> None:
    config = _config()
    optimizer = TrainingRunConfig(optimizer_steps=3)
    first, receipt = make_bindings(
        config=config,
        family="kda_mla_hybrid",
        seed=7,
        train_identity_sha256="1" * 64,
        development_identity_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        optimizer=optimizer,
        batch_size=1,
        development_sequences=2,
    )
    changed, _ = make_bindings(
        config=config,
        family="kda_mla_hybrid",
        seed=8,
        train_identity_sha256="1" * 64,
        development_identity_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        optimizer=optimizer,
        batch_size=1,
        development_sequences=2,
    )
    assert first.run_sha256 != changed.run_sha256
    assert receipt["delta_backend"] == "fla"
    assert receipt["four_b_training_authorized"] is False
    assert receipt["scientific_promotion_authorized"] is False


def test_streams_must_be_source_disjoint_and_hashes_exact() -> None:
    config = _config()
    with pytest.raises(ShortScreenError, match="streams must differ"):
        make_bindings(
            config=config,
            family="kda_mla_hybrid",
            seed=7,
            train_identity_sha256="1" * 64,
            development_identity_sha256="1" * 64,
            code_sha256="3" * 64,
            environment_sha256="4" * 64,
            optimizer=TrainingRunConfig(optimizer_steps=1),
            batch_size=1,
            development_sequences=1,
        )


def test_job_is_one_h100_no_requeue_and_has_no_retry_or_4b() -> None:
    job = (ROOT / "jobs" / "sai-short-screen-single-h100.sbatch").read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "--no-requeue" in job
    assert 'gpu_count" = 1' in job
    assert "EXPECTED_COMMIT" in job
    assert "TRAIN_IDENTITY" in job
    assert "DEVELOPMENT_IDENTITY" in job
    assert "retry" not in job.lower()
    assert "4b" not in job.lower()
