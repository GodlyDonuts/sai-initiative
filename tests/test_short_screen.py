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
    update_micro_batch_sizes,
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


def test_exact_global_update_partition_preserves_the_partial_final_update() -> None:
    assert (
        update_micro_batch_sizes(
            global_step=1,
            training_sequences=48_828,
            sequences_per_update=256,
            micro_batch_size=8,
        )
        == (8,) * 32
    )
    assert update_micro_batch_sizes(
        global_step=191,
        training_sequences=48_828,
        sequences_per_update=256,
        micro_batch_size=8,
    ) == (8,) * 23 + (4,)
    with pytest.raises(ShortScreenError, match="update sequence geometry"):
        update_micro_batch_sizes(
            global_step=192,
            training_sequences=48_828,
            sequences_per_update=256,
            micro_batch_size=8,
        )


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
        micro_batch_size=1,
        sequences_per_update=2,
        training_sequences=5,
        training_utf8_bytes=100,
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
        micro_batch_size=1,
        sequences_per_update=2,
        training_sequences=5,
        training_utf8_bytes=100,
        development_sequences=2,
    )
    assert first.run_sha256 != changed.run_sha256
    assert receipt["delta_backend"] == "fla"
    assert receipt["four_b_training_authorized"] is False
    assert receipt["scientific_promotion_authorized"] is False

    milestone_bindings, milestone_receipt = make_bindings(
        config=config,
        family="kda_mla_hybrid",
        seed=7,
        train_identity_sha256="1" * 64,
        development_identity_sha256="2" * 64,
        code_sha256="3" * 64,
        environment_sha256="4" * 64,
        optimizer=optimizer,
        micro_batch_size=1,
        sequences_per_update=2,
        training_sequences=5,
        training_utf8_bytes=100,
        development_sequences=2,
        milestone_steps=(1, 2),
    )
    assert milestone_receipt["milestone_steps"] == [1, 2]
    assert milestone_bindings.run_sha256 != first.run_sha256


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
            micro_batch_size=1,
            sequences_per_update=2,
            training_sequences=5,
            training_utf8_bytes=100,
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
    assert "TRAINING_SEQUENCES" in job
    assert "SEQUENCES_PER_UPDATE" in job
    assert "ENVIRONMENT_RECEIPT" in job
    assert '"production_cuda_qualified"] is True' in job
    assert 'git -C "$SAI_ROOT" archive --format=tar "$EXPECTED_COMMIT"' in job
    assert 'sha256sum "$ENVIRONMENT_RECEIPT"' in job
    assert 'if [[ -n "${MIN_QUOTA_HEADROOM_KIB:-}"' in job
    assert "/usr/bin/lfs quota" in job
    assert 'test "$quota_headroom_kib" -ge "$MIN_QUOTA_HEADROOM_KIB"' in job
    assert 'test "$quota_headroom_files" -ge "$MIN_QUOTA_HEADROOM_FILES"' in job
    assert "MECHANICS_ONLY" in job
    assert "MILESTONE_STEPS" in job
    assert "retry" not in job.lower()
    assert "4b" not in job.lower()


def test_launcher_submits_three_canaries_then_three_independent_screens() -> None:
    launcher = (ROOT / "jobs" / "sai-launch-100m-short-screens-cpu.sbatch").read_text()
    assert "#SBATCH --gres" not in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "for family in gated_gqa gdn_hybrid kda_mla_hybrid" in launcher
    assert 'gpu_jobs_submitted": 6' in launcher
    assert 'maximum_concurrent_gpu_jobs": 3' in launcher
    assert "MECHANICS_ONLY=1" in launcher
    assert '"${LOG_ROOT:?LOG_ROOT is required}"' in launcher
    assert '--output="$LOG_ROOT/sai_short_screen_%j.out"' in launcher
    assert '--error="$LOG_ROOT/sai_short_screen_%j.err"' in launcher
    assert '--dependency="afterok:$canary_job_id"' in launcher
    assert "trap cancel_partial_graph ERR" in launcher
    assert "scancel" in launcher
    assert "TRAINING_SEQUENCES=48828" in launcher
    assert "SEQUENCES_PER_UPDATE=256" in launcher
    assert "OPTIMIZER_STEPS=191" in launcher
    assert "DEVELOPMENT_SEQUENCES=1024" in launcher
    assert "DEVELOPMENT_SEQUENCES=4096" not in launcher
    assert "--array" not in launcher
    assert "retry" not in launcher.lower()


def test_250m_token_launcher_is_matched_independent_and_fail_closed() -> None:
    launcher = (
        ROOT / "jobs" / "sai-launch-100m-250m-token-screens-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in launcher
    assert "#SBATCH --no-requeue" in launcher
    assert "SCREEN_SCOPE:?SCREEN_SCOPE is required" in launcher
    assert "gqa_only)" in launcher
    assert "three_family)" in launcher
    assert "families=(gated_gqa)" in launcher
    assert "families=(gated_gqa gdn_hybrid kda_mla_hybrid)" in launcher
    assert (
        'comparison_class="single_family_reference_baseline_not_tournament"' in launcher
    )
    assert "FULL_MODEL_PARITY_RECEIPT" in launcher
    assert "FULL_MODEL_PARITY_SHA256" in launcher
    assert 'receipt["status"] == "production_cuda_qualified"' in launcher
    assert 'for family in "${families[@]}"' in launcher
    assert (
        '"canary_job_ids": [int(value) for value in sys.argv[10].split()]' in launcher
    )
    assert '"full_job_ids": [int(value) for value in sys.argv[11].split()]' in launcher
    assert (
        '"gpu_jobs_submitted": len(sys.argv[10].split()) + len(sys.argv[11].split())'
        in launcher
    )
    assert '"maximum_concurrent_gpu_jobs": len(sys.argv[11].split())' in launcher
    assert '"training_tokens": 249_999_360' in launcher
    assert '"training_sequences": 122_070' in launcher
    assert '"optimizer_steps": 477' in launcher
    assert 'train["prefix_utf8_bytes"]["122070"] > 0' in launcher
    assert 'train["prefix_utf8_bytes"]["256"] > 0' in launcher
    assert "TRAINING_SEQUENCES=122070" in launcher
    assert "OPTIMIZER_STEPS=477" in launcher
    assert "DEVELOPMENT_SEQUENCES=1024" in launcher
    assert "SEED=20260821" in launcher
    assert "MECHANICS_ONLY=1" in launcher
    assert "TRAINING_SEQUENCES=256" in launcher
    assert '--dependency="afterok:$canary_job_id"' in launcher
    assert "trap cancel_partial_graph ERR" in launcher
    assert "scancel" in launcher
    assert "--array" not in launcher
    assert "4B" not in launcher
    assert "retry" not in launcher.lower()
