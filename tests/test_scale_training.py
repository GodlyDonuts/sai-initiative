from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.model.config import SaiModelConfig, parameter_ledger
from sai.model.planner import build_plan
from sai.training.runner import TrainingRunConfig
from sai.training.scale_training import (
    ADMISSION_SCHEMA,
    ScaleTrainingError,
    build_parser,
    load_scale_admission,
    validate_scale_geometry_plan,
)
from sai.training.short_screen import (
    ShortScreenError,
    load_bounded_config,
    make_bindings,
)

ROOT = Path(__file__).resolve().parents[1]


def _admission(path: Path, *, scale: str, family: str) -> Path:
    payload = {
        "schema": ADMISSION_SCHEMA,
        "target_scale": scale,
        "prior_scale": {"300m": "100m", "1b": "300m"}[scale],
        "selected_family": family,
        "real_development_benchmark_gate_passed": True,
        "matched_equal_compute_control": True,
        "source_disjoint_evaluation": True,
        "evidence_receipt_sha256": "a" * 64,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return path


def _tiny_config() -> SaiModelConfig:
    return SaiModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        mixer_family="gdn_hybrid",
        mla_kv_rank=8,
        mla_qk_head_dim=8,
        mla_value_head_dim=8,
    )


def test_all_frozen_300m_and_1b_rows_are_admitted_without_selecting_family() -> None:
    geometry = ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json"
    expected = {
        "300m": {
            "gated_gqa": 298_786_560,
            "gdn_hybrid": 299_283_072,
            "kda_mla_hybrid": 298_246_872,
        },
        "1b": {
            "gated_gqa": 1_001_012_736,
            "gdn_hybrid": 1_002_005_760,
            "kda_mla_hybrid": 998_163_672,
        },
    }
    for scale, families in expected.items():
        for family, count in families.items():
            config, row = load_bounded_config(geometry, family, scale)
            assert config.mixer_family == family
            assert row["scale"] == scale
            assert row["parameter_ledger"]["total"] == count


def test_scale_loader_rejects_a_small_model_relabelled_as_300m(tmp_path: Path) -> None:
    config = _tiny_config()
    geometry = tmp_path / "relabelled.json"
    geometry.write_text(
        json.dumps(
            {
                "geometries": [
                    {
                        "scale": "300m",
                        "target_parameters": 300_000_000,
                        "mixer_family": config.mixer_family,
                        "config": config.as_dict(),
                        "parameter_ledger": parameter_ledger(config),
                    }
                ]
            }
        )
    )
    with pytest.raises(ShortScreenError, match="scale envelope"):
        load_bounded_config(geometry, "gdn_hybrid", "300m")


def test_scale_training_reopens_the_self_hashed_geometry_plan(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.json"
    payload = build_plan(48_000)
    geometry.write_text(json.dumps(payload, sort_keys=True) + "\n")
    assert validate_scale_geometry_plan(geometry) == payload
    payload["geometries"][3]["config"]["intermediate_size"] += 64
    geometry.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(ScaleTrainingError, match="geometry plan differs"):
        validate_scale_geometry_plan(geometry)


def test_admission_is_exact_hash_bound_and_cannot_cross_scale_or_family(
    tmp_path: Path,
) -> None:
    path = _admission(tmp_path / "admission.json", scale="300m", family="gdn_hybrid")
    receipt = load_scale_admission(path, scale="300m", family="gdn_hybrid")
    assert receipt["real_development_benchmark_gate_passed"] is True

    with pytest.raises(ScaleTrainingError, match="evidence differs"):
        load_scale_admission(path, scale="300m", family="kda_mla_hybrid")
    payload = json.loads(path.read_text())
    payload["source_disjoint_evaluation"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ScaleTrainingError, match="hash differs"):
        load_scale_admission(path, scale="300m", family="gdn_hybrid")


def test_scale_binding_requires_and_binds_the_external_promotion() -> None:
    config = _tiny_config()
    common = {
        "config": config,
        "family": "gdn_hybrid",
        "seed": 7,
        "train_identity_sha256": "1" * 64,
        "development_identity_sha256": "2" * 64,
        "code_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
        "optimizer": TrainingRunConfig(optimizer_steps=1),
        "micro_batch_size": 1,
        "sequences_per_update": 1,
        "training_sequences": 1,
        "training_utf8_bytes": 10,
        "development_sequences": 1,
        "scale": "300m",
    }
    with pytest.raises(ShortScreenError, match="promotion receipt is required"):
        make_bindings(**common)
    bindings, specification = make_bindings(**common, promotion_receipt_sha256="5" * 64)
    assert specification["schema"] == "sai-sub-4b-scale-training-v1"
    assert specification["scale"] == "300m"
    assert specification["promotion_receipt_sha256"] == "5" * 64
    assert specification["four_b_training_authorized"] is False
    assert bindings.run_sha256 == specification["run_sha256"]


def test_generic_cli_and_job_have_no_family_default_or_self_submission() -> None:
    parser = build_parser()
    scale_action = next(action for action in parser._actions if action.dest == "scale")
    family_action = next(
        action for action in parser._actions if action.dest == "family"
    )
    assert tuple(scale_action.choices) == ("300m", "1b")
    assert scale_action.required is True
    assert family_action.required is True
    assert family_action.default is None

    job = (ROOT / "jobs" / "sai-sub4b-scale-training-single-h100.sbatch").read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert ': "${SCALE:?SCALE is required}"' in job
    assert ': "${FAMILY:?FAMILY is required}"' in job
    assert "300m|1b" in job
    assert "FULL_MODEL_PARITY_RECEIPT" in job
    assert 'full["production_cuda_qualified"] is True' in job
    assert "ADMISSION_RECEIPT" in job
    assert "sai.training.scale_training" in job
    executable_lines = "\n".join(
        line for line in job.splitlines() if not line.startswith("#SBATCH")
    )
    assert "sbatch " not in executable_lines.lower()
    assert "scancel" not in job.lower()
    assert "--requeue" not in job
