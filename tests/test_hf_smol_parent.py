from __future__ import annotations

from pathlib import Path

from sai.evaluation.hf_smol_parent import (
    EXPECTED_MODEL_CLASS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_VOCAB_SIZE,
    SNAPSHOT_SPEC,
)


def test_smol_parent_identity_is_exact() -> None:
    assert EXPECTED_MODEL_CLASS == "SmolLM3ForCausalLM"
    assert EXPECTED_PARAMETER_COUNT == 3_075_098_624
    assert EXPECTED_VOCAB_SIZE == 128_256
    assert SNAPSHOT_SPEC.repository == "HuggingFaceTB/SmolLM3-3B"
    assert SNAPSHOT_SPEC.revision == "a07cc9a04f16550a088caea529712d1d335b0ac1"
    assert SNAPSHOT_SPEC.tree_sha256 == (
        "6badcd593aee3052e3d66afb315b979e2cc62c4a61f9cef31c07203912478a0f"
    )


def test_smol_mechanics_is_one_h100_offline_and_no_training() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-smollm3-3b-mechanics-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "evc50" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
    assert "sai.evaluation.hf_smol_parent" in job
    assert "retry" not in job.lower()
    assert "train" not in job.lower().replace("training", "")
