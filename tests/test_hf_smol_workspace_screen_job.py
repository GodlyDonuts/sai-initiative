from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_smol_workspace_job_is_one_h100_and_sub_4b() -> None:
    job = (
        ROOT / "jobs" / "sai-smollm3-3b-workspace-screen-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --mem=128G" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --array" not in job
    assert "srun" not in job
    assert "evc50" in job
    assert "hf_smol_workspace_screen" in job
    assert 'case "$TRAINING_SEQUENCES" in 256|61035)' in job
    for field in (
        "MODEL_MANIFEST",
        "RESTORATION_RECEIPT",
        "MECHANICS_RECEIPT_SHA256",
        "TRAIN_IDENTITY",
        "CODE_SHA256",
        "ENVIRONMENT_SHA256",
        "CHECKPOINT",
        "OUTPUT",
    ):
        assert f"${{{field}" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
