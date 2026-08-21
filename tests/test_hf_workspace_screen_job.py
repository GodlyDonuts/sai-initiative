from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_workspace_screen_job_is_one_h100_and_sub_4b() -> None:
    job = (
        ROOT / "jobs" / "sai-qwen35-0p8b-workspace-screen-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --array" not in job
    assert "scancel" not in job
    assert "srun" not in job
    assert "--state-mode" in job
    assert "--training-sequences" in job
    assert 'case "$TRAINING_SEQUENCES" in 256|61035)' in job
    assert "4b" not in job.lower()
    assert "evc50" in job


def test_workspace_screen_job_binds_exact_evidence() -> None:
    job = (
        ROOT / "jobs" / "sai-qwen35-0p8b-workspace-screen-single-h100.sbatch"
    ).read_text()
    for field in (
        "EXPECTED_COMMIT",
        "MECHANICS_RECEIPT_SHA256",
        "TRAIN_IDENTITY",
        "CODE_SHA256",
        "ENVIRONMENT_SHA256",
        "CHECKPOINT",
        "OUTPUT",
    ):
        assert f"${{{field}" in job
    assert 'git -C "$SAI_ROOT" status --short' in job
    assert 'sha256sum "$MECHANICS_RECEIPT"' in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
