from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_smol_comparison_stage_waits_for_parent_and_both_arm_launchers() -> None:
    job = (
        ROOT / "jobs" / "sai-stage-smollm3-3b-workspace-comparison-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "sai-smollm3-3b-workspace-development-mc-stage-v1" in job
    assert "eventual_h100_jobs" in job
    assert "afterok:$PARENT_LAUNCHER_JOB_ID:$recurrent_launcher:$reset_launcher" in job
    assert "sai-release-smollm3-3b-workspace-comparison-cpu.sbatch" in job


def test_smol_comparison_release_replays_dispatches_and_waits_for_results() -> None:
    job = (
        ROOT / "jobs" / "sai-release-smollm3-3b-workspace-comparison-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "sai-smollm3-3b-development-mc-dispatch-v1" in job
    assert "sai-smollm3-3b-workspace-development-mc-dispatch-v1" in job
    assert 'dependency="$parent_musr:$parent_merge:' in job
    assert "sai-compare-smollm3-3b-workspace-cpu.sbatch" in job
    assert "QWEN_FACTOR_RECEIPT" in job
    assert '"gpu_jobs_submitted": 0' in job
    assert '"four_b_training_authorized": False' in job
    assert '"four_b_training_executed": False' in job
