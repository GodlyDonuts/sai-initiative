from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_comparison_release_binds_six_terminal_result_jobs() -> None:
    job = (
        ROOT / "jobs" / "sai-release-qwen35-0p8b-workspace-comparison-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "sai-qwen35-0p8b-development-mc-dispatch-v1" in job
    assert "sai-qwen35-0p8b-workspace-development-mc-dispatch-v1" in job
    assert (
        'dependency="$parent_musr:$parent_merge:$recurrent_musr:$recurrent_merge:$reset_musr:$reset_merge"'
        in job
    )
    assert "sai-compare-qwen35-0p8b-workspace-cpu.sbatch" in job
    assert '"gpu_jobs_submitted": 0' in job
    assert '"four_b_training_executed": False' in job


def test_comparison_stage_waits_for_all_three_result_launchers() -> None:
    job = (
        ROOT / "jobs" / "sai-stage-qwen35-0p8b-workspace-comparison-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "sai-qwen35-0p8b-workspace-development-mc-stage-v1" in job
    assert 'value["evaluation_launcher_jobs"]["recurrent"]' in job
    assert 'value["evaluation_launcher_jobs"]["reset_average"]' in job
    assert (
        '--dependency="afterok:$PARENT_LAUNCHER_JOB_ID:$recurrent_launcher:$reset_launcher"'
        in job
    )
    assert "sai-release-qwen35-0p8b-workspace-comparison-cpu.sbatch" in job
