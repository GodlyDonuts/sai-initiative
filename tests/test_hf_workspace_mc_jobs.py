from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_workspace_mc_worker_is_independent_single_h100() -> None:
    job = (
        ROOT / "jobs" / "sai-qwen35-0p8b-workspace-development-mc-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --array" not in job
    assert "evc50" in job
    assert "TRAINING_RESULT_SHA256" in job
    assert 'test -f "$CHECKPOINT.manifest.json"' in job
    assert "sai.evaluation.hf_workspace_mc" in job


def test_arm_launcher_fans_out_full_boards_as_single_h100_jobs() -> None:
    job = (
        ROOT / "jobs" / "sai-launch-qwen35-0p8b-workspace-development-mc-arm-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "--shard-count 8" in job
    assert 'test "${#mmlu_jobs[@]}" = 8' in job
    assert 'musr_job="$(env -u SLURM_OVERLAP' in job
    assert 'scancel "${admitted_jobs[@]}"' in job
    assert '"h100_jobs": 9' in job
    assert 'case "$STATE_MODE" in recurrent|reset_average)' in job
    assert "_load_result" in job


def test_stage_job_replays_training_dispatch_and_stages_both_arms() -> None:
    job = (
        ROOT / "jobs" / "sai-stage-qwen35-0p8b-workspace-development-mc-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "sai-qwen35-0p8b-matched-workspace-dispatch-v1" in job
    assert 'payload["full_jobs"]["recurrent"]' in job
    assert 'payload["full_jobs"]["reset_average"]' in job
    assert 'submit_arm_launcher recurrent "$recurrent_job"' in job
    assert 'submit_arm_launcher reset_average "$reset_job"' in job
    assert '"eventual_h100_jobs": 18' in job
    assert '"one_h100_per_job": True' in job
    assert 'scancel "${admitted_jobs[@]}"' in job
