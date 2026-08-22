from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_smol_workspace_arm_launcher_fans_out_full_boards() -> None:
    job = (
        ROOT / "jobs" / "sai-launch-smollm3-3b-workspace-development-mc-arm-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in job
    assert 'case "$STATE_MODE" in recurrent|reset_average)' in job
    assert "MODEL_MANIFEST" in job
    assert "RESTORATION_RECEIPT" in job
    assert "hf_smol_workspace_mc import _load_result" in job
    assert "--shard-count 8" in job
    assert 'test "${#mmlu_jobs[@]}" = 8' in job
    assert 'musr_job="$(env -u SLURM_OVERLAP' in job
    assert "sai-smollm3-3b-workspace-development-mc-single-h100.sbatch" in job
    assert 'scancel "${admitted_jobs[@]}"' in job
    assert '"h100_jobs": 9' in job
    assert '"maximum_concurrent_h100_jobs": 9' in job
    assert '"cross_family_confirmation": True' in job
    assert '"four_b_training_executed": False' in job


def test_smol_workspace_stage_replays_training_and_launches_both_arms() -> None:
    job = (
        ROOT / "jobs" / "sai-stage-smollm3-3b-workspace-development-mc-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in job
    assert "sai-smollm3-3b-matched-workspace-dispatch-v1" in job
    assert 'payload["full_jobs"]["recurrent"]' in job
    assert 'payload["full_jobs"]["reset_average"]' in job
    assert 'submit_arm_launcher recurrent "$recurrent_job"' in job
    assert 'submit_arm_launcher reset_average "$reset_job"' in job
    assert "MODEL_MANIFEST" in job
    assert "RESTORATION_RECEIPT" in job
    assert 'scancel "${admitted_jobs[@]}"' in job
    assert '"eventual_h100_jobs": 18' in job
    assert '"one_h100_per_job": True' in job
    assert '"cross_family_confirmation": True' in job
    assert '"four_b_training_executed": False' in job
