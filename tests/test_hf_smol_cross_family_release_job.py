from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_cross_family_release_requires_qwen_pass_and_stages_complete_graph() -> None:
    job = (
        ROOT / "jobs" / "sai-release-smollm3-3b-after-qwen-pass-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in job
    assert "_load_qwen_factor_receipt" in job
    assert "QWEN_COMPARISON_JOB_ID" in job
    assert "sai-smollm3-3b-mechanics-single-h100.sbatch" in job
    assert "sai-launch-smollm3-3b-development-mc-cpu.sbatch" in job
    assert "sai-launch-smollm3-3b-workspace-screen-cpu.sbatch" in job
    assert "sai-stage-smollm3-3b-workspace-development-mc-cpu.sbatch" in job
    assert "sai-stage-smollm3-3b-workspace-comparison-cpu.sbatch" in job
    assert 'scancel "${admitted_jobs[@]}"' in job
    assert '"eventual_h100_jobs": 32' in job
    assert '"maximum_concurrent_h100_jobs": 27' in job
    assert '"one_h100_per_job": True' in job
    assert '"four_b_training_executed": False' in job
    assert '"four_b_training_authorized": False' in job
