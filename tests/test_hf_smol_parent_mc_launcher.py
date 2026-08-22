from pathlib import Path


def test_smol_parent_launcher_fans_independent_full_boards() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs/sai-launch-smollm3-3b-development-mc-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "sai-smollm3-3b-development-mc-single-h100.sbatch" in job
    assert "--shard-count 8" in job
    assert 'test "${#mmlu_jobs[@]}" = 8' in job
    assert '"h100_jobs": 9' in job
    assert '"maximum_concurrent_h100_jobs": 9' in job
    assert '"one_h100_per_job": True' in job
    assert "cleanup_partial_submission" in job
    assert "env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch" in job
    assert "retry" not in job.lower()
