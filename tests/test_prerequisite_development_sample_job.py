from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_development_audit_job_is_cpu_only_bounded_and_nonretrying() -> None:
    job = (
        ROOT / "jobs" / "sai-select-prerequisite-development-audit-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --time=00:30:00" in job
    assert "#SBATCH --cpus-per-task=2" in job
    assert "sai.data.prerequisite_development_sample build" in job
    assert "sai.data.prerequisite_development_sample validate" in job
    assert "--per-stratum 8" in job
    assert "SLURM_TMPDIR" not in job
