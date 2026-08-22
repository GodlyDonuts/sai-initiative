from pathlib import Path


def test_curriculum_job_is_cpu_only_dependency_bound_and_create_only() -> None:
    job = (
        Path(__file__).parents[1] / "jobs" / "sai-build-500m-curriculum-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres=" not in job
    assert "#SBATCH --no-requeue" in job
    assert "CPU curriculum builder was exposed to a GPU" in job
    assert 'test "$state" = "COMPLETED"' in job
    assert 'test ! -e "$CURRICULUM_OUTPUT"' in job
    assert 'test ! -e "$CURRICULUM_RECEIPT"' in job
    assert "sai.data.curriculum build" in job
    assert "sai.data.curriculum validate" in job
    assert "--minimum-documents-per-band 10000" in job
    assert '--workers "$SLURM_CPUS_PER_TASK"' in job
    assert (
        "sbatch "
        not in "\n".join(
            line for line in job.splitlines() if not line.startswith("#SBATCH")
        ).lower()
    )
    assert "scancel" not in job.lower()
