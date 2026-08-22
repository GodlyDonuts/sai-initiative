from pathlib import Path


def test_curriculum_control_job_is_cpu_only_exact_and_dependency_bound() -> None:
    job = (
        Path(__file__).parents[1]
        / "jobs"
        / "sai-build-curriculum-order-control-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres=" not in job
    assert "#SBATCH --no-requeue" in job
    assert "CPU curriculum control was exposed to a GPU" in job
    assert ': "${PARENT_STREAM_JOB_ID:?PARENT_STREAM_JOB_ID is required}"' in job
    assert 'test "$state" = "COMPLETED"' in job
    assert 'test ! -e "$CONTROL_STREAM"' in job
    assert "sai.data.curriculum_control build" in job
    assert "sai.data.curriculum_control validate" in job
    assert "--seed 2026082201" in job
    executable = "\n".join(
        line for line in job.splitlines() if not line.startswith("#SBATCH")
    ).lower()
    assert "sbatch " not in executable
    assert "scancel" not in executable
