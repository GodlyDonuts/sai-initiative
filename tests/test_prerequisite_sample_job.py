from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prerequisite_sample_job_is_cpu_only_exact_and_nonretrying() -> None:
    job = (ROOT / "jobs" / "sai-select-prerequisite-audit-cpu.sbatch").read_text()
    assert "#SBATCH --cpus-per-task=8" in job
    assert "#SBATCH --mem=16G" in job
    assert "#SBATCH --time=04:00:00" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres=" not in job
    assert "retry" not in job.lower()
    assert 'rev-parse HEAD)" = "$EXPECTED_COMMIT"' in job
    assert "sai.data.prerequisite_sample build" in job
    assert "sai.data.prerequisite_sample validate" in job
    assert "--per-stratum 8" in job
    assert 'case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles)' in job
