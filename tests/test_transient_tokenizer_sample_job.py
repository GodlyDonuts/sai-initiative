from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_transient_tokenizer_sample_job_is_bounded_and_dependency_safe() -> None:
    job = (
        ROOT / "scripts" / "sample_pleias_transient_tokenizer_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-127%16" in job
    assert "#SBATCH --cpus-per-task=2" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "set -euo pipefail" in job
    assert "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}" in job
    assert "--maximum-utf8-bytes 64000000" in job
    assert job.count("-m sai.data.pleias_virtual_transient_stream") == 1
    assert job.count("-m sai.data.transient_tokenizer_sample") == 1
    assert "| python" in job
    assert "rm " not in job
