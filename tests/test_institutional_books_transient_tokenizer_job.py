from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_book_tokenizer_sample_job_is_private_bounded_and_cpu_only() -> None:
    job = (
        ROOT / "scripts" / "sample_institutional_books_tokenizer_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-63%16" in job
    assert "#SBATCH --cpus-per-task=2" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "set -euo pipefail" in job
    assert "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}" in job
    assert "--maximum-utf8-bytes 32000000" in job
    assert job.count("institutional_books_transient_tokenizer_stream") == 1
    assert job.count("transient_tokenizer_sample") == 1
    assert "| python" in job
    assert "huggingface" not in job.casefold()


def test_book_tokenizer_sample_aggregate_is_exact_and_cpu_only() -> None:
    job = (
        ROOT
        / "scripts"
        / "aggregate_institutional_books_tokenizer_samples_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --cpus-per-task=1" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "--logical-shards 64" in job
    assert "--maximum-bytes-per-shard 32000000" in job
    assert "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}" in job
