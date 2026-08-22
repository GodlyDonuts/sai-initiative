from pathlib import Path


def test_numeric_tokenizer_ablation_job_is_cpu_only_and_bounded() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-numeric-tokenizer-ablation-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "--vocab-size 48000" in job
    assert 'sha256sum "$CORPUS"' in job
    assert 'sha256sum "$PROTECTED_SUITE"' in job
    assert 'test ! -e "$OUTPUT_ROOT"' in job
    assert "sai.tokenizer.numeric_pretokenization" in job
    assert "4b" not in job.lower()
