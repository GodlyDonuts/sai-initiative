from pathlib import Path


def test_qwen_stream_job_is_cpu_only_exact_and_replay_validated() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs/sai-freeze-qwen35-0p8b-500m-stream-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "validate_snapshot(model_root)" in job
    assert "--prefix-sequences 244140" in job
    assert 'stream["vocab_size"] == 248077' in job
    assert 'stream["eos_token_id"] == 248046' in job
    assert "verify_sources=True" in job
    assert "retry" not in job.lower()


def test_qwen_125m_stream_is_the_exact_consumed_prefix_without_unused_tail() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs/sai-freeze-qwen35-0p8b-125m-stream-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "--prefix-sequences 256" in job
    assert "--prefix-sequences 61035" in job
    assert "--prefix-sequences 122070" not in job
    assert 'stream["sequences"] == 61035' in job
    assert 'stream["valid_tokens"] == 124999680' in job
    assert 'set(stream["prefix_utf8_bytes"]) == {"256", "61035"}' in job
    assert "sha256_file(corpus)" in job
    assert "read_bytes()" not in job
    assert "verify_sources=True" in job
