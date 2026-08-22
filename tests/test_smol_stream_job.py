from pathlib import Path


def test_smol_stream_job_is_cpu_only_exact_and_replay_validated() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs/sai-freeze-smollm3-3b-500m-stream-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "validate_external_snapshot" in job
    assert "SNAPSHOT_SPEC" in job
    assert "--prefix-sequences 244140" in job
    assert 'stream["vocab_size"] == 128256' in job
    assert 'stream["eos_token_id"] == 128012' in job
    assert "verify_sources=True" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
    assert "retry" not in job.lower()


def test_smol_125m_stream_contains_only_the_consumed_prefix() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs/sai-freeze-smollm3-3b-125m-stream-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "--prefix-sequences 256" in job
    assert "--prefix-sequences 61035" in job
    assert "--prefix-sequences 122070" not in job
    assert 'stream["sequences"] == 61035' in job
    assert 'stream["valid_tokens"] == 124999680' in job
    assert 'stream["vocab_size"] == 128256' in job
    assert 'stream["eos_token_id"] == 128012' in job
    assert "sha256_file(corpus)" in job
    assert "read_bytes()" not in job
    assert "validate_external_snapshot" in job
    assert "verify_sources=True" in job
