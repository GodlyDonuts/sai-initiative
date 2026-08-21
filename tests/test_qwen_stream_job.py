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
    assert 'stream["vocab_size"] == 248320' in job
    assert 'stream["eos_token_id"] == 248044' in job
    assert "verify_sources=True" in job
    assert "retry" not in job.lower()
