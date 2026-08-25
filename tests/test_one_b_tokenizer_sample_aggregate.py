from __future__ import annotations

from pathlib import Path


def test_tokenizer_sample_aggregate_job_is_cpu_only_and_fixed_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "scripts/aggregate_one_b_tokenizer_samples_stokes.sbatch").read_text()
    assert "--gres=gpu" not in job
    assert "one_b_tokenizer_sample_aggregate" in job
    source = (root / "src/sai/data/one_b_tokenizer_sample_aggregate.py").read_text()
    assert '"books": 64' in source
    assert '"pleias": 1' in source
    assert '"code": 1' in source
    assert '"connections": 1' in source
