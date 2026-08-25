from __future__ import annotations

from pathlib import Path


def test_unique_token_ledger_job_is_complete_and_cpu_only() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "scripts/ledger_one_b_unique_tokens_stokes.sbatch").read_text()
    assert "--gres=gpu" not in job
    assert "one_b_unique_token_ledger" in job
    source = (root / "src/sai/data/one_b_unique_token_ledger.py").read_text()
    assert "source_token_estimates_are_not_production_48k_counts" in source
