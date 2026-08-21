from __future__ import annotations

from pathlib import Path

from sai.evaluation.population_refresh import SCHEMA


def test_refresh_schema_and_cpu_job_contract() -> None:
    assert SCHEMA == "sai-development-mc-populations-refresh-v1"
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-refresh-development-mc-populations-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "DECONTAMINATION_JOB_ID" in job
    assert "sai.evaluation.population_refresh" in job
    assert "retry" not in job.lower()
