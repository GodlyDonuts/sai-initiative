from pathlib import Path


def test_dispatcher_recovers_only_missing_receipts_and_rewires_aggregate() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dispatch_pleias_metadata_census_recovery_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres=" not in job
    assert "if [[ -f \"${sai_shard}/receipt.json\" ]]" in job
    assert "continue" in job
    assert "mv \"${sai_shard}\" \"${sai_incomplete}\"" in job
    assert "--array=0-7%8" in job
    assert "--dependency=afterok:\"${sai_segment_job}\"" in job
    assert "Dependency=afterok:\"${sai_dependency}\"" in job
    assert "SAI_AGGREGATE_JOB_ID is required" in job
