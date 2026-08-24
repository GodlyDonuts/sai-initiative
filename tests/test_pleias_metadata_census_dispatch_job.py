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
    assert "sai-pleias-metadata-census-recovery-dispatch-v1" in job
    assert "existing_dispatch" in job


def test_accelerated_dispatch_terminates_before_segmenting_and_rejoins_graph() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dispatch_pleias_metadata_census_accelerated_subset_stokes.sh"
    ).read_text()
    assert "SAI_SHARD_INDICES is required" in script
    assert 'scancel "${sai_original_job}"' in script
    assert 'squeue -h -j "${sai_original_job}"' in script
    assert "completed_during_cancel" in script
    assert "--array=0-7%8" in script
    assert "sai-pleias-metadata-census-recovery-dispatch-v1" in script
    assert 'Dependency=afterok:"${sai_dependency}"' in script
