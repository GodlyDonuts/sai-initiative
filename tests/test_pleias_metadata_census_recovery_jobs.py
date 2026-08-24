from pathlib import Path


def test_recovery_jobs_are_segmented_cpu_only_and_nonrequeueing() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts"
    segment = (
        root / "recover_pleias_metadata_census_segment_stokes.sbatch"
    ).read_text()
    merge = (root / "merge_pleias_metadata_census_recovery_stokes.sbatch").read_text()
    assert "#SBATCH --array=0-7%8" in segment
    assert "#SBATCH --exclude=ec65" in segment
    assert "--segments-per-shard 8" in segment
    assert '--segment-index "${SLURM_ARRAY_TASK_ID}"' in segment
    assert "SAI_SHARD_INDEX is required" in segment
    assert "#SBATCH --no-requeue" in segment
    assert "#SBATCH --gres=" not in segment
    assert "merge-segments" in merge
    assert "--segments-per-shard 8" in merge
    assert "#SBATCH --no-requeue" in merge
    assert "#SBATCH --gres=" not in merge
