from pathlib import Path


def test_learnability_score_job_is_one_h100_create_only_and_no_training() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs" / "sai-score-learnability-single-h100.sbatch").read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --array" not in job
    assert "torchrun" not in job
    assert "sbatch " not in job
    assert '--scale "$SCALE"' in job
    assert "100m|300m|1b" in job
    assert "4b" not in job
    assert 'test ! -e "$OUTPUT"' in job
    assert 'gpu_count" = 1' in job
    assert "NVIDIA H100 PCIe" in job
    assert "--weak-milestone" in job
    assert "--strong-checkpoint" in job
    assert "--probe-training-stream" in job
    assert "--target-stream" in job
    assert "-m sai.data.learnability_score" in job
