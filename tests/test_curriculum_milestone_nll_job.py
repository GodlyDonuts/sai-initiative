from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_milestone_evaluator_is_read_only_single_h100_and_fail_closed() -> None:
    job = (
        ROOT / "jobs" / "sai-curriculum-milestone-nll-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "--exclude=" in job
    assert "sai.evaluation.curriculum_milestone_nll" in job
    assert "--short-screen-result" in job
    assert "--training-stream" in job
    assert "--development-stream" in job
    assert "--checkpoint-manifest" in job
    assert "optimizer" not in job.lower()
    assert "sbatch " not in job
    assert "srun " not in job
    assert "4b" not in job.lower()
