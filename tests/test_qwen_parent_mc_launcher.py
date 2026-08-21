from pathlib import Path


def test_parent_mc_launcher_submits_nine_independent_single_h100_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "jobs/sai-launch-qwen35-0p8b-development-mc-cpu.sbatch"
    ).read_text()
    evaluator = (
        root / "jobs/sai-qwen35-0p8b-development-mc-single-h100.sbatch"
    ).read_text()
    merge = (root / "jobs/sai-merge-development-mc-shards-cpu.sbatch").read_text()
    assert "--shard-count 8" in launcher
    assert 'test "${#mmlu_jobs[@]}" = 8' in launcher
    assert '"h100_jobs": 9' in launcher
    assert "cleanup_partial_submission" in launcher
    assert '--dependency="afterok:$mmlu_dependency"' in launcher
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in evaluator
    assert "#SBATCH --gres" not in launcher
    assert "#SBATCH --gres" not in merge
    assert "#SBATCH --no-requeue" in launcher
    assert "#SBATCH --no-requeue" in merge
    assert "retry" not in launcher.lower()
