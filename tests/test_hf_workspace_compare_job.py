from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_workspace_comparator_is_cpu_only_and_terminally_bounded() -> None:
    job = (ROOT / "jobs" / "sai-compare-qwen35-0p8b-workspace-cpu.sbatch").read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in job
    for argument in (
        "--parent-mmlu-pro",
        "--parent-musr",
        "--recurrent-mmlu-pro",
        "--recurrent-musr",
        "--reset-mmlu-pro",
        "--reset-musr",
        "--recurrent-training-result",
        "--reset-training-result",
    ):
        assert argument in job
    assert "sbatch" not in job
    assert "scancel" not in job
    assert "4B" not in job
