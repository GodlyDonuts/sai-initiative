from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_smol_workspace_mc_worker_is_independent_single_h100() -> None:
    job = (
        ROOT / "jobs" / "sai-smollm3-3b-workspace-development-mc-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --mem=96G" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --array" not in job
    assert "evc50" in job
    assert "MODEL_MANIFEST" in job
    assert "RESTORATION_RECEIPT" in job
    assert "TRAINING_RESULT_SHA256" in job
    assert 'test -f "$CHECKPOINT.manifest.json"' in job
    assert "sai.evaluation.hf_smol_workspace_mc" in job
