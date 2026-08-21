from pathlib import Path


def test_fla_parity_job_is_one_h100_no_requeue_and_never_trains() -> None:
    script = (
        Path(__file__).parents[1] / "jobs" / "sai-fla-parity-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --exclude=" in script
    for node in ("evc26", "evc31", "evc36", "evc38", "evc43", "evc50"):
        assert node in script
    assert "torch.cuda.device_count() == 1" in script
    assert 'flash-linear-attention") == "0.4.2"' in script
    assert "sai.training.fla_parity" in script
    assert "sai.training.runner" not in script
    assert "optimizer" not in script
    assert "4b" not in script.casefold()
