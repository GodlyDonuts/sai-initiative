from __future__ import annotations

from pathlib import Path


def test_preflight_prohibits_optimizer_and_checkpoint_writes() -> None:
    source = (
        Path(__file__).parents[1] / "src/sai/training/one_b_gpu_preflight.py"
    ).read_text()
    assert 'optimizer_constructed": False' in source
    assert 'optimizer_update_performed": False' in source
    assert 'checkpoint_written": False' in source
    assert ".step(" not in source
