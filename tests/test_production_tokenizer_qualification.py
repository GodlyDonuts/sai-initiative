from __future__ import annotations

from pathlib import Path


def test_production_tokenizer_jobs_are_cpu_only_and_fixed_48k() -> None:
    root = Path(__file__).resolve().parents[1]
    build = (
        root / "scripts/build_one_b_production_tokenizer_stokes.sbatch"
    ).read_text()
    qualify = (
        root / "scripts/qualify_one_b_production_tokenizer_stokes.sbatch"
    ).read_text()
    assert "--gres=gpu" not in build + qualify
    assert "--size 48k=48000" in build
    assert "production_qualification" in qualify
    assert "SAI_TOKENIZER_PROTECTED_SUITE.jsonl" in qualify
    assert "${#sai_books[@]} -eq 64" in build
