from pathlib import Path


def test_bridge_model_prefetch_is_cpu_only_and_exactly_pinned() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/prefetch_bridge_transfer_model_stokes.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --gres" not in script
    assert 'SAI_RUNTIME_ROOT:?' in script
    assert 'SAI_RUNTIME_COMMIT:?' in script
    assert 'export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"' in script
    assert "MODEL_REPOSITORY, MODEL_REVISION" in script
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        assert f'"{name}"' in script
    assert "allow_patterns=allowed" in script
    assert "model snapshot file geometry differs" in script
    assert '"gpu_requested": False' in script
    assert '"four_b_training_authorized": False' in script
