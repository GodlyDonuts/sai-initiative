from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.training.model_mechanics import ModelMechanicsError, load_100m_configs

ROOT = Path(__file__).resolve().parents[1]


def test_loads_exact_two_non_gqa_100m_configs() -> None:
    configs = load_100m_configs(ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json")
    assert tuple(configs) == ("gdn_hybrid", "kda_mla_hybrid")
    assert configs["gdn_hybrid"].hidden_size == 512
    assert configs["kda_mla_hybrid"].head_dim == 64


def test_geometry_duplicate_or_missing_fails_closed(tmp_path: Path) -> None:
    source = json.loads((ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json").read_text())
    source["geometries"] = [
        row
        for row in source["geometries"]
        if row.get("mixer_family") != "kda_mla_hybrid"
    ]
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ModelMechanicsError, match="set differs"):
        load_100m_configs(path)


def test_model_mechanics_job_is_single_h100_and_never_4b() -> None:
    job = (ROOT / "jobs" / "sai-model-mechanics-single-h100.sbatch").read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "--no-requeue" in job
    assert "EXPECTED_COMMIT" in job
    assert "PARITY_RECEIPT" in job
    assert "4b" not in job.lower()
