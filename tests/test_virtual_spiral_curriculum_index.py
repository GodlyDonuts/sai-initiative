import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from sai.data.token_stream import canonical_sha256
from sai.data.virtual_spiral_curriculum_index import (
    BANDS,
    POLICY,
    POLICY_SHA256,
    ROW_FIELD_NAMES,
    ROW_SCHEMA,
    VirtualSpiralCurriculumIndexError,
    _index_row,
    _schema,
    _validated_index_row,
    _write_shard,
    spiral_band,
)


@pytest.mark.parametrize(
    ("difficulty", "burden", "score", "band"),
    [
        (0, 0, 0, "foundation"),
        (999, 200, 999, "foundation"),
        (1_000, 500, 1_000, "intermediate"),
        (1_500, 2_500, 2_500, "advanced"),
        (2_999, 3_000, 3_000, "expert"),
        (4_000, 4_000, 4_000, "expert"),
    ],
)
def test_spiral_band_uses_difficulty_or_prerequisite_burden(
    difficulty: int, burden: int, score: int, band: str
) -> None:
    assert spiral_band(difficulty, burden) == (score, band)


def test_moving_center_policy_keeps_fundamentals_and_experts_in_every_stage() -> None:
    assert POLICY_SHA256 == canonical_sha256(POLICY)
    stages = POLICY["stage_policy"]
    assert sum(stage["token_fraction_ppm"] for stage in stages.values()) == 1_000_000
    for stage in stages.values():
        assert sum(stage["band_fraction_ppm"]) == 1_000_000
        assert stage["band_fraction_ppm"][BANDS.index("foundation")] > 0
        assert stage["band_fraction_ppm"][BANDS.index("expert")] > 0
    assert (
        stages["foundation"]["band_fraction_ppm"][0]
        > stages["annealing"]["band_fraction_ppm"][0]
    )
    assert (
        stages["foundation"]["band_fraction_ppm"][3]
        < stages["annealing"]["band_fraction_ppm"][3]
    )


def _row(index: int = 0) -> dict:
    return _index_row(
        component="pleias_common_corpus",
        component_shard=0,
        component_row_index=index,
        document_identity_sha256=f"{index + 1:064x}",
        content_sha256=f"{index + 100:064x}",
        output_text_utf8_bytes=1_000 + index,
        corpus_split="train",
        source_group_sha256=f"{index + 200:064x}",
        rights_label="Public Domain",
        quality_floor_milli=8_000,
        difficulty_milli=2_500,
        prerequisite_burden_milli=1_500,
        semantic_phase_hint="depth",
        semantic_domains=["science", "science"],
        concepts=["energy", "energy"],
        prerequisites=["algebra"],
        source_custody_sha256=f"{index + 300:064x}",
    )


def test_index_row_is_source_text_free_and_deterministic() -> None:
    row = _row()
    assert row["schema"] == ROW_SCHEMA
    assert row["spiral_band"] == "advanced"
    assert row["semantic_domains"] == ["science"]
    assert row["concepts"] == ["energy"]
    assert "text" not in row
    assert row["training_ready"] is False
    assert row == _row()
    assert set(_schema().names) == ROW_FIELD_NAMES


def test_index_shard_persists_only_metadata_and_hashes(tmp_path: Path) -> None:
    output = tmp_path / "shard_00000"
    result = _write_shard(
        output,
        [_row()],
        component="pleias_common_corpus",
        logical_shards=1,
        shard_index=0,
        source_receipt_sha256="f" * 64,
    )
    assert result["source_text_persisted"] is False
    assert result["exact_token_allocation_complete"] is False
    rows = pq.read_table(output / "curriculum-index.parquet").to_pylist()
    assert rows == [_row()]
    assert "Public Domain" not in (output / "receipt.json").read_text() or (
        "text" not in json.loads((output / "receipt.json").read_text())
    )


def test_spiral_band_rejects_out_of_range_semantics() -> None:
    with pytest.raises(VirtualSpiralCurriculumIndexError, match="difficulty differs"):
        spiral_band(4_001, 0)


def test_index_row_rejects_malformed_custody_hash() -> None:
    with pytest.raises(VirtualSpiralCurriculumIndexError, match="source differs"):
        _index_row(
            component="pleias_common_corpus",
            component_shard=0,
            component_row_index=0,
            document_identity_sha256="1" * 64,
            content_sha256="2" * 64,
            output_text_utf8_bytes=100,
            corpus_split="train",
            source_group_sha256="3" * 64,
            rights_label="Public Domain",
            quality_floor_milli=8_000,
            difficulty_milli=1_000,
            prerequisite_burden_milli=0,
            semantic_phase_hint="foundation",
            semantic_domains=["science"],
            concepts=[],
            prerequisites=[],
            source_custody_sha256="not-a-hash",
        )


def test_index_row_replay_rejects_derived_field_tampering() -> None:
    row = _row()
    row["spiral_band"] = "expert"
    with pytest.raises(VirtualSpiralCurriculumIndexError, match="replay differs"):
        _validated_index_row(row)


def test_curriculum_index_jobs_are_cpu_only_immutable_and_non_requeueing() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = {
        "index_pleias_virtual_spiral_curriculum_stokes.sbatch": (
            "#SBATCH --array=0-127%16"
        ),
        "index_institutional_books_virtual_spiral_curriculum_stokes.sbatch": (
            "#SBATCH --array=0-63%16"
        ),
        "aggregate_virtual_spiral_curriculum_index_stokes.sbatch": ("--scratch-root"),
    }
    for name, contract in jobs.items():
        job = (root / "scripts" / name).read_text()
        assert contract in job
        assert "#SBATCH --no-requeue" in job
        assert "#SBATCH --gres" not in job
        assert "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}" in job
        assert "rm " not in job
