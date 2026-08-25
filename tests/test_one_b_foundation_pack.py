from __future__ import annotations

from sai.data.one_b_foundation_pack import (
    PART_SEQUENCES,
    PART_TOKENS,
    SEQUENCE_LENGTH,
    _selected,
)


def test_uint16_part_geometry_is_sequence_aligned() -> None:
    assert SEQUENCE_LENGTH == 4_096
    assert PART_SEQUENCES == 10_000
    assert PART_TOKENS == 40_960_000
    assert PART_TOKENS % SEQUENCE_LENGTH == 0


def test_selection_matches_frozen_first_16_hex_contract() -> None:
    row = {
        "curriculum_band": "foundation",
        "split": "train",
        "curriculum_priority_sha256": "00000000000f4241" + "0" * 48,
    }
    plan = {"bands": {"foundation": {"selection_ppm": 1}}}
    assert not _selected(row, plan)
    row["curriculum_priority_sha256"] = "0" * 64
    assert _selected(row, plan)
    row["split"] = "development"
    assert not _selected(row, plan)
