from __future__ import annotations

from pathlib import Path

from sai.data.one_b_foundation_pack import SEQUENCE_LENGTH
from sai.data.one_b_stage_schedule import _cycle
from sai.data.token_stream import sha256_file


def test_cycle_uses_references_and_one_exact_prefix(tmp_path: Path) -> None:
    parts = []
    for index, sequences in enumerate((3, 2)):
        path = tmp_path / f"part-{index}.bin"
        path.write_bytes(bytes([index + 1]) * sequences * SEQUENCE_LENGTH * 2)
        parts.append(
            {
                "path": str(path),
                "sequences": sequences,
                "tokens": sequences * SEQUENCE_LENGTH,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    entries = _cycle(parts, 11, tmp_path, "band")
    assert sum(row["sequences_per_repeat"] * row["repeat"] for row in entries) == 11
    assert any(row["source"] == "exact_stage_prefix" for row in entries)


def test_cycle_records_atomic_final_prefix_location(tmp_path: Path) -> None:
    source = tmp_path / "part.bin"
    source.write_bytes(b"a" * 3 * SEQUENCE_LENGTH * 2)
    parts = [
        {
            "path": str(source),
            "sequences": 3,
            "tokens": 3 * SEQUENCE_LENGTH,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    ]
    stage = tmp_path / ".schedule.partial"
    stage.mkdir()
    final = tmp_path / "schedule"

    entries = _cycle(parts, 4, stage, "band", receipt_root=final)

    tail = next(row for row in entries if row["source"] == "exact_stage_prefix")
    assert tail["path"] == str((final / "band-exact-tail.bin").resolve())
    assert (stage / "band-exact-tail.bin").is_file()
