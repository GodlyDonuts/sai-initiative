from __future__ import annotations

from pathlib import Path

from sai.data.one_b_foundation_aggregate import _copy_prefix
from sai.data.one_b_foundation_pack import SEQUENCE_LENGTH


def test_copy_prefix_is_exact_uint16_sequence_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * (SEQUENCE_LENGTH * 2 // 256) * 3)
    output = tmp_path / "prefix.bin"
    descriptor = _copy_prefix(source, output, 2)
    assert descriptor["sequences"] == 2
    assert descriptor["tokens"] == 2 * SEQUENCE_LENGTH
    assert descriptor["bytes"] == 2 * SEQUENCE_LENGTH * 2
    assert output.read_bytes() == source.read_bytes()[: 2 * SEQUENCE_LENGTH * 2]


def test_copy_prefix_can_record_atomic_final_location(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"a" * SEQUENCE_LENGTH * 2)
    stage = tmp_path / ".exact.partial" / "prefix.bin"
    stage.parent.mkdir()
    final = tmp_path / "exact" / "prefix.bin"

    descriptor = _copy_prefix(source, stage, 1, receipt_path=final)

    assert descriptor["path"] == str(final.resolve())
    assert stage.is_file()
