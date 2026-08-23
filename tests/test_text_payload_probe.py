from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sai.data.text_payload_probe import (
    TextPayloadProbeError,
    _text_measurement,
    measure_local_member,
)


def test_text_measurement_separates_useful_payload_bytes() -> None:
    result = _text_measurement(["", "a" * 199, "b" * 200, "c" * (128 * 1024 + 1), None])
    assert result["rows"] == 5
    assert result["string_text_rows"] == 4
    assert result["empty_text_rows"] == 1
    assert result["short_text_rows"] == 2
    assert result["useful_text_rows"] == 1
    assert result["oversized_text_rows"] == 1
    assert result["useful_text_utf8_bytes"] == 200


def test_measures_exact_parquet_text_column(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    table = pytest.importorskip("pyarrow").table(
        {"text": ["x" * 200, "y" * 300], "embedding": [[1.0], [2.0]]}
    )
    path = tmp_path / "member.parquet"
    parquet.write_table(table, path, row_group_size=1)
    result = measure_local_member(path, text_column="text")
    assert result["rows"] == 2
    assert result["text_utf8_bytes"] == 500
    assert result["useful_text_utf8_bytes"] == 500


def test_measures_exact_zstandard_jsonl(tmp_path: Path) -> None:
    zstandard = pytest.importorskip("zstandard")
    path = tmp_path / "member.jsonl.zst"
    encoded = "".join(
        json.dumps({"text": text}) + "\n" for text in ("x" * 200, "y" * 199)
    ).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(encoded))
    result = measure_local_member(path, text_column="text")
    assert result["rows"] == 2
    assert result["useful_text_rows"] == 1
    assert result["useful_text_utf8_bytes"] == 200


def test_rejects_unknown_member_format(tmp_path: Path) -> None:
    path = tmp_path / "member.txt"
    path.write_text("x")
    with pytest.raises(TextPayloadProbeError, match="format is unsupported"):
        measure_local_member(path, text_column="text")


def test_measures_exact_gzip_json(tmp_path: Path) -> None:
    path = tmp_path / "member.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write(json.dumps({"text": "x" * 201}) + "\n")
    result = measure_local_member(path, text_column="text")
    assert result["rows"] == 1
    assert result["useful_text_utf8_bytes"] == 201
