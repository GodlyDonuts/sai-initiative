from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.external_exact_deduplication import (
    ExternalExactDeduplicationError,
    build_exact_deduplication,
)
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256


def _document(text: str, row_id: str, domain: str = "english") -> dict:
    payload = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "example/corpus",
            "row_id": row_id,
            "license": "CC-BY-4.0",
            "domain": domain,
        },
        "verification": {"benchmark_disjoint": True, "evidence_sha256": "e" * 64},
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _run(root: Path, sources: list[Path], name: str) -> tuple[dict, Path, Path, Path]:
    output = root / f"{name}-survivors.jsonl"
    duplicates = root / f"{name}-duplicates.jsonl"
    receipt = root / f"{name}-receipt.json"
    scratch = root / f"{name}-scratch"
    scratch.mkdir()
    result = build_exact_deduplication(
        sources,
        output,
        duplicates,
        receipt,
        chunk_records=1,
        maximum_line_bytes=4096,
        maximum_open_chunks=2,
        temporary_root=scratch,
    )
    assert not list(scratch.iterdir())
    return result, output, duplicates, receipt


def test_external_exact_dedup_is_cross_source_bounded_and_deterministic(
    tmp_path: Path,
) -> None:
    first_duplicate = _document("  Café   systems ARE elegant.  ", "row-a")
    second_duplicate = _document("café systems are ELEGANT.", "row-b")
    unique_first = _document("A unique discussion of poetry.", "row-c")
    unique_second = _document("A distinct proof about prime numbers.", "row-d", "math")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, [first_duplicate, unique_first])
    _write(second, [unique_second, second_duplicate])

    result, output, duplicates, receipt = _run(tmp_path, [first, second], "forward")
    reverse, reverse_output, _, _ = _run(tmp_path, [second, first], "reverse")
    survivors = [json.loads(line) for line in output.read_text().splitlines()]
    drops = [json.loads(line) for line in duplicates.read_text().splitlines()]
    assert result["counts"] == {
        "blank_lines": 0,
        "documents": 4,
        "survivors": 3,
        "duplicates_dropped": 1,
        "duplicate_groups": 1,
    }
    assert result["index"]["initial_chunk_count"] == 4
    assert result["index"]["merge_passes"] == 1
    assert result["index"]["final_merge_chunk_count"] == 2
    assert result["index"]["temporary_index_removed"] is True
    assert result["global_normalized_exact_deduplication_complete"] is True
    assert result["global_near_duplicate_filtering_complete"] is False
    assert result["training_ready"] is False
    assert len(survivors) == 3
    assert len(drops) == 1
    assert "text" not in drops[0]
    assert "Café" not in duplicates.read_text()
    expected_survivor = min(
        first_duplicate["identity_sha256"], second_duplicate["identity_sha256"]
    )
    assert drops[0]["survivor_identity_sha256"] == expected_survivor
    assert output.read_bytes() == reverse_output.read_bytes()
    assert reverse["counts"] == result["counts"]
    assert json.loads(receipt.read_text())["receipt_sha256"] == result["receipt_sha256"]


def test_external_exact_dedup_rejects_aliased_input(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    alias = tmp_path / "alias.jsonl"
    _write(source, [_document("One valid document.", "row-a")])
    alias.hardlink_to(source)
    with pytest.raises(ExternalExactDeduplicationError, match="aliased"):
        build_exact_deduplication(
            [source, alias],
            tmp_path / "output.jsonl",
            tmp_path / "duplicates.jsonl",
            tmp_path / "receipt.json",
        )


def test_external_exact_dedup_fails_closed_on_oversized_line(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write(source, [_document("x" * 2_000, "row-a")])
    output = tmp_path / "output.jsonl"
    duplicates = tmp_path / "duplicates.jsonl"
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ExternalExactDeduplicationError, match="frozen cap"):
        build_exact_deduplication(
            [source],
            output,
            duplicates,
            receipt,
            maximum_line_bytes=256,
        )
    assert not output.exists()
    assert not duplicates.exists()
    assert not receipt.exists()
