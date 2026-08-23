from __future__ import annotations

import json
from pathlib import Path

from sai.data.frequency_length_subdocument_deduplication import (
    build_frequency_length_deduplication,
    retention_budget,
    segment_subdocuments,
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


def test_retention_budget_matches_frequency_and_length_boundaries() -> None:
    assert retention_budget(1, 10) == 1
    assert retention_budget(2, 0) == 2
    assert retention_budget(20, 512) == 1
    assert retention_budget(20, 1) > retention_budget(20, 256)
    assert retention_budget(20, 256) >= retention_budget(20, 511)


def test_segmentation_is_lossless_and_preserves_fenced_code() -> None:
    text = "Title\nA short sentence. Another sentence.\n```python\nx = 1\n```\nTail"
    chunks = segment_subdocuments(text, minimum_characters=16)
    assert "".join(chunk["text"] for chunk in chunks) == text
    code = [chunk for chunk in chunks if chunk["code"]]
    assert len(code) == 1
    assert code[0]["text"] == "```python\nx = 1\n```\n"
    assert segment_subdocuments(text, code_document=True) == [
        {
            "text": text,
            "character_start": 0,
            "character_end": len(text),
            "code": True,
        }
    ]


def test_adaptive_subdocument_dedup_is_global_coherent_and_text_free(
    tmp_path: Path,
) -> None:
    repeated = "Copyright 2026 Example Corporation. All rights reserved. " * 3
    unique = [
        "A unique discussion of music and geometry.",
        "A distinct explanation of biology and information theory.",
        "An independent analysis of history and economics.",
        "A fourth account of poetry and psychology.",
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [
        _document(f"{value}\n{repeated}", f"row-{index}")
        for index, value in enumerate(unique)
    ]
    _write(first, rows[:2])
    _write(second, rows[2:])
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    receipt = tmp_path / "receipt.json"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = build_frequency_length_deduplication(
        [first, second],
        output,
        manifest,
        receipt,
        minimum_characters=16,
        delete_characters=100,
        reference_characters=64,
        effective_shards_numerator=2,
        effective_shards_denominator=1,
        chunk_records=1,
        maximum_line_bytes=4096,
        maximum_open_chunks=2,
        temporary_root=scratch,
    )
    outputs = [json.loads(line) for line in output.read_text().splitlines()]
    transforms = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(outputs) == 4
    assert result["counts"]["duplicate_groups"] >= 1
    assert result["counts"]["deleted_chunks"] == 18
    assert result["counts"]["coherence_restored_chunks"] == 0
    assert sum(repeated in row["text"] for row in outputs) == 1
    assert len(transforms) == 3
    assert repeated not in manifest.read_text()
    assert all(record["contains_source_text"] is False for record in transforms)
    assert result["index"]["merge_passes"] > 0
    assert not list(scratch.iterdir())
    assert result["training_ready"] is False


def test_short_isolated_candidate_is_restored_for_coherence(tmp_path: Path) -> None:
    repeated = "Repeated but intentionally too short to delete."
    rows = [
        _document(
            f"Unique preface {index}. {repeated} Unique ending {index}.", f"row-{index}"
        )
        for index in range(3)
    ]
    source = tmp_path / "source.jsonl"
    _write(source, rows)
    result = build_frequency_length_deduplication(
        [source],
        tmp_path / "output.jsonl",
        tmp_path / "manifest.jsonl",
        tmp_path / "receipt.json",
        minimum_characters=8,
        delete_characters=1_000,
        reference_characters=8,
        effective_shards_numerator=2,
        effective_shards_denominator=1,
        chunk_records=2,
        maximum_line_bytes=4096,
        maximum_open_chunks=2,
    )
    assert result["counts"]["deleted_chunks"] == 0
    assert result["counts"]["coherence_restored_chunks"] >= 1
    assert (tmp_path / "output.jsonl").read_text() != ""


def test_unique_population_replays_without_transform_records(tmp_path: Path) -> None:
    rows = [
        _document("A singular account of ceramic glazing.", "row-a"),
        _document("A separate proof about finite groups.", "row-b", "math"),
    ]
    source = tmp_path / "source.jsonl"
    _write(source, rows)
    output = tmp_path / "output.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    result = build_frequency_length_deduplication(
        [source],
        output,
        manifest,
        tmp_path / "receipt.json",
        minimum_characters=8,
        chunk_records=1,
        maximum_line_bytes=4096,
        maximum_open_chunks=2,
    )
    outputs = [json.loads(line) for line in output.read_text().splitlines()]
    assert outputs == rows
    assert manifest.read_text() == ""
    assert result["counts"]["initial_candidate_chunks"] == 0
    assert result["counts"]["output_documents"] == 2


def test_boundary_document_retains_all_its_same_group_occurrences(
    tmp_path: Path,
) -> None:
    repeated = "A repeated sufficiently long navigational template sentence. " * 2
    first = _document(repeated + repeated, "row-first")
    second = _document(repeated, "row-second")
    source = tmp_path / "source.jsonl"
    _write(source, [first, second])
    output = tmp_path / "output.jsonl"
    result = build_frequency_length_deduplication(
        [source],
        output,
        tmp_path / "manifest.jsonl",
        tmp_path / "receipt.json",
        minimum_characters=16,
        delete_characters=16,
        reference_characters=16,
        effective_shards_numerator=2,
        effective_shards_denominator=1,
        chunk_records=1,
        maximum_line_bytes=4096,
        maximum_open_chunks=2,
    )
    outputs = [json.loads(line) for line in output.read_text().splitlines()]
    survivor_id = min(first["identity_sha256"], second["identity_sha256"])
    survivor = next(
        row
        for row in outputs
        if row["source"]["row_id"]
        == ("row-first" if survivor_id == first["identity_sha256"] else "row-second")
    )
    expected = first if survivor_id == first["identity_sha256"] else second
    assert survivor["text"] == expected["text"]
    assert result["counts"]["initial_candidate_chunks"] >= 1


def test_keep_one_control_is_frozen_separately_from_adaptive_policy(
    tmp_path: Path,
) -> None:
    repeated = "A short recurring phrase with enough words."
    rows = [
        _document(f"Unique opening {index}. {repeated}", f"row-{index}")
        for index in range(4)
    ]
    source = tmp_path / "source.jsonl"
    _write(source, rows)
    results = {}
    for policy in ("adaptive_frequency_length", "keep_one_control"):
        root = tmp_path / policy
        root.mkdir()
        results[policy] = build_frequency_length_deduplication(
            [source],
            root / "output.jsonl",
            root / "manifest.jsonl",
            root / "receipt.json",
            minimum_characters=8,
            delete_characters=8,
            retention_policy=policy,
        )
    adaptive = results["adaptive_frequency_length"]
    keep_one = results["keep_one_control"]
    assert adaptive["frequency_length_retention_complete"] is True
    assert adaptive["keep_one_control_complete"] is False
    assert keep_one["frequency_length_retention_complete"] is False
    assert keep_one["keep_one_control_complete"] is True
    assert keep_one["counts"]["deleted_chunks"] > adaptive["counts"].get(
        "deleted_chunks", 0
    )
