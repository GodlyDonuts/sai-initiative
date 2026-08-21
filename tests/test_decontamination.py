from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.decontamination import (
    RAW_SCHEMA,
    DecontaminationError,
    _shingles,
    build,
    validate,
)
from sai.data.token_stream import canonical_sha256


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def raw(index: int, text: str) -> dict:
    return {
        "schema": RAW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "unit-corpus",
            "revision": "abc123",
            "source_file": "part-000.parquet",
            "row_index": index,
            "license": "ODC-By-1.0",
            "domain": "english",
        },
    }


def test_build_and_replay_reject_word_code_overlap_and_duplicates(
    tmp_path: Path,
) -> None:
    benchmark_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    code = "def solve value return value plus one if value else zero"
    boundary = write_jsonl(
        tmp_path / "boundary.jsonl", [{"prompt": benchmark_text, "answer": code}]
    )
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [
            raw(
                0, "A clean technical article about matrix factorization and compilers."
            ),
            raw(1, f"prefix {benchmark_text} suffix"),
            raw(2, f"example {code} trailing"),
            raw(
                3, "A clean technical article about matrix factorization and compilers."
            ),
        ],
    )
    output = tmp_path / "admitted.jsonl"
    receipt = tmp_path / "receipt.json"
    report = build(source, [boundary], output, receipt)
    assert report["scanned"] == 4
    assert report["accepted"] == 1
    assert report["dropped"] == 3
    assert validate(receipt) == report
    row = json.loads(output.read_text())
    assert row["verification"]["benchmark_disjoint"] is True
    assert len(row["verification"]["evidence_sha256"]) == 64


def test_boundary_source_or_output_tampering_fails_replay(tmp_path: Path) -> None:
    benchmark_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    boundary = write_jsonl(
        tmp_path / "boundary.jsonl",
        [{"prompt": benchmark_text}],
    )
    source = write_jsonl(
        tmp_path / "source.jsonl", [raw(0, "A separate clean document")]
    )
    output = tmp_path / "admitted.jsonl"
    receipt = tmp_path / "receipt.json"
    build(source, [boundary], output, receipt)
    output.write_text(output.read_text() + "\n")
    with pytest.raises(DecontaminationError, match="output"):
        validate(receipt)


def test_empty_or_malformed_boundary_fails_closed(tmp_path: Path) -> None:
    source = write_jsonl(tmp_path / "source.jsonl", [raw(0, "clean text")])
    boundary = tmp_path / "boundary.jsonl"
    boundary.write_text("\n")
    with pytest.raises(DecontaminationError, match="no usable text"):
        build(source, [boundary], tmp_path / "out", tmp_path / "receipt")


def test_cpu_decontamination_job_requires_exact_twenty_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs" / "sai-decontaminate-mechanics-cpu.sbatch").read_text()
    assert 'test "${#boundary_paths[@]}" = 20' in job
    assert "--no-requeue" in job
    assert "--gres=" not in job
    assert "EXPECTED_COMMIT" in job


def test_shingle_index_uses_compact_byte_exact_sha256_keys() -> None:
    tokens = [str(index) for index in range(15)]
    observed = _shingles(tokens, 13)
    assert observed == {
        bytes.fromhex(canonical_sha256(tokens[index : index + 13]))
        for index in range(3)
    }
    assert all(isinstance(value, bytes) and len(value) == 32 for value in observed)
