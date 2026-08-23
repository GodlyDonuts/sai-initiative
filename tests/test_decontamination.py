from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.decontamination import (
    POLICY,
    RAW_SCHEMA,
    DecontaminationError,
    _code_overlap_count,
    _code_shingles,
    _overlap_count,
    _shingles,
    build,
    validate,
)
from sai.data.token_stream import canonical_sha256


def write_digest_boundary(root: Path, word: set[bytes], code: set[bytes]) -> Path:
    root.mkdir()
    descriptors = {}
    for key, values in (("word_index", word), ("code_index", code)):
        filename = "word.bin" if key == "word_index" else "code.bin"
        path = root / filename
        path.write_bytes(b"".join(sorted(values)))
        descriptors[key] = {
            "file": filename,
            "digest_bytes": 32,
            "observed_shingles": len(values),
            "unique_shingles": len(values),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "schema": "sai-official-benchmark-boundary-index-v2",
        "status": "complete",
        "policy": POLICY,
        "policy_sha256": canonical_sha256(POLICY),
        **descriptors,
        "benchmark_contamination_gate_ready": True,
        "raw_benchmark_text_persisted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt) + "\n")
    return root


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


def test_build_and_replay_accepts_non_reversible_binary_boundary(
    tmp_path: Path,
) -> None:
    benchmark_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    code_text = "def solve value return value plus one if value else zero"
    boundary = write_digest_boundary(
        tmp_path / "binary-boundary",
        _shingles(benchmark_text.split(), 13),
        _shingles(code_text.split(), 8),
    )
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [
            raw(0, "A separate clean document about stellar spectroscopy."),
            raw(1, f"prefix {benchmark_text} suffix"),
            raw(2, f"example {code_text} trailing"),
        ],
    )
    output = tmp_path / "admitted.jsonl"
    receipt = tmp_path / "receipt.json"
    report = build(
        source,
        [],
        output,
        receipt,
        boundary_indexes=[boundary],
    )
    assert report["accepted"] == 1
    assert report["dropped"] == 2
    assert validate(receipt) == report


def test_materialized_binary_boundary_is_byte_exact_and_replays(tmp_path: Path) -> None:
    benchmark_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    boundary = write_digest_boundary(
        tmp_path / "binary-boundary",
        _shingles(benchmark_text.split(), 13),
        {hashlib.sha256(b"unused-code-window").digest()},
    )
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [
            raw(0, "A clean document about the geometry of elliptic curves."),
            raw(1, f"prefix {benchmark_text} suffix"),
        ],
    )
    output = tmp_path / "admitted.jsonl"
    receipt = tmp_path / "receipt.json"
    report = build(
        source,
        [],
        output,
        receipt,
        boundary_indexes=[boundary],
        materialize_boundary_indexes=True,
    )
    assert report["accepted"] == 1
    assert report["dropped"] == 1
    assert validate(receipt) == report


def test_parallel_build_is_byte_exact_and_replays_sequentially(tmp_path: Path) -> None:
    boundary = write_jsonl(
        tmp_path / "boundary.jsonl",
        [
            {
                "prompt": (
                    "one two three four five six seven eight nine ten eleven twelve "
                    "thirteen"
                )
            }
        ],
    )
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [
            raw(index, f"Unique clean scientific document number {index} about tensors")
            for index in range(150)
        ]
        + [raw(150, "Unique clean scientific document number 17 about tensors")],
    )
    sequential_output = tmp_path / "sequential.jsonl"
    sequential_receipt = tmp_path / "sequential.receipt.json"
    parallel_output = tmp_path / "parallel.jsonl"
    parallel_receipt = tmp_path / "parallel.receipt.json"
    sequential = build(
        source,
        [boundary],
        sequential_output,
        sequential_receipt,
        workers=1,
    )
    parallel = build(
        source,
        [boundary],
        parallel_output,
        parallel_receipt,
        workers=3,
    )
    assert parallel_output.read_bytes() == sequential_output.read_bytes()
    assert parallel["output"]["sha256"] == sequential["output"]["sha256"]
    for key in (
        "source",
        "boundaries",
        "boundary_manifest_sha256",
        "policy",
        "policy_sha256",
        "scanned",
        "accepted",
        "dropped",
        "accepted_identity_sha256",
        "dropped_evidence_sha256",
    ):
        assert parallel[key] == sequential[key]
    assert validate(parallel_receipt) == parallel


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
    assert '--workers "$SLURM_CPUS_PER_TASK"' in job


def test_shingle_index_uses_compact_byte_exact_sha256_keys() -> None:
    tokens = [str(index) for index in range(15)]
    observed = _shingles(tokens, 13)
    assert observed == {
        bytes.fromhex(canonical_sha256(tokens[index : index + 13]))
        for index in range(3)
    }
    assert all(isinstance(value, bytes) and len(value) == 32 for value in observed)


def test_streaming_overlap_count_is_exact_unique_intersection() -> None:
    tokens = [str(index % 17) for index in range(250)]
    source = _shingles(tokens, 8)
    boundary = set(list(source)[::3]) | {b"x" * 32}
    assert _overlap_count(tokens, 8, boundary) == len(source.intersection(boundary))
    assert _overlap_count(tokens[:4], 8, boundary) == 0


def test_code_shingles_exclude_punctuation_only_and_generic_short_windows() -> None:
    punctuation = ["(", ")", "{", "}", "[", "]", ",", ";"]
    generic_short = ["a", ",", "b", ",", "c", ",", "d", ","]
    distinctive = ["if", "(", "value", "==", "0", ")", "return", "result"]
    assert _code_shingles(punctuation) == set()
    assert _code_shingles(generic_short) == set()
    assert len(_code_shingles(distinctive)) == 1


def test_code_overlap_applies_the_same_eligibility_rule_as_indexing() -> None:
    punctuation = ["(", ")", "{", "}", "[", "]", ",", ";"]
    distinctive = ["if", "(", "value", "==", "0", ")", "return", "result"]
    boundary = _shingles(punctuation, 8) | _code_shingles(distinctive)
    assert _code_overlap_count(punctuation, boundary) == 0
    assert _code_overlap_count(distinctive, boundary) == 1
