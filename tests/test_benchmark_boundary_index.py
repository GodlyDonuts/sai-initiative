from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sai.data.benchmark_boundary_index import (
    BenchmarkBoundaryError,
    DigestChunks,
    SourceSpec,
    _git_blob_sha1,
    _json_array,
    projected_strings,
)


def test_json_array_streams_across_small_internal_buffers(tmp_path: Path) -> None:
    path = tmp_path / "board.json"
    rows = [{"id": index, "text": "long context " * 10_000} for index in range(3)]
    path.write_text(json.dumps(rows))
    assert list(_json_array(path)) == rows


def test_livecodebench_projection_never_persists_private_grading_payload() -> None:
    spec = SourceSpec(
        "livecodebench",
        "huggingface",
        "livecodebench/code_generation_lite",
        "0" * 40,
        "test.jsonl",
        1,
        "1" * 64,
        projection="livecodebench_model_visible",
    )
    groups = projected_strings(
        spec,
        {
            "question_title": "A useful title",
            "question_content": "Explain the exact public problem statement.",
            "starter_code": "def solve():\n    pass",
            "public_test_cases": "public examples belong in the boundary",
            "private_test_cases": "PRIVATE_SENTINEL must never be redistributed",
        },
    )
    flattened = "\n".join(groups[0])
    assert "public examples" in flattened
    assert "PRIVATE_SENTINEL" not in flattened


def test_livebench_projection_honors_release_and_removal_dates() -> None:
    spec = SourceSpec(
        "livebench",
        "huggingface",
        "livebench/reasoning",
        "0" * 40,
        "test.parquet",
        1,
        "1" * 64,
        projection="livebench_release",
    )
    admitted = {
        "question_id": "one useful identity",
        "turns": ["A sufficiently long released prompt"],
        "livebench_release_date": "2024-11-25",
        "livebench_removal_date": "",
    }
    assert projected_strings(spec, admitted)
    assert not projected_strings(
        spec, {**admitted, "livebench_release_date": "2024-11-26"}
    )
    assert not projected_strings(
        spec, {**admitted, "livebench_removal_date": "2024-11-25"}
    )


def test_digest_chunks_external_merge_is_sorted_unique(tmp_path: Path) -> None:
    chunks = DigestChunks(tmp_path, "word", maximum_buffered=2)
    values = [hashlib.sha256(value.encode()).digest() for value in ("c", "a", "c", "b")]
    chunks.add(values)
    output = tmp_path / "index.bin"
    assert chunks.finalize(output) == 3
    observed = [output.read_bytes()[index : index + 32] for index in range(0, 96, 32)]
    assert observed == sorted(set(values))
    assert chunks.observations == 4


def test_git_blob_identity_binds_bytes_and_length(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b'{"prompt":"hello world"}\n')
    expected = hashlib.sha1(  # noqa: S324 - test replays Git's object format
        f"blob {path.stat().st_size}\0".encode() + path.read_bytes()
    ).hexdigest()
    assert _git_blob_sha1(path) == expected
    path.write_bytes(path.read_bytes() + b" ")
    assert _git_blob_sha1(path) != expected


def test_musr_projection_rejects_missing_question_geometry() -> None:
    spec = SourceSpec(
        "musr",
        "github",
        "Zayne-sprague/MuSR",
        "0" * 40,
        "datasets/example.json",
        1,
        git_blob_sha1="1" * 40,
        projection="musr",
    )
    try:
        projected_strings(spec, {"context": "A sufficiently long context"})
    except BenchmarkBoundaryError as error:
        assert "MuSR" in str(error)
    else:
        raise AssertionError("missing MuSR questions were accepted")
