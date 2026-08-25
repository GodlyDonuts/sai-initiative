import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from sai.data import common_pile_stack_edu_practical_scan as subject
from sai.data.token_stream import canonical_sha256, sha256_file


def _python_text() -> str:
    return "\n".join(
        f"def educational_example_{index}(value: int) -> int:\n"
        f"    return value + {index}\n"
        for index in range(24)
    )


def _row(**changes):
    blob_id = "a" * 40
    row = {
        "id": blob_id,
        "int_score": 3,
        "text": _python_text(),
        "metadata": {
            "blob_id": blob_id,
            "detected_licenses": ["MIT"],
            "is_generated": False,
            "is_vendor": False,
            "language": "Python",
            "license_type": "permissive",
            "path": "/src/educational.py",
            "repo_name": "example/educational",
            "src_encoding": "UTF-8",
        },
    }
    row.update(changes)
    return row


def test_route_accepts_useful_permissive_code_and_rejects_unsafe_rows():
    route, selected = subject._route(_row())
    assert route == "pass_practical_code_gate"
    assert selected is not None
    assert (
        selected["content_sha256"]
        == hashlib.sha256(_python_text().encode()).hexdigest()
    )
    assert selected["licenses"] == ["MIT"]

    route, _ = subject._route(_row(int_score=2))
    assert route == "hold_educational_score"

    bad_rights = _row()
    bad_rights["metadata"] = {
        **bad_rights["metadata"],
        "detected_licenses": ["GPL-3.0"],
    }
    route, _ = subject._route(bad_rights)
    assert route == "hold_provenance_or_rights"

    route, _ = subject._route(
        _row(text=_python_text() + '\nAPI_KEY = "sk_live_12345678901234567890"\n')
    )
    assert route == "hold_high_confidence_safety"

    broken = "def broken(:\n" + "\n".join(
        f"# distinct context {index}" for index in range(100)
    )
    route, _ = subject._route(_row(text=broken))
    assert route == "hold_python_syntax"


def test_route_quarantines_python_that_exhausts_parser_memory(monkeypatch):
    def exhaust_parser_memory(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr("sai.data.stack_edu_safety.ast.parse", exhaust_parser_memory)
    route, selected = subject._route(_row())
    assert route == "hold_python_syntax"
    assert selected is None


def test_load_parents_rejects_incomplete_geometry(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_id": subject.SOURCE_ID,
                "source_repository": subject.SOURCE_REPOSITORY,
                "source_revision": subject.SOURCE_REVISION,
                "source_path": "stack-edu-0000.json.gz",
                "bytes": 10,
                "sha256": "1" * 64,
            }
        )
        + "\n"
    )
    assert subject.load_parents(path, expected_parents=1)[0]["source_path"] == (
        "stack-edu-0000.json.gz"
    )
    try:
        subject.load_parents(path, expected_parents=2)
    except subject.StackEduPracticalScanError as error:
        assert "coverage" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("incomplete source geometry was accepted")


def test_run_shard_emits_text_free_verified_locators(tmp_path: Path, monkeypatch):
    source = tmp_path / "stack-edu-0000.json.gz"
    rows = [
        _row(),
        _row(
            id="b" * 40,
            metadata={**_row()["metadata"], "blob_id": "b" * 40},
            int_score=4,
        ),
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_id": subject.SOURCE_ID,
                "source_repository": subject.SOURCE_REPOSITORY,
                "source_revision": subject.SOURCE_REVISION,
                "source_path": source.name,
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            },
            sort_keys=True,
        )
        + "\n"
    )
    monkeypatch.setattr(subject, "_download", lambda parent, token, root: source)
    output = tmp_path / "output"
    receipt = subject.run_shard(
        manifest,
        output,
        0,
        "test-token",
        expected_parents=1,
        scratch_root=tmp_path,
    )
    assert receipt["selected"]["rows"] == 2
    assert receipt["all_source_rows_accounted"] is True
    assert receipt["source_text_copied"] is False
    assert receipt["training_ready"] is False
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    table = pq.read_table(output / "locators.parquet")
    assert table.num_rows == 2
    assert "text" not in table.column_names
    assert table["integer_score"].to_pylist() == [3, 4]
    assert (output / "receipt.json").is_file()


def test_stokes_job_preserves_one_parent_per_array_identity():
    script = Path(
        "scripts/run_common_pile_stack_edu_practical_scan_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-94%95" in script
    assert "#SBATCH --exclude=ec65" in script
    assert "#SBATCH --no-requeue" in script
    assert "--expected-parents 95" in script
    assert "shard_$(printf '%05d' \"${SLURM_ARRAY_TASK_ID}\")" in script
