import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.common_pile_stack_edu_practical_admission import (
    StackEduPracticalAdmissionError,
    _valid_locator,
    build_admission,
)
from sai.data.common_pile_stack_edu_practical_scan import (
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    SOURCE_ID,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    _schema,
)
from sai.data.quarantine_exclusion_registry import (
    RECORD_SCHEMA as QUARANTINE_RECORD_SCHEMA,
)
from sai.data.quarantine_exclusion_registry import (
    SCHEMA as QUARANTINE_REGISTRY_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _quarantine_registry(root: Path, content_hash: str) -> Path:
    root.mkdir()
    row = {
        "schema": QUARANTINE_RECORD_SCHEMA,
        "candidate_identity_sha256": "9" * 64,
        "source_content_sha256": content_hash,
        "source_manifest_receipt_sha256": "a" * 64,
        "source_record_sha256": "b" * 64,
        "route": "quarantine",
        "dataset_materialization_allowed": False,
        "source_text_persisted": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    registry = root / "quarantine_registry.jsonl"
    registry.write_text(json.dumps(row, sort_keys=True) + "\n")
    receipt = _signed(
        {
            "schema": QUARANTINE_REGISTRY_SCHEMA,
            "status": "complete_quarantine_exclusion_registry",
            "unique_quarantine_rows": 1,
            "registry": {
                "path": registry.name,
                "rows": 1,
                "bytes": registry.stat().st_size,
                "sha256": sha256_file(registry),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"]]
                ),
            },
            "dataset_materialization_allowed": False,
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def _parent(index: int) -> dict:
    return {
        "source_id": SOURCE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_path": f"stack-edu-{index:04d}.json.gz",
        "bytes": 100 + index,
        "sha256": f"{index + 1:064x}",
    }


def _locator(
    parent: dict,
    row_index: int,
    identity: str,
    content_hash: str,
    text_bytes: int,
) -> dict:
    return {
        "schema": LOCATOR_SCHEMA,
        "source_id": SOURCE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_path": parent["source_path"],
        "source_parent_sha256": parent["sha256"],
        "source_row_index": row_index,
        "source_row_identity_sha256": identity,
        "blob_id": f"{row_index + 1:040x}",
        "repo_name": "example/educational",
        "repo_path": f"/src/example_{row_index}.py",
        "language": "Python",
        "licenses": ["MIT"],
        "integer_score": 3 + row_index % 2,
        "text_utf8_bytes": text_bytes,
        "content_sha256": content_hash,
    }


def _scan_shard(
    root: Path,
    manifest: Path,
    parent: dict,
    shard_index: int,
    rows: list[dict],
) -> None:
    shard = root / "shards" / f"shard_{shard_index:05d}"
    shard.mkdir(parents=True)
    locator = shard / "locators.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), locator)
    selected_bytes = sum(row["text_utf8_bytes"] for row in rows)
    receipt = _signed(
        {
            "schema": SHARD_SCHEMA,
            "status": "complete_common_pile_stack_edu_practical_scan_shard",
            "shard_index": shard_index,
            "expected_parents": 2,
            "source": {
                "manifest_sha256": sha256_file(manifest),
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
                "path": parent["source_path"],
                "parent_bytes": parent["bytes"],
                "parent_sha256": parent["sha256"],
            },
            "selected": {"rows": len(rows), "text_utf8_bytes": selected_bytes},
            "output": {
                "path": locator.name,
                "bytes": locator.stat().st_size,
                "sha256": sha256_file(locator),
            },
            "full_parent_byte_identity_verified": True,
            "all_source_rows_accounted": True,
            "training_ready": False,
        }
    )
    (shard / "receipt.json").write_text(json.dumps(receipt))


def test_valid_locator_rejects_unbound_or_unlicensed_rows():
    parent = _parent(0)
    row = _locator(parent, 0, "1" * 64, "2" * 64, 800)
    assert _valid_locator(row)
    row["licenses"] = []
    assert not _valid_locator(row)


def test_admission_deduplicates_quarantines_and_caps(tmp_path: Path):
    parents = [_parent(0), _parent(1)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in parents))
    scan = tmp_path / "scan"
    _scan_shard(
        scan,
        manifest,
        parents[0],
        0,
        [
            _locator(parents[0], 0, "f" * 64, "3" * 64, 1_000),
            _locator(parents[0], 1, "1" * 64, "4" * 64, 1_200),
        ],
    )
    _scan_shard(
        scan,
        manifest,
        parents[1],
        1,
        [
            _locator(parents[1], 2, "0" * 64, "3" * 64, 1_000),
            _locator(parents[1], 3, "2" * 64, "5" * 64, 1_500),
            _locator(parents[1], 4, "3" * 64, "6" * 64, 2_000),
        ],
    )
    quarantine = _quarantine_registry(tmp_path / "quarantine", "4" * 64)
    output = tmp_path / "output"
    receipt = build_admission(
        manifest,
        scan,
        quarantine,
        output,
        expected_parents=2,
        maximum_text_bytes=3_000,
        output_shards=2,
        scratch_root=tmp_path,
    )
    assert receipt["counts"]["candidate_rows"] == 5
    assert receipt["counts"]["known_quarantine_rows_excluded"] == 1
    assert receipt["counts"]["exact_duplicate_rows_excluded"] == 1
    assert receipt["counts"]["byte_cap_excluded_rows"] == 1
    assert receipt["counts"]["admitted_rows"] == 2
    assert receipt["counts"]["admitted_text_utf8_bytes"] == 2_500
    admitted = []
    for descriptor in receipt["outputs"]["descriptors"]:
        admitted.extend(pq.read_table(output / descriptor["path"]).to_pylist())
    assert sorted(row["content_sha256"] for row in admitted) == ["3" * 64, "5" * 64]
    assert next(row for row in admitted if row["content_sha256"] == "3" * 64)[
        "source_row_identity_sha256"
    ] == "0" * 64
    assert receipt["global_exact_content_deduplication_complete"] is True
    assert receipt["training_ready"] is True


def test_admission_fails_on_missing_scan_parent(tmp_path: Path):
    parents = [_parent(0), _parent(1)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in parents))
    quarantine = _quarantine_registry(tmp_path / "quarantine", "4" * 64)
    try:
        build_admission(
            manifest,
            tmp_path / "missing",
            quarantine,
            tmp_path / "output",
            expected_parents=2,
            maximum_text_bytes=3_000,
            output_shards=2,
            scratch_root=tmp_path,
        )
    except (StackEduPracticalAdmissionError, RuntimeError):
        pass
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("missing scan parent was accepted")


def test_stokes_admission_job_uses_bounded_local_exact_dedup():
    script = Path(
        "scripts/admit_common_pile_stack_edu_practical_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=64G" in script
    assert "#SBATCH --exclude=ec65" in script
    assert "#SBATCH --no-requeue" in script
    assert "--maximum-text-bytes 150000000000" in script
    assert '--scratch-root "${TMPDIR:-/tmp}"' in script
