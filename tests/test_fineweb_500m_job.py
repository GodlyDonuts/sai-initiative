import json
from pathlib import Path


def _job() -> str:
    return (
        Path(__file__).resolve().parents[1] / "jobs" / "sai-fineweb-edu-500m-cpu.sbatch"
    ).read_text()


def test_source_manifest_is_an_exact_superset_prefix() -> None:
    root = Path(__file__).resolve().parents[1]
    mechanics = json.loads(
        (root / "docs" / "SAI_FINEWEB_EDU_MECHANICS_SOURCE.json").read_text()
    )
    expanded = json.loads(
        (root / "docs" / "SAI_FINEWEB_EDU_500M_SCREEN_SOURCE.json").read_text()
    )
    assert expanded["schema"] == mechanics["schema"]
    assert expanded["dataset"] == mechanics["dataset"]
    assert expanded["revision"] == mechanics["revision"]
    assert (
        expanded["upstream_full_manifest_sha256"]
        == mechanics["upstream_full_manifest_sha256"]
    )
    assert expanded["files"][:6] == mechanics["files"]
    assert len(expanded["files"]) == 30
    assert expanded["selection"] == {
        "method": "complete_prefix_sorted_tree",
        "selected_count": 30,
    }
    assert expanded["selected_bytes"] == sum(row["size"] for row in expanded["files"])
    assert [row["path"] for row in expanded["files"]] == [
        f"sample/100BT/{group:03d}_{index:05d}.parquet"
        for group in range(3)
        for index in range(10)
    ]


def test_source_job_is_cpu_only_pinned_and_uses_node_local_scratch() -> None:
    job = _job()
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres=" not in job
    assert "CPU source acquisition was exposed to a GPU" in job
    assert "${SLURM_TMPDIR:?" in job
    assert 'SOURCE_ROOT="$SLURM_TMPDIR/sai-fineweb-edu-500m-$SLURM_JOB_ID"' in job
    assert "required_bytes=75100000000" in job
    assert "sample/100BT/{group:03d}_{index:05d}.parquet" in job
    assert "64_562_434_300" in job
    assert 'selected_count": 30' in job


def test_source_job_cleans_only_the_exact_node_local_tree() -> None:
    job = _job()
    assert 'case "$SOURCE_ROOT" in' in job
    assert '"$SLURM_TMPDIR"/sai-fineweb-edu-500m-"$SLURM_JOB_ID"' in job
    assert 'find "$SOURCE_ROOT" -xdev -depth -delete' in job
    assert 'test "$(stat -c %u "$SOURCE_ROOT")" = "$(id -u)"' in job
    assert "rm -rf" not in job
    assert "trap cleanup_source EXIT" in job


def test_source_job_validates_outputs_before_cleanup() -> None:
    job = _job()
    assert "sai.data.fineweb_edu" in job
    assert '--source-root "$SOURCE_ROOT"' in job
    assert '--output "$OUTPUT"' in job
    assert '--receipt "$RECEIPT"' in job
    assert 'payload["status"] == "passed"' in job
    assert 'len(payload["source_receipts"]) == 30' in job
    assert "cleanup_source\ntrap - EXIT" in job
