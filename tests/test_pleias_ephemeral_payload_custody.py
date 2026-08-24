from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_path", "scratch_assignment"),
    [
        (
            "src/sai/data/pleias_production_materializer.py",
            'temporary = Path(payload_directory.name) / '
            '"benchmark_disjoint_candidates.parquet"',
        ),
        (
            "src/sai/data/pleias_subdocument_rewrite.py",
            'temporary = scratch / f"rewrite-partial-',
        ),
        (
            "src/sai/data/pleias_cross_source_subdocument_rewrite.py",
            "split: scratch / f\"{split}.rewrite.partial.",
        ),
    ],
)
def test_large_pleias_payloads_use_ephemeral_scratch(
    module_path: str, scratch_assignment: str
) -> None:
    source = Path(module_path).read_text()
    assert scratch_assignment in source
    assert "local_payload_removed_after_remote_verification\": True" in source
    assert source.index("output_root.mkdir(parents=True)") > source.index(
        'payload["receipt_sha256"]'
    )


@pytest.mark.parametrize(
    "job_path",
    [
        "scripts/run_pleias_production_materializer_stokes.sbatch",
        "scripts/rewrite_pleias_subdocuments_stokes.sbatch",
        "scripts/rewrite_pleias_cross_source_subdocuments_stokes.sbatch",
    ],
)
def test_large_pleias_jobs_bind_node_local_scratch(job_path: str) -> None:
    job = Path(job_path).read_text()
    assert "sai_scratch=${TMPDIR:-/tmp}" in job
    assert '--scratch-root "${sai_scratch}"' in job
