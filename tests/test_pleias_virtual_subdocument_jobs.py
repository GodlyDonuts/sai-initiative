from pathlib import Path


def test_virtual_signature_job_streams_without_gpu_or_bulk_output() -> None:
    job = Path(
        "scripts/run_pleias_virtual_subdocument_signature_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-127%8" in job
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "sai_scratch=${TMPDIR:-/tmp}" in job
    assert "sai.data.pleias_virtual_subdocument_signature shard" in job
    assert '--selection-root "${sai_selection}"' in job
    assert "sai_official_benchmark_boundary_index_20260824_r2" in job
    assert '--scratch-root "${sai_scratch}"' in job
    assert "upload" not in job.casefold()


def test_virtual_signature_aggregate_and_decision_are_dependency_ready() -> None:
    aggregate = Path(
        "scripts/aggregate_pleias_virtual_subdocument_signature_stokes.sbatch"
    ).read_text()
    decision = Path(
        "scripts/decide_pleias_virtual_subdocuments_stokes.sbatch"
    ).read_text()
    assert "sai.data.pleias_virtual_subdocument_signature aggregate" in aggregate
    assert "pleias-virtual-subdocument-signatures-20260826-r1" in aggregate
    assert "#SBATCH --array=0-15%16" in decision
    assert "#SBATCH --gres" not in decision
    assert "sai.data.pleias_subdocument_decision" in decision
    assert "pleias-virtual-subdocument-signatures-20260826-r1" in decision
    assert "pleias-virtual-subdocument-decision-20260826-r1" in decision


def test_virtual_internal_rewrite_jobs_preserve_source_safe_custody() -> None:
    shard = Path(
        "scripts/run_pleias_virtual_internal_rewrite_signature_stokes.sbatch"
    ).read_text()
    aggregate = Path(
        "scripts/aggregate_pleias_virtual_internal_rewrite_signature_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-127%8" in shard
    assert "#SBATCH --gres" not in shard
    assert "#SBATCH --no-requeue" in shard
    assert "sai_scratch=${TMPDIR:-/tmp}" in shard
    assert "sai.data.pleias_virtual_internal_rewrite_signature shard" in shard
    assert "pleias-virtual-subdocument-decision-20260826-r1" in shard
    assert '--scratch-root "${sai_scratch}"' in shard
    assert "upload" not in shard.casefold()
    assert (
        "sai.data.pleias_virtual_internal_rewrite_signature aggregate" in aggregate
    )
    assert "pleias-virtual-internal-signatures-20260826-r1" in aggregate


def test_virtual_cross_source_jobs_join_exact_component_signatures() -> None:
    decision = Path(
        "scripts/decide_virtual_cross_source_subdocuments_stokes.sbatch"
    ).read_text()
    aggregate = Path(
        "scripts/aggregate_virtual_cross_source_subdocument_decision_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-15%16" in decision
    assert "#SBATCH --gres" not in decision
    assert "#SBATCH --no-requeue" in decision
    assert "sai_scratch=${TMPDIR:-/tmp}" in decision
    assert "sai.data.cross_source_subdocument_decision" in decision
    assert "institutional-books-subdocument-signatures-20260826-r1" in decision
    assert "pleias-virtual-internal-signatures-20260826-r1" in decision
    assert "virtual-cross-source-subdocument-decision-20260826-r1" in decision
    assert "upload" not in decision.casefold()
    assert "sai.data.cross_source_subdocument_decision_aggregate" in aggregate
    assert "virtual-cross-source-subdocument-decision-20260826-r1" in aggregate


def test_virtual_final_reconstruction_jobs_do_not_persist_pleias_text() -> None:
    shard = Path(
        "scripts/reconstruct_pleias_virtual_cross_source_stokes.sbatch"
    ).read_text()
    aggregate = Path(
        "scripts/aggregate_pleias_virtual_cross_source_reconstruction_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-127%8" in shard
    assert "#SBATCH --gres" not in shard
    assert "#SBATCH --no-requeue" in shard
    assert "sai_scratch=${TMPDIR:-/tmp}" in shard
    assert "sai.data.pleias_virtual_cross_source_reconstruction shard" in shard
    assert "pleias-virtual-internal-signatures-20260826-r1" in shard
    assert "virtual-cross-source-subdocument-decision-20260826-r1" in shard
    assert "upload" not in shard.casefold()
    assert "sai.data.pleias_virtual_cross_source_reconstruction aggregate" in aggregate


def test_virtual_book_rewrite_uses_the_same_cross_source_decision() -> None:
    shard = Path(
        "scripts/rewrite_institutional_books_virtual_cross_source_stokes.sbatch"
    ).read_text()
    aggregate = Path(
        "scripts/aggregate_institutional_books_virtual_cross_source_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-63%16" in shard
    assert "#SBATCH --gres" not in shard
    assert "#SBATCH --no-requeue" in shard
    assert "sai.data.institutional_books_cross_source_subdocument_rewrite" in shard
    assert "virtual-cross-source-subdocument-decision-20260826-r1" in shard
    assert "institutional-books-virtual-cross-source-rewritten-20260826-r1" in shard
    assert (
        "sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate"
        in aggregate
    )
    assert "virtual-cross-source-subdocument-decision-20260826-r1" in aggregate
