from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "jobs" / name).read_text()


def test_stage_launches_two_matched_benchmark_fanouts_only_after_nll_pass() -> None:
    job = _read("sai-stage-curriculum-development-mc-cpu.sbatch")
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'payload["curriculum_order_supported_by_heldout_nll"] is True' in job
    assert 'payload["heldout_phase_no_regression"] is True' in job
    assert "for arm in curriculum order_control" in job
    assert job.count("sbatch --parsable") == 2
    assert "sai-launch-short-screen-development-mc-cpu.sbatch" in job
    assert "sai-stage-curriculum-development-mc-comparison-cpu.sbatch" in job
    assert '"expected_evaluation_gpu_jobs": 18' in job
    assert '"maximum_concurrent_single_h100_jobs": 18' in job
    assert '"four_b_training_authorized": False' in job
    assert "trap cancel_partial_graph EXIT" in job


def test_continuation_extracts_live_ids_and_dependency_stages_benchmarks() -> None:
    job = _read("sai-stage-curriculum-benchmark-continuation-cpu.sbatch")
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'payload["schema"] == "sai-curriculum-order-screen-dispatch-v1"' in job
    assert 'payload["training_tokens_per_arm"] == 499_998_720' in job
    assert 'quota["minimum_headroom_kib"] == 25_165_824' in job
    assert 'quota["minimum_headroom_files"] == 10_000' in job
    assert 'dependency="$comparison_job"' in job
    assert 'sacct -X -j "$job_id"' in job
    assert 'dependency="$comparison_job:$POPULATION_JOB_ID"' not in job
    assert job.count("sbatch --parsable") == 1
    assert "sai-stage-curriculum-development-mc-cpu.sbatch" in job
    assert (
        '"benchmark_work_condition": "heldout_nll_and_every_phase_nonregression"' in job
    )
    assert '"gpu_jobs_submitted": 0' in job
    assert '"four_b_training_authorized": False' in job


def test_comparison_stage_binds_four_terminals_and_eighteen_h100_jobs() -> None:
    job = _read("sai-stage-curriculum-development-mc-comparison-cpu.sbatch")
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'test "${#terminal_ids[@]}" = 4' in job
    assert 'test "${#h100_ids[@]}" = 18' in job
    assert job.count("sbatch --parsable") == 1
    assert '--dependency="afterok:$dependency"' in job
    assert "sai-compare-curriculum-development-benchmarks-cpu.sbatch" in job
    assert '"expected_completed_single_h100_evaluations": 18' in job
    assert '"comparison_terminal_dependencies": 4' in job
    assert '"four_b_training_authorized": False' in job


def test_terminal_comparator_requires_all_underlying_h100_accounting() -> None:
    job = _read("sai-compare-curriculum-development-benchmarks-cpu.sbatch")
    assert "H100_EVALUATION_JOB_IDS" in job
    assert 'test "${#h100_job_ids[@]}" = 18' in job
    assert 'test "$row" = "COMPLETED|0:0|0"' in job
    assert "sai.evaluation.curriculum_benchmark_compare" in job
