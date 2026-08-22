from pathlib import Path


def _job() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-compare-curriculum-development-benchmarks-cpu.sbatch"
    ).read_text()


def test_comparator_is_cpu_only_one_shot_and_source_clean() -> None:
    job = _job()
    assert "--gres=" not in job
    assert "#SBATCH --no-requeue" in job
    assert "benchmark comparator was exposed to a GPU" in job
    assert 'rev-parse HEAD)" = "$EXPECTED_COMMIT"' in job
    assert 'status --short)"' in job
    assert "retry" not in job.lower()
    assert "scancel" not in job
    assert "sbatch --parsable" not in job


def test_comparator_requires_exact_terminal_accounting_and_all_results() -> None:
    job = _job()
    for binding in (
        "ORDER_COMPARISON_JOB_ID",
        "CURRICULUM_MMLU_JOB_ID",
        "CURRICULUM_MUSR_JOB_ID",
        "CONTROL_MMLU_JOB_ID",
        "CONTROL_MUSR_JOB_ID",
        "ORDER_COMPARISON",
        "CURRICULUM_MMLU",
        "CURRICULUM_MUSR",
        "CONTROL_MMLU",
        "CONTROL_MUSR",
    ):
        assert binding in job
    assert "State,ExitCode,Restarts" in job
    assert "COMPLETED|0:0|0" in job
    assert "sai.evaluation.curriculum_benchmark_compare" in job
    assert 'test ! -e "$OUTPUT"' in job
