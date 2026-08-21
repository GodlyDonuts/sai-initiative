from pathlib import Path


def _job() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "jobs" / "sai-launch-short-screen-development-mc-cpu.sbatch"
    ).read_text()


def test_fanout_is_cpu_only_bounded_and_waits_for_exact_completed_inputs() -> None:
    job = _job()
    assert "--no-requeue" in job
    assert "--gres=" not in job
    assert '""|NoDevFiles' in job
    assert "CPU evaluation launcher was exposed to a GPU" in job
    assert 'rev-parse --is-inside-work-tree)" = "true"' in job
    assert 'test -d "$SAI_ROOT/.git"' not in job
    assert 'short_screen_state" = "COMPLETED"' in job
    assert 'population_state" = "COMPLETED"' in job
    assert 'geometry_row["scale"] == "100m"' in job
    assert "101_000_000" in job
    assert "gated_gqa|gdn_hybrid|kda_mla_hybrid" in job
    assert "4b" not in job.lower().replace("four_b", "")
    assert '"$LOG_ROOT"/*' in job


def test_fanout_submits_exactly_two_independent_single_h100_evaluators() -> None:
    job = _job()
    assert job.count("sbatch --parsable") == 2
    assert job.count("env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin") == 2
    assert job.count('--chdir="$SAI_ROOT"') == 2
    assert "ALL,SAI_ROOT=" not in job
    assert job.count('"$EVALUATOR_JOB")') == 2
    assert "BENCHMARK=mmlu_pro" in job
    assert "BENCHMARK=musr" in job
    assert "--array" not in job
    assert "--dependency" not in job
    assert "scancel" in job
    assert "trap cancel_partial_fanout EXIT" in job
    assert '"gpu_jobs_submitted": 2' in job
    assert '"maximum_concurrent_single_h100_jobs": 2' in job
    assert '"retry": False' in job
    assert '"requeue": False' in job


def test_fanout_binds_all_scientific_and_runtime_identities() -> None:
    job = _job()
    for binding in (
        "EXPECTED_COMMIT",
        "FLA_ROOT",
        "ENVIRONMENT_RECEIPT",
        "ENVIRONMENT_RECEIPT_SHA256",
        "EVALUATOR_RUNTIME_SHA256",
        "GEOMETRY_SHA256",
        "SHORT_SCREEN_RESULT_SHA256",
        "CHECKPOINT_SHA256",
        "CHECKPOINT_MANIFEST_SHA256",
        "TRAINING_STREAM_IDENTITY",
        "TOKENIZER_SHA256",
        "POPULATION_AGGREGATE_SHA256",
        "BENCHMARK_SOURCE_SHA256",
        "DISJOINT_RECEIPT_SHA256",
        "TRAINING_SOURCE_SHA256",
        "EXPECTED_IDENTITY_ORDER_SHA256",
    ):
        assert binding in job
    assert "load_validated_model_state" in job
    assert "validate_short_screen_result" in job
    assert 'test ! -e "$DISPATCH"' in job
    assert "sai-short-screen-development-mc-dispatch-v1" in job
