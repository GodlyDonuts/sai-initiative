from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "jobs" / name).read_text()


def test_100m_stage_submits_three_family_launchers_and_one_comparison_stage() -> None:
    job = _read("sai-stage-100m-250m-development-mc-cpu.sbatch")
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert '""|NoDevFiles' in job
    assert "sai-100m-250m-token-screen-dispatch-v1" in job
    assert "SCREEN_SOURCE_COMMIT" in job
    assert 'families = ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"]' in job
    assert 'dispatch["training_tokens"] == 249_999_360' in job
    assert job.count("sbatch --parsable") == 2
    assert 'for row in "${validated[@]:4}"' in job
    assert '--dependency="afterok:$training_job"' in job
    assert "sai-launch-short-screen-development-mc-cpu.sbatch" in job
    assert "sai-stage-short-screen-development-mc-comparison-cpu.sbatch" in job
    assert '"expected_evaluation_gpu_jobs": 27' in job
    assert '"expected_evaluation_cpu_merge_jobs": 3' in job
    assert '"expected_total_evaluation_jobs": 30' in job
    assert '"maximum_concurrent_single_h100_jobs": 27' in job
    assert '"gpus_per_evaluation_job": 1' in job
    assert '"scientific_promotion_authorized": False' in job
    assert '"four_b_training_authorized": False' in job
    assert "trap cancel_partial_graph EXIT" in job
    assert "scancel" in job
    assert "--array" not in job


def test_comparison_stage_binds_27_h100s_and_six_terminal_dependencies() -> None:
    job = _read("sai-stage-short-screen-development-mc-comparison-cpu.sbatch")
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'payload["schema"] == "sai-short-screen-development-mc-dispatch-v1"' in job
    assert '["mmlu_pro", "musr"]' in job
    assert 'test "${#evaluation_job_ids[@]}" = 6' in job
    assert 'test "${#h100_evaluation_job_ids[@]}" = 27' in job
    assert 'assert payload["gpu_jobs_submitted"] == 9' in job
    assert 'assert payload["cpu_merge_jobs_submitted"] == 1' in job
    assert 'assert mmlu["shard_count"] == 8' in job
    assert 'not pathlib.Path(job["output"]).exists()' not in job
    assert job.count("sbatch --parsable") == 1
    assert '--dependency="afterok:$dependency"' in job
    assert "sai-short-screen-development-mc-comparison-cpu.sbatch" in job
    assert '"expected_completed_single_h100_evaluations": 27' in job
    assert '"expected_cpu_merges": 3' in job
    assert '"comparison_terminal_dependencies": 6' in job
    assert '"gpu_jobs_submitted": 0' in job
    assert '"four_b_training_authorized": False' in job


def test_comparison_job_requires_27_h100s_six_terminals_and_exact_matrix() -> None:
    job = _read("sai-short-screen-development-mc-comparison-cpu.sbatch")
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'test "${#evaluation_job_ids[@]}" = 6' in job
    assert 'test "${#h100_evaluation_job_ids[@]}" = 27' in job
    assert "H100_EVALUATION_JOB_IDS" in job
    assert 'test "$state" = "COMPLETED"' in job
    assert 'test "$exit_code" = "0:0"' in job
    assert 'test "$restarts" = "0"' in job
    assert 'test "${#comparison_arguments[@]}" = 12' in job
    assert "sai.evaluation.short_screen_compare" in job
    assert '--output "$OUTPUT"' in job
    assert "sbatch" not in job
