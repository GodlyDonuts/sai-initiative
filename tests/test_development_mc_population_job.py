from pathlib import Path


def test_cpu_population_job_freezes_exact_two_benchmark_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "jobs" / "sai-build-development-mc-populations-cpu.sbatch"
    ).read_text()
    assert "--no-requeue" in job
    assert "--gres=" not in job
    assert '""|NoDevFiles' in job
    assert "CPU population job was exposed to a GPU" in job
    assert "DECONTAMINATION_JOB_ID" in job
    assert 'decontamination_state" = "COMPLETED"' in job
    assert "fineweb_edu_mechanics_admitted_be505b6_r1.receipt.json" in job
    assert "public_bench_qwen9_766196_r1/data/mmlu_pro/full.questions.jsonl" in job
    assert "public_bench_qwen9_766196_r1/data/musr/full.assessors.jsonl" in job
    assert "--expected-rows 12032" in job
    assert "--expected-rows 756" in job
    assert "MMLU_IDENTITY_ORDER_SHA256" in job
    assert "MUSR_IDENTITY_ORDER_SHA256" in job
    assert '"$LOG_ROOT"/*' in job
    assert "sai-development-mc-populations-aggregate-v1" in job
    assert 'test ! -e "$path"' in job
    assert "#SBATCH --array" not in job


def test_population_job_has_one_create_only_call_per_benchmark() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "jobs" / "sai-build-development-mc-populations-cpu.sbatch"
    ).read_text()
    assert job.count("-m sai.evaluation.population_builder") == 2
    assert job.count("--output-source") == 2
    assert job.count("--output-disjoint-receipt") == 2
    assert job.count("--output-conversion-receipt") == 2
    assert "--benchmark mmlu_pro" in job
    assert "--benchmark musr" in job
    assert "trap cleanup_partial_publication EXIT" in job
