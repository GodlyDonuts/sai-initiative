from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "sai-aggregate-stack-edu-language-cpu.sbatch"


def test_stack_edu_aggregate_job_is_cpu_only_and_replays_every_receipt() -> None:
    script = JOB.read_text()
    assert "#SBATCH --gres" not in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --no-requeue" in script
    assert "sai.data.stack_edu_aggregate aggregate" in script
    assert "sai.data.stack_edu_aggregate validate" in script
    assert 'IFS=: read -r -a receipt_paths <<< "$RECEIPTS"' in script
    assert 'test "${#receipt_paths[@]}" -ge 2' in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert "sbatch" not in script
    assert "srun" not in script
