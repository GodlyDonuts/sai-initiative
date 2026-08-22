from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "sai-audit-stack-edu-cpu.sbatch"


def test_stack_edu_audit_job_is_cpu_only_and_fail_closed() -> None:
    script = JOB.read_text()
    assert "#SBATCH --gres" not in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "sai.data.stack_edu_audit audit" in script
    assert "sai.data.stack_edu_audit validate" in script
    assert 'test ! -e "$SAMPLE_OUTPUT"' in script
    assert 'test ! -e "$RECEIPT_OUTPUT"' in script
    assert "sbatch" not in script
    assert "srun" not in script
