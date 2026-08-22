from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "sai-build-prerequisite-blind-review-cpu.sbatch"


def test_blind_review_packet_job_is_cpu_only_and_create_once() -> None:
    script = JOB.read_text()
    assert "#SBATCH --gres" not in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --mem=4G" in script
    assert "#SBATCH --time=00:15:00" in script
    assert "sai.data.prerequisite_review_packet build" in script
    assert "sai.data.prerequisite_review_packet validate" in script
    assert 'test ! -e "$REVIEW_OUTPUT"' in script
    assert 'test ! -e "$KEY_OUTPUT"' in script
    assert 'test ! -e "$RECEIPT_OUTPUT"' in script
    assert "sbatch" not in script
    assert "srun" not in script
