from pathlib import Path


def test_semantic_learnability_job_is_create_only_cpu_and_replays_all_evidence() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs" / "sai-compose-semantic-learnability-cpu.sbatch").read_text()
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres=" not in job
    assert "#SBATCH --array" not in job
    assert "sbatch " not in job
    assert "CPU curriculum composer was exposed to a GPU" in job
    assert 'test ! -e "$OUTPUT"' in job
    assert "--parent-stream" in job
    assert "--scores" in job
    assert "--taxonomy" in job
    assert "--curriculum-receipt" in job
    assert "--annotations" in job
    assert "--progression-report" in job
    assert "-m sai.data.semantic_learnability_curriculum build" in job
