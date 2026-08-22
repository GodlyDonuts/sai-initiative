from pathlib import Path

ROOT = Path(__file__).parents[1]
EXTRACT = ROOT / "jobs" / "sai-freeze-stack-edu-candidates-cpu.sbatch"
AGGREGATE = ROOT / "jobs" / "sai-aggregate-stack-edu-candidates-cpu.sbatch"


def test_candidate_jobs_are_cpu_only_create_only_and_replay_outputs() -> None:
    for path in (EXTRACT, AGGREGATE):
        script = path.read_text()
        assert "#SBATCH --no-requeue" in script
        assert "#SBATCH --gres" not in script
        assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in script
        assert 'status --short)"' in script
        assert 'test ! -e "$CANDIDATES_OUTPUT"' in script
        assert 'test ! -e "$RECEIPT_OUTPUT"' in script
        assert "chmod 0444" in script
        assert "stack_edu_candidates" in script
        assert "training" not in script.lower()
    assert "validate-shard" in EXTRACT.read_text()
    assert "validate-aggregate" in AGGREGATE.read_text()
    assert "--shard-receipt" in AGGREGATE.read_text()
