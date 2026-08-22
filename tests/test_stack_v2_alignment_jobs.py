from pathlib import Path

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "jobs" / "sai-freeze-stack-v2-current-python-snapshot-cpu.sbatch"
ALIGN = ROOT / "jobs" / "sai-align-stack-edu-current-cpu.sbatch"
CONTENT = ROOT / "jobs" / "sai-verify-stack-edu-content-cpu.sbatch"
SAFETY = ROOT / "jobs" / "sai-scan-stack-edu-content-safety-cpu.sbatch"
SAFETY_SELECT = ROOT / "jobs" / "sai-select-stack-edu-safety-candidates-cpu.sbatch"


def test_stack_v2_alignment_jobs_are_cpu_only_create_only_and_replay() -> None:
    for path in (SNAPSHOT, ALIGN, CONTENT, SAFETY, SAFETY_SELECT):
        script = path.read_text()
        assert "#SBATCH --no-requeue" in script
        assert "#SBATCH --gres" not in script
        assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in script
        assert 'status --short)"' in script
        assert "chmod 0444" in script
        assert "training" not in script.lower()
    snapshot = SNAPSHOT.read_text()
    assert "stack_v2_alignment" in snapshot
    assert 'test ! -e "$RECEIPT_OUTPUT"' in snapshot
    assert "freeze-snapshot" in snapshot
    assert "validate-snapshot" in snapshot
    assert "--access-evidence" in snapshot
    alignment = ALIGN.read_text()
    assert "stack_v2_alignment" in alignment
    assert 'test ! -e "$ALIGNED_OUTPUT"' in alignment
    assert 'test ! -e "$RECEIPT_OUTPUT"' in alignment
    assert "validate-alignment" in alignment
    content = CONTENT.read_text()
    assert "stack_edu_content verify" in content
    assert "stack_edu_content validate" in content
    assert 'test ! -e "$RECEIPT_OUTPUT"' in content
    assert "input must be sealed" in content
    safety = SAFETY.read_text()
    assert "stack_edu_safety scan" in safety
    assert "stack_edu_safety validate" in safety
    assert 'test ! -e "$FINDINGS_OUTPUT"' in safety
    assert 'test ! -e "$RECEIPT_OUTPUT"' in safety
    selection = SAFETY_SELECT.read_text()
    assert "stack_edu_safety_select select" in selection
    assert "stack_edu_safety_select validate" in selection
    assert 'test ! -e "$SELECTED_OUTPUT"' in selection
    assert 'test ! -e "$RECEIPT_OUTPUT"' in selection
