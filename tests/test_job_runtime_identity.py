from pathlib import Path


def test_all_git_bound_jobs_admit_sealed_linked_worktrees() -> None:
    jobs = Path(__file__).parents[1] / "jobs"
    checked = 0
    for path in sorted(jobs.glob("*.sbatch")):
        script = path.read_text()
        if "SAI_ROOT" not in script or "EXPECTED_COMMIT" not in script:
            continue
        checked += 1
        assert 'test -d "$SAI_ROOT/.git"' not in script, path.name
        assert 'rev-parse --is-inside-work-tree)" = "true"' in script, path.name
        assert 'test ! -L "$SAI_ROOT"' in script, path.name
        assert 'rev-parse HEAD)" = "$EXPECTED_COMMIT"' in script, path.name
        assert 'status --short)"' in script, path.name
    assert checked == 67
