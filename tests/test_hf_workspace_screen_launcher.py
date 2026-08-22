from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_workspace_launcher_submits_matched_canary_then_full_jobs() -> None:
    job = (
        ROOT / "jobs" / "sai-launch-qwen35-0p8b-workspace-screen-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert 'case "${CUDA_VISIBLE_DEVICES:-}"' in job
    assert "submit_arm recurrent 256 '' 01:00:00 canary" in job
    assert "submit_arm reset_average 256 '' 01:00:00 canary" in job
    assert 'submit_arm recurrent 61035 "$canary_dependency" 08:00:00 full' in job
    assert 'submit_arm reset_average 61035 "$canary_dependency" 08:00:00 full' in job
    assert 'canary_dependency="$canary_recurrent:$canary_reset"' in job
    assert 'scancel "${admitted_jobs[@]}"' in job
    assert "maximum_concurrent_h100_jobs" in job
    assert "four_b_training_executed" in job


def test_workspace_launcher_replays_inputs_and_binds_dispatch() -> None:
    job = (
        ROOT / "jobs" / "sai-launch-qwen35-0p8b-workspace-screen-cpu.sbatch"
    ).read_text()
    for text in (
        "validate_mechanics_receipt(",
        "validate_frozen_stream(stream_root, verify_sources=True)",
        'stream["sequence_length"] != 2048',
        'stream["vocab_size"] != 248077',
        'str(value) in stream["prefix_utf8_bytes"]',
        '"git", "-C", os.environ["SAI_ROOT"], "ls-tree"',
        "python_path = pathlib.Path(os.path.realpath(sys.argv[4]))",
        'test "$(sacct -j "$MECHANICS_JOB_ID"',
        'test "$(sacct -j "$STREAM_JOB_ID"',
    ):
        assert text in job


def test_git_bound_job_count_includes_workspace_launcher() -> None:
    scripts = list((ROOT / "jobs").glob("*.sbatch"))
    bound = [
        path
        for path in scripts
        if "SAI_ROOT" in path.read_text() and "EXPECTED_COMMIT" in path.read_text()
    ]
    assert len(bound) == 79
