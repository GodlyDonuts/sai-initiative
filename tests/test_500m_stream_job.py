from pathlib import Path


def _job() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-freeze-500m-screen-stream-cpu.sbatch"
    ).read_text()


def test_stream_job_is_cpu_only_exact_and_nonretrying() -> None:
    job = _job()
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres=" not in job
    assert "CPU stream freezer was exposed to a GPU" in job
    assert 'rev-parse HEAD)" = "$EXPECTED_COMMIT"' in job
    assert 'test ! -e "$TRAIN_STREAM"' in job
    assert "499_998_720" in job
    assert "244_140" in job


def test_stream_job_freezes_every_declared_budget_prefix() -> None:
    job = _job()
    expected = (
        256,
        48_896,
        61_035,
        109_824,
        122_070,
        183_040,
        183_105,
        244_140,
    )
    for value in expected:
        assert f"--prefix-sequences {value}" in job
        assert f'"{value}"' in job
    assert "--sequence-length 2048" in job
    assert 'stream["sequence_length"] == 2_048' in job
    assert 'stream["sequences"] == 244_140' in job


def test_stream_job_reopens_corpus_and_stream_evidence() -> None:
    job = _job()
    assert "CURRICULUM_RECEIPT" in job
    assert "validate_curriculum_split" in job
    assert 'receipt["status"] == "qualified"' in job
    assert 'receipt["split_qualified"] is True' in job
    assert 'train["curriculum_qualified"] is True' in job
    assert 'all(train["progression_checks"].values())' in job
    assert 'train["sha256"] == sha256_file(corpus_path)' in job
    assert "--source-qualification-sha256" in job
    assert '--curriculum-receipt "$CURRICULUM_RECEIPT"' in job
    assert "--curriculum-phase-sequences grounding=48896" in job
    assert "--curriculum-phase-sequences integration=60928" in job
    assert "--curriculum-phase-sequences reasoning=73216" in job
    assert "--curriculum-phase-sequences specialization=61100" in job
    assert "--require-all-curriculum-phases-at-prefix 244140" in job
    for value in (61_035, 122_070, 183_105):
        assert f"--require-all-curriculum-phases-at-prefix {value}" not in job
    assert 'stream["source_qualification_sha256"] == sys.argv[3]' in job
    assert (
        'stream["curriculum"]["all_required_prefixes_cover_every_phase"] is True' in job
    )
    assert (
        'stream["curriculum"]["phase_order_contiguous_at_every_prefix"] is True' in job
    )
    assert 'stream["curriculum"]["phase_token_budget_enforced"] is True' in job
    assert "phase_boundaries == [48_896, 109_824, 183_040]" in job
    assert "boundary % 256 == 0" in job
    assert "validate_frozen_stream" in job
    assert "verify_sources=True" in job
    assert 'stream["benchmark_disjoint"] is True' in job
    assert 'stream["cross_document_targets_masked"] is True' in job
