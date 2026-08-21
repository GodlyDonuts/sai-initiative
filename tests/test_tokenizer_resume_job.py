from pathlib import Path


def test_tokenizer_continuation_is_bounded_to_failed_atomic_split() -> None:
    job = (
        Path(__file__).parents[1]
        / "jobs"
        / "sai-resume-tokenizer-tournament-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert '""|NoDevFiles' in job
    assert "TIMEOUT|FAILED" in job
    assert "sai-decontamination-receipt-v1" in job
    assert "sai-document-split-receipt-v1" in job
    assert 'split["scientific_promotion_allowed"] is False' in job
    assert 'split["near_duplicate_cluster_split_qualified"] is False' in job
    assert "sha256_file(source)" in job
    assert "for label, path in" in job
    assert '("source", source)' in job
    assert '("train", train)' in job
    assert '("development", development)' in job
    assert 'entry["sha256"] == sha256_file(path)' in job
    assert "sai-tokenizer-build-manifest-v1" in job
    assert "sha256_tree(root)" in job
    assert "build_required" in job
    assert "qualification_required" in job
    assert job.count("-m sai.tokenizer.build") == 1
    assert job.count("-m sai.tokenizer.qualification") == 1
    assert "sai.data.decontamination validate" not in job
    assert "sai.data.split" not in job
    assert "scientific_promotion_allowed" in job
