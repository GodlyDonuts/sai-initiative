from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_build_is_one_cpu_job_per_exact_size() -> None:
    job = (
        ROOT / "scripts" / "build_combined_tokenizer_candidate_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --cpus-per-task=24" in job
    assert "#SBATCH --mem=96G" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "32k=32000|48k=48000|64k=64000" in job
    assert "${#sai_book_samples[@]} -eq 64" in job
    assert "${#sai_pleias_samples[@]} -eq 128" in job
    assert '--size "${SAI_TOKENIZER_NAME}=${SAI_TOKENIZER_SIZE}"' in job
    assert "--corpus" in job


def test_qualification_uses_all_candidates_and_identical_corpora() -> None:
    job = (
        ROOT / "scripts" / "qualify_combined_tokenizer_tournament_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --mem=192G" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "${#sai_book_samples[@]} -eq 64" in job
    assert "${#sai_pleias_samples[@]} -eq 128" in job
    for name in ("32k", "48k", "64k"):
        assert f'--candidate "{name}=' in job
    assert "SAI_TOKENIZER_PROTECTED_SUITE.jsonl" in job
    assert "--selected-48k-output" in job
    assert "sai.tokenizer.tournament_custody" in job
    assert "--pleias-samples-root" in job
    assert "--book-samples-root" in job
