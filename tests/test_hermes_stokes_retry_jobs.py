from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_pleias_hermes_worker_is_resumable_under_provider_throttling() -> None:
    job = (ROOT / "scripts/run_pleias_semantic_compiler_stokes.sbatch").read_text()
    assert "stealth/ox-alpha" in job
    assert "https://openrouter.ai/api/v1" in job
    assert "--api-key-env OPENROUTER_API_KEY" in job
    assert "if [[ -f \"${sai_summary}\" ]]" in job
    assert "while ! python -m sai.data.nous_compiler_worker" in job
    assert "sleep 60" in job
    assert "#SBATCH --no-requeue" in job


def test_book_hermes_worker_is_resumable_under_provider_throttling() -> None:
    job = (
        ROOT / "scripts/run_institutional_books_semantic_compiler_stokes.sbatch"
    ).read_text()
    assert "stealth/ox-alpha" in job
    assert "https://openrouter.ai/api/v1" in job
    assert "--api-key-env OPENROUTER_API_KEY" in job
    assert "if [[ -f \"${sai_summary}\" ]]" in job
    assert "while ! python -m sai.data.nous_book_compiler_worker" in job
    assert "sleep 60" in job
    assert "#SBATCH --no-requeue" in job
