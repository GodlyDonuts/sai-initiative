from pathlib import Path


def test_selection_reserves_book_headroom_and_uses_no_gpu() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "select_pleias_production_bytes_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres=" not in job
    assert "--maximum-bytes 2000000000000" in job
    assert job.count("-m sai.data.pleias_production_byte_selection") == 1
