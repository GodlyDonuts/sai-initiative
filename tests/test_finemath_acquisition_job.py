from pathlib import Path


def test_finemath_acquisition_is_exact_resumable_and_candidate_only() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-acquire-finemath-4plus-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --no-requeue" in script
    assert "e92b25a616738fe95dc186b64dfb19f9c8525594" in script
    assert "len(rows) != 64" in script
    assert "18_365_184_633" in script
    assert '"source_admitted": False' in script
    assert '"training_authorized": False' in script
    assert "hf_hub_download" in script
    assert "partial FineMath member differs" in script
    assert 'find "$STAGE" -type f -exec chmod 0444' in script
