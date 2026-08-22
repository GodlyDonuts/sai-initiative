from __future__ import annotations

from pathlib import Path

from sai.data.fineweb_edu import _manifest, text_quality

ROOT = Path(__file__).resolve().parents[1]


def test_mechanics_source_manifest_is_exact_six_file_prefix() -> None:
    payload = _manifest(ROOT / "docs" / "SAI_FINEWEB_EDU_MECHANICS_SOURCE.json")
    assert len(payload["files"]) == 6
    assert sum(row["size"] for row in payload["files"]) == 12_914_392_174
    assert payload["files"][0]["path"] == "sample/100BT/000_00000.parquet"
    assert payload["files"][-1]["path"] == "sample/100BT/000_00005.parquet"


def test_quality_filter_accepts_clean_english_and_rejects_pathologies() -> None:
    clean = (
        "This is a detailed technical explanation of compiler optimization. " * 10
    ).strip()
    assert text_quality(clean)["accepted"] is True
    assert text_quality("tiny")["accepted"] is False
    repeated = ("Repeated educational line.\n" * 30) + (
        "Distinct explanatory material. " * 20
    )
    assert text_quality(repeated)["accepted"] is False
    controls = ("Technical language and scientific context. " * 20) + ("\x00" * 20)
    assert text_quality(controls)["accepted"] is False


def test_cpu_acquisition_job_holds_no_gpu_and_is_no_requeue() -> None:
    job = (ROOT / "jobs" / "sai-fineweb-edu-mechanics-cpu.sbatch").read_text()
    assert "--no-requeue" in job
    assert "--gres=" not in job
    assert "--acquire" in job
    assert "EXPECTED_COMMIT" in job
