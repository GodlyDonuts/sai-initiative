from __future__ import annotations

import argparse
import json
from pathlib import Path

import sai.evaluation.population_refresh as refresh_module
from sai.data.token_stream import canonical_sha256
from sai.evaluation.population_refresh import SCHEMA, refresh


def test_refresh_schema_and_cpu_job_contract() -> None:
    assert SCHEMA == "sai-development-mc-populations-aggregate-v1"
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-refresh-development-mc-populations-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "DECONTAMINATION_JOB_ID" in job
    assert "sai.evaluation.population_refresh" in job
    assert "retry" not in job.lower()


def test_refresh_emits_canonical_aggregate_contract(tmp_path, monkeypatch) -> None:
    decontamination = tmp_path / "decontamination.json"
    decontamination.write_text("{}\n")

    def fake_convert(**kwargs):
        kwargs["output_source_path"].write_text("{}\n")
        kwargs["output_disjoint_receipt_path"].write_text("{}\n")
        receipt = {"receipt_sha256": "a" * 64}
        kwargs["output_conversion_receipt_path"].write_text(json.dumps(receipt))
        return receipt

    monkeypatch.setattr(refresh_module, "convert", fake_convert)
    digest = "b" * 64
    args = argparse.Namespace(
        source_commit="c" * 40,
        decontamination_receipt=decontamination,
        mmlu_questions=tmp_path / "mmlu.questions",
        mmlu_assessors=tmp_path / "mmlu.assessors",
        mmlu_questions_sha256=digest,
        mmlu_assessors_sha256=digest,
        mmlu_identity_order_sha256=digest,
        musr_questions=tmp_path / "musr.questions",
        musr_assessors=tmp_path / "musr.assessors",
        musr_questions_sha256=digest,
        musr_assessors_sha256=digest,
        musr_identity_order_sha256=digest,
        output_root=tmp_path / "output",
    )
    payload = refresh(args)
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256")
    assert set(payload) == {
        "schema",
        "status",
        "development_only",
        "official_benchmark_result",
        "architecture_promotion_allowed",
        "code_commit",
        "training_decontamination_receipt",
        "populations",
        "total_rows",
        "receipt_sha256",
    }
    assert payload["schema"] == SCHEMA
    assert payload["code_commit"] == "c" * 40
    assert payload["architecture_promotion_allowed"] is False
    assert "four_b_training_authorized" not in payload
    assert payload["total_rows"] == 12_788
    assert claimed == canonical_sha256(unsigned)
