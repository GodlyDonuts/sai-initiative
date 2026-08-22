from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sai.readiness import BENCHMARK_ROWS, DATA_ROLES, ReadinessError, validate


def artifact(tmp_path: Path, name: str) -> dict[str, object]:
    path = tmp_path / name
    content = f"artifact:{name}\n".encode()
    path.write_bytes(content)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def manifest(tmp_path: Path) -> dict:
    data = {}
    for index, role in enumerate(sorted(DATA_ROLES)):
        data[role] = {
            **artifact(tmp_path, f"{role}.jsonl"),
            "rows": 10 + index,
            "identity_sha256": f"{index + 1:064x}",
            "contamination_report_sha256": f"{index + 10:064x}",
        }
    benchmarks = {
        name: {
            **artifact(tmp_path, f"{name}.jsonl"),
            "rows": rows,
            "version": "frozen-v1",
            "official_scoring": True,
        }
        for name, rows in BENCHMARK_ROWS.items()
    }
    shared = {
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "updates": 64,
        "skill_tokens": 100_000,
        "replay_tokens": 50_000,
        "seed": 20260821,
        "role": "sai_candidate",
        "replay_weight": 0.25,
        "output": "candidate.pt",
    }
    control = {
        **shared,
        "role": "equal_compute_control",
        "replay_weight": 0,
        "output": "control.pt",
    }
    return {
        "schema": "sai-4b-pretraining-readiness-v1",
        "status": "prepared",
        "training_hold": True,
        "official_training_order_received": False,
        "gpu_jobs_submitted": 0,
        "training_updates_completed": 0,
        "parent": {
            "model_id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "license": "Apache-2.0",
            "manifest": artifact(tmp_path, "parent.SHA256SUMS"),
        },
        "runtime_manifest": artifact(tmp_path, "runtime.SHA256SUMS"),
        "environment_receipt": artifact(tmp_path, "environment.json"),
        "tokenizer_audit": artifact(tmp_path, "tokenizer.json"),
        "data": data,
        "benchmarks": benchmarks,
        "training": {
            "candidate": shared,
            "equal_compute_control": control,
        },
        "resource_plan": {
            "independent_single_h100_jobs": True,
            "estimated_h100_hours": 12.5,
        },
        "checks": {
            "parent_manifest_verified": True,
            "data_decontamination_verified": True,
            "replay_disjoint_from_benchmarks": True,
            "tokenizer_roundtrip_verified": True,
            "evaluator_official_scoring_verified": True,
            "candidate_control_compute_matched": True,
            "runtime_tests_passed": True,
            "zero_gpu_jobs_submitted": True,
            "no_training_performed": True,
        },
    }


def test_complete_preparation_still_requires_user_order(tmp_path: Path) -> None:
    receipt = validate(manifest(tmp_path))
    assert receipt["schema"] == "sai-4b-qwen-replay-rehearsal-receipt-v1"
    assert receipt["status"] == "provisional_qwen_replay_rehearsal_complete"
    assert not receipt["sai_4b_pretraining_ready"]
    assert receipt["selected_300m_1b_architecture_receipts_required"]
    assert receipt["selected_tokenizer_and_base_stream_receipts_required"]
    assert not receipt["training_authorized"]
    assert receipt["official_training_order_required"]


def test_missing_readiness_check_fails_closed(tmp_path: Path) -> None:
    payload = manifest(tmp_path)
    del payload["checks"]["no_training_performed"]
    with pytest.raises(ReadinessError, match="checks are incomplete"):
        validate(payload)


def test_unmatched_control_fails_closed(tmp_path: Path) -> None:
    payload = manifest(tmp_path)
    payload["training"]["equal_compute_control"]["updates"] = 63
    with pytest.raises(ReadinessError, match="differences are not exact"):
        validate(payload)


def test_tampered_artifact_fails_closed(tmp_path: Path) -> None:
    payload = manifest(tmp_path)
    Path(payload["parent"]["manifest"]["path"]).write_text("tampered\n")
    with pytest.raises(ReadinessError, match="artifact content differs"):
        validate(payload)


def test_any_execution_before_order_fails_closed(tmp_path: Path) -> None:
    payload = manifest(tmp_path)
    payload["gpu_jobs_submitted"] = 1
    with pytest.raises(ReadinessError, match="zero-execution"):
        validate(payload)
