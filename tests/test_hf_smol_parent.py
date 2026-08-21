from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.evaluation.hf_smol_parent import (
    EXPECTED_MODEL_CLASS,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_VOCAB_SIZE,
    SNAPSHOT_SPEC,
    SmolParentError,
    validate_smol_mechanics_receipt,
)


def test_smol_parent_identity_is_exact() -> None:
    assert EXPECTED_MODEL_CLASS == "SmolLM3ForCausalLM"
    assert EXPECTED_PARAMETER_COUNT == 3_075_098_624
    assert EXPECTED_VOCAB_SIZE == 128_256
    assert SNAPSHOT_SPEC.repository == "HuggingFaceTB/SmolLM3-3B"
    assert SNAPSHOT_SPEC.revision == "a07cc9a04f16550a088caea529712d1d335b0ac1"
    assert SNAPSHOT_SPEC.tree_sha256 == (
        "6badcd593aee3052e3d66afb315b979e2cc62c4a61f9cef31c07203912478a0f"
    )


def test_smol_mechanics_is_one_h100_offline_and_no_training() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-smollm3-3b-mechanics-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "evc50" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
    assert "sai.evaluation.hf_smol_parent" in job
    assert "retry" not in job.lower()
    assert "train" not in job.lower().replace("training", "")


def test_smol_mechanics_validator_rejects_resigned_training_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "manifest.json"
    restoration = tmp_path / "restoration.json"
    snapshot = {"tree_sha256": SNAPSHOT_SPEC.tree_sha256}
    monkeypatch.setattr(
        "sai.evaluation.hf_smol_parent.validate_external_snapshot",
        lambda *args, **kwargs: snapshot,
    )
    payload = {
        "schema": "sai-smollm3-3b-text-mechanics-v1",
        "status": "pass",
        "model_root": str(model.resolve()),
        "manifest_path": str(manifest.resolve()),
        "restoration_receipt_path": str(restoration.resolve()),
        "runtime": {
            "snapshot": snapshot,
            "model_class": EXPECTED_MODEL_CLASS,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "all_parameters_and_buffers_cuda_zero": True,
        },
        "environment": {"gpu_name": "NVIDIA H100 PCIe"},
        "forward": {"finite": True, "logits_shape": [1, 1, EXPECTED_VOCAB_SIZE]},
        "training_executed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_state_unchanged": True,
        "architecture_result": False,
        "four_b_training_executed": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt = tmp_path / "mechanics.json"
    receipt.write_text(json.dumps(payload))
    file_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert (
        validate_smol_mechanics_receipt(
            receipt,
            expected_file_sha256=file_sha,
            model_root=model,
            manifest_path=manifest,
            restoration_receipt_path=restoration,
        )["status"]
        == "pass"
    )

    payload["training_executed"] = True
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    payload["receipt_sha256"] = canonical_sha256(unsigned)
    receipt.write_text(json.dumps(payload))
    with pytest.raises(SmolParentError, match="evidence differs"):
        validate_smol_mechanics_receipt(
            receipt,
            expected_file_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            model_root=model,
            manifest_path=manifest,
            restoration_receipt_path=restoration,
        )
