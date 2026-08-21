from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.training.lineage import (
    CompletedRunLineageError,
    canonical_sha256,
    sha256_file,
    validate_receipt,
)

HEX = "ab" * 32


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fixture_bundle(
    root: Path, role: str = "workspace_treatment", *, salt: str = "adaptive"
) -> dict:
    run_identity = hashlib.sha256(f"run-{salt}".encode()).hexdigest()
    group = "02" * 32
    budget = {
        "optimizer_steps": 2,
        "sequences": 4,
        "valid_tokens": 8,
        "admitted_utf8_bytes": 16,
        "modeled_training_flops": 32,
    }
    hashes = {
        "tokenizer_sha256": "10" * 32,
        "ordered_stream_sha256": "11" * 32,
        "environment_sha256": "12" * 32,
        "system_config_sha256": "13" * 32,
        "workspace_plan_sha256": "14" * 32,
        "workspace_candidate_identity_sha256": "15" * 32,
    }
    parent_fast_state = canonical_sha256(
        [
            {
                "name": "fast_path_state_sha256.weight",
                "dtype": "float32",
                "shape": [1],
                "raw_little_endian_sha256": hashlib.sha256(
                    b"parent-fast-path-state"
                ).hexdigest(),
            }
        ]
    )
    planned = {
        "run_identity_sha256": run_identity,
        "comparison_group_sha256": group,
        "role": role,
        "changed_factor": (
            "latent_workspace"
            if role == "workspace_treatment"
            else "fast_path_capacity"
        ),
        "scale": "300m",
        "mixer_family": "gated_gqa",
        "contrast": "iso_flop",
        "seed": 20260821,
        "parent_completed_run_receipt_sha256": "21" * 32,
        "parent_checkpoint_tree_sha256": "22" * 32,
        "parent_fast_path_state_sha256": parent_fast_state,
        "training_budget": budget,
        **hashes,
    }
    plan = {
        "schema": "sai-300m-adaptive-experiment-plan-v1",
        "status": "frozen",
        "training_authorized": False,
        "official_training_order_received": False,
        "runs": [planned],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_descriptor = write_json(root / "plan.json", plan)
    authorization = {
        "schema": "sai-training-authorization-receipt-v1",
        "status": "authorized",
        "official_training_order_received": True,
        "training_authorized": True,
        "terminal_public_board_accessed": False,
        "plan_sha256": plan_descriptor["sha256"],
        "authorized_run_identities": [run_identity],
    }
    authorization["receipt_sha256"] = canonical_sha256(authorization)
    authorization_descriptor = write_json(root / "authorization.json", authorization)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    checkpoint_bytes = f"weights-{salt}".encode()
    (checkpoint / "model.safetensors").write_bytes(checkpoint_bytes)
    member = {
        "path": "model.safetensors",
        "bytes": len(checkpoint_bytes),
        "sha256": sha256_file(checkpoint / "model.safetensors"),
    }
    projection_descriptors = {}
    for component in (
        "system_state_sha256",
        "fast_path_state_sha256",
        "slow_path_state_sha256",
    ):
        raw_identity = (
            "parent-fast-path-state"
            if component == "fast_path_state_sha256" and role == "workspace_treatment"
            else f"{salt}-{component}"
        )
        tensor_rows = [
            {
                "name": f"{component}.weight",
                "dtype": "float32",
                "shape": [1],
                "raw_little_endian_sha256": hashlib.sha256(
                    raw_identity.encode()
                ).hexdigest(),
            }
        ]
        projection = {
            "schema": "sai-tensor-state-projection-v1",
            "status": "complete",
            "component": component,
            "tensors": tensor_rows,
            "state_sha256": canonical_sha256(tensor_rows),
        }
        descriptor = write_json(root / f"{component}.json", projection)
        descriptor["state_sha256"] = projection["state_sha256"]
        projection_descriptors[component] = descriptor
    fast_state = projection_descriptors["fast_path_state_sha256"]["state_sha256"]
    if role == "workspace_treatment":
        assert fast_state == parent_fast_state
    receipt = {
        "schema": "sai-completed-run-lineage-receipt-v1",
        "status": "complete",
        "scientific_status": "complete",
        "training_authorized": True,
        "official_training_order_received": True,
        "terminal_public_board_accessed": False,
        "created_at_utc": "2026-08-21T00:00:00Z",
        "role": role,
        "run_identity_sha256": run_identity,
        "comparison_group_sha256": group,
        "changed_factor": (
            "latent_workspace"
            if role == "workspace_treatment"
            else "fast_path_capacity"
        ),
        "plan": plan_descriptor,
        "authorization": authorization_descriptor,
        "parent": {
            "completed_run_receipt_sha256": "21" * 32,
            "checkpoint_tree_sha256": "22" * 32,
            "fast_path_state_sha256": parent_fast_state,
        },
        "immutable_inputs": {
            "architecture_sha256": "30" * 32,
            "geometry_sha256": "31" * 32,
            **hashes,
            "source_tree_sha256": "32" * 32,
            "runtime_sha256": "33" * 32,
            "kernel_contract_sha256": "34" * 32,
        },
        "execution": {
            "attempts": [
                {
                    "job_id": "1",
                    "attempt_index": 0,
                    "host": "test-host",
                    "gpu_identity_sha256": "40" * 32,
                    "exit_code": 0,
                    "signal": 0,
                    "restarts": 0,
                    "committed_update_start": 0,
                    "committed_update_end": 2,
                    "stdout_sha256": "41" * 32,
                    "stderr_sha256": "42" * 32,
                    "accounting_sha256": "43" * 32,
                }
            ],
            **budget,
            "skipped_updates": 0,
            "nonfinite_updates": 0,
            "overflow_updates": 0,
            "measured_gpu_seconds": 1.5,
            "update_ledger_sha256": "44" * 32,
        },
        "checkpoint_tree": {
            "root": "checkpoint",
            "format": "safetensors",
            "files": [member],
            "file_count": 1,
            "total_bytes": len(checkpoint_bytes),
            "tree_sha256": canonical_sha256([member]),
        },
        "state_projections": projection_descriptors,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def resign(receipt: dict) -> None:
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_completed_lineage_reopens_every_portable_artifact(tmp_path: Path) -> None:
    receipt = fixture_bundle(tmp_path)
    assert validate_receipt(receipt, tmp_path) == receipt


def test_zero_commit_infrastructure_attempt_can_precede_exact_success(
    tmp_path: Path,
) -> None:
    receipt = fixture_bundle(tmp_path)
    successful = receipt["execution"]["attempts"][0]
    successful["attempt_index"] = 1
    failed = {
        **successful,
        "job_id": "failed-job",
        "attempt_index": 0,
        "exit_code": 1,
        "committed_update_end": 0,
    }
    receipt["execution"]["attempts"] = [failed, successful]
    resign(receipt)
    assert validate_receipt(receipt, tmp_path) == receipt

    receipt["execution"]["attempts"][0]["committed_update_end"] = 1
    resign(receipt)
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, tmp_path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row["execution"].update(skipped_updates=1),
        lambda row: row["execution"]["attempts"][0].update(restarts=1),
        lambda row: row["execution"]["attempts"][0].update(committed_update_start=1),
        lambda row: row["execution"].update(optimizer_steps=3),
        lambda row: row["state_projections"].update(fast_path_state_sha256="99" * 32),
        lambda row: row.update(terminal_public_board_accessed=True),
        lambda row: row.update(training_authorized=False),
    ],
)
def test_resigned_scientific_or_execution_tampering_fails_closed(
    tmp_path: Path, mutator
) -> None:
    receipt = fixture_bundle(tmp_path)
    mutator(receipt)
    resign(receipt)
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, tmp_path)


def test_checkpoint_change_extra_member_and_symlink_fail_closed(tmp_path: Path) -> None:
    receipt = fixture_bundle(tmp_path)
    (tmp_path / "checkpoint" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, tmp_path)

    root = tmp_path / "extra"
    root.mkdir()
    receipt = fixture_bundle(root)
    (root / "checkpoint" / "extra.bin").write_bytes(b"extra")
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, root)

    root = tmp_path / "link"
    root.mkdir()
    receipt = fixture_bundle(root)
    (root / "checkpoint" / "alias").symlink_to("model.safetensors")
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, root)


def test_plan_authorization_and_checkpoint_artifact_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    receipt = fixture_bundle(tmp_path)
    (tmp_path / "plan.json").write_text("{}\n")
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, tmp_path)

    root = tmp_path / "authorization"
    root.mkdir()
    receipt = fixture_bundle(root)
    payload = json.loads((root / "authorization.json").read_text())
    payload["authorized_run_identities"] = []
    (root / "authorization.json").write_text(json.dumps(payload) + "\n")
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, root)


def test_unsafe_artifact_paths_and_unknown_fields_fail_closed(tmp_path: Path) -> None:
    receipt = fixture_bundle(tmp_path)
    receipt["plan"]["path"] = "../plan.json"
    resign(receipt)
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, tmp_path)
    receipt = fixture_bundle(tmp_path / "second")
    receipt["unexpected"] = True
    resign(receipt)
    with pytest.raises(CompletedRunLineageError):
        validate_receipt(receipt, tmp_path / "second")
