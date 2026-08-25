import json
from pathlib import Path

import pytest

from sai.data.bridge_transfer_confirmation_stage import (
    BridgeTransferConfirmationStageError,
    load_screen_launch,
    write_stage_receipt,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCREEN_COMMIT = "a" * 40
CONFIRMATION_COMMIT = "b" * 40


def _write_screen(path: Path) -> dict:
    payload = {
        "schema": "sai-bridge-transfer-newton-launch-v1",
        "status": "complete_newton_transfer_graph_launch",
        "runtime_commit": SCREEN_COMMIT,
        "launcher_job_id": 50,
        "newton_jobs": {
            "unchanged": 51,
            "source_control": 52,
            "connections": 53,
            "aggregate": 54,
        },
        "one_h100_per_arm": True,
        "matched_token_budget": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def test_stage_binds_exact_screen_and_confirmation_runtime(tmp_path: Path) -> None:
    screen_path = tmp_path / "screen.json"
    screen = _write_screen(screen_path)
    output = tmp_path / "stage" / "receipt.json"

    receipt = write_stage_receipt(
        output=output,
        screen_launch_path=screen_path,
        screen_runtime_commit=SCREEN_COMMIT,
        confirmation_runtime_commit=CONFIRMATION_COMMIT,
        confirmation_launcher_job=91,
    )

    assert receipt["screen_launch"] == {
        "bytes": screen_path.stat().st_size,
        "sha256": sha256_file(screen_path),
        "receipt_sha256": screen["receipt_sha256"],
        "runtime_commit": SCREEN_COMMIT,
        "aggregate_job": 54,
    }
    assert receipt["dependency"] == "afterok:54"
    assert receipt["confirmation_runtime_commit"] == CONFIRMATION_COMMIT
    assert receipt["one_h100_per_confirmation_arm"] is True
    assert receipt["four_b_training_authorized"] is False
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    assert receipt["receipt_sha256"] == canonical_sha256(unsigned)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(matched_token_budget=False), "contract"),
        (
            lambda payload: payload["newton_jobs"].update(
                aggregate=payload["newton_jobs"]["connections"]
            ),
            "overlap",
        ),
        (lambda payload: payload.update(runtime_commit="c" * 40), "contract"),
    ],
)
def test_stage_rejects_tampered_screen(
    tmp_path: Path, mutation, message: str
) -> None:
    screen_path = tmp_path / "screen.json"
    payload = _write_screen(screen_path)
    mutation(payload)
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    screen_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BridgeTransferConfirmationStageError, match=message):
        load_screen_launch(screen_path, SCREEN_COMMIT)


def test_stage_rejects_bad_signature_and_existing_output(tmp_path: Path) -> None:
    screen_path = tmp_path / "screen.json"
    _write_screen(screen_path)
    payload = json.loads(screen_path.read_text(encoding="utf-8"))
    payload["receipt_sha256"] = "0" * 64
    screen_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BridgeTransferConfirmationStageError, match="signature"):
        load_screen_launch(screen_path, SCREEN_COMMIT)

    _write_screen(screen_path)
    output = tmp_path / "receipt.json"
    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(BridgeTransferConfirmationStageError, match="already exists"):
        write_stage_receipt(
            output=output,
            screen_launch_path=screen_path,
            screen_runtime_commit=SCREEN_COMMIT,
            confirmation_runtime_commit=CONFIRMATION_COMMIT,
            confirmation_launcher_job=91,
        )


def test_stage_job_has_no_gpu_and_uses_exact_afterok_handoff() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/stage_bridge_transfer_confirmation_stokes.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --gres" not in script
    assert '[[ ! -e "${stage_root}" ]]' in script
    assert '--dependency="afterok:${screen_aggregate_id}"' in script
    assert "--kill-on-invalid-dep=yes" in script
    assert script.count("sbatch --parsable") == 1
    assert "SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT}" in script
    assert "SAI_SCREEN_RUNTIME_COMMIT" in script
    assert "SAI_NEWTON_SLURM_CONF_PATH" in script
    assert "SAI_NEWTON_SLURM_CONF_SHA256" in script
    assert "SAI_STOKES_SLURM_CONF_PATH" in script
    assert "SAI_STOKES_SLURM_CONF_SHA256" in script
    assert "sai_activate_verified_slurm_config" in script
    assert "SAI_FOUNDATION_AUDIT_JOB_ID" in script
    assert "four_b_training_authorized" not in script


def test_final_release_stage_waits_for_both_clusters_without_a_gpu() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/stage_final_training_release_newton.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --gres" not in script
    assert 'SAI_STOKES_SLURM_CONF_PATH:?' in script
    assert 'SAI_STOKES_SLURM_CONF_SHA256:?' in script
    assert 'SAI_FOUNDATION_AUDIT_JOB_ID:?' in script
    assert 'complete_bridge_training_component_hf_publication' in script
    assert 'development_rows_uploaded' in script
    assert '--dependency="afterok:${SAI_FOUNDATION_AUDIT_JOB_ID}"' in script
    assert script.count("sbatch --parsable") == 1
    assert "build_final_training_release_stokes.sbatch" in script


def test_confirmation_launcher_stages_final_release_after_publication() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/launch_bridge_transfer_confirmation_newton.sh"
    ).read_text(encoding="utf-8")
    assert "stage_final_training_release_newton.sbatch" in script
    assert '--dependency="afterok:${publication_id}"' in script
    assert 'SAI_FOUNDATION_AUDIT_JOB_ID=${SAI_FOUNDATION_AUDIT_JOB_ID}' in script
    assert 'SAI_STOKES_SLURM_CONF_PATH=${SAI_STOKES_SLURM_CONF_PATH}' in script
    assert 'SAI_STOKES_SLURM_CONF_SHA256=${SAI_STOKES_SLURM_CONF_SHA256}' in script
    assert '"final_release_stage_job": int(final_stage_id)' in script


def test_initial_screen_launcher_forces_verified_newton_configuration() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/launch_bridge_transfer_screen_newton_stokes.sbatch"
    ).read_text(encoding="utf-8")
    assert 'SAI_NEWTON_SLURM_CONF_PATH:?' in script
    assert 'SAI_NEWTON_SLURM_CONF_SHA256:?' in script
    assert "sai_activate_verified_slurm_config" in script
    assert "newton" in script
    assert "SAI_NEWTON_SLURM_CONF_SERVER" not in script


def test_verified_slurm_config_rejects_unbound_or_wrong_cluster_files() -> None:
    script = (
        Path(__file__).parents[1] / "scripts/verified_slurm_config.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -f "${config_path}" && ! -L "${config_path}" ]]' in script
    assert "sha256sum" in script
    assert "unset SLURM_CONF_SERVER" in script
    assert 'actual_cluster' in script
    assert '[[ "${actual_cluster}" == "${expected_cluster}" ]]' in script
