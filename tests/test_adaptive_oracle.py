from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sai.adaptive.oracle import (
    GATE_SLOTS,
    MANIFEST_SCHEMA,
    OracleEvaluationError,
    analyze,
    canonical_sha256,
    validate_analysis,
    validate_manifest,
    write_analysis,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def build_manifest(slot: str, mode: str) -> dict:
    identities = [digest(f"{slot}-row-{index}") for index in range(2)]
    scores = {
        "forced_fast": (0.5, 0.5),
        "forced_slow": (0.8, 0.4),
        "equal_flop_fast_control": (0.6, 0.5),
    }[mode]
    is_slow = mode == "forced_slow"
    is_control = mode == "equal_flop_fast_control"
    rows = []
    for index, identity in enumerate(identities):
        rows.append(
            {
                "row_identity_sha256": identity,
                "prompt_sha256": digest(f"{slot}-prompt-{index}"),
                "output_sha256": digest(f"{slot}-{mode}-output-{index}"),
                "official_score": scores[index],
                "score_weight": 1.0,
                "modeled_inference_flops": 200 if is_slow or is_control else 100,
                "executed_inference_flops": 240 if is_slow or is_control else 120,
                "output_tokens": 16 + index,
                "infrastructure_status": "complete",
                "workspace_diagnostics": (
                    {
                        "iterations": 4,
                        "workspace_plan_sha256": digest("workspace-plan"),
                        "workspace_candidate_identity_sha256": digest(
                            "workspace-candidate"
                        ),
                        "last_update_rms": 0.25,
                        "output_delta_rms": 0.5,
                    }
                    if is_slow
                    else None
                ),
            }
        )
    adaptive_checkpoint = digest("adaptive-checkpoint")
    adaptive_config = digest("adaptive-config")
    adaptive_run = digest("adaptive-completed-run")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "mode": mode,
        "gate_slot": slot,
        "benchmark_name": f"sai-{slot}-development",
        "benchmark_version": "frozen-v1",
        "benchmark_source_sha256": digest(f"{slot}-source"),
        "identity_order_sha256": canonical_sha256(identities),
        "prompt_contract_sha256": digest(f"{slot}-prompt-contract"),
        "decoding_contract_sha256": digest("decoding-contract"),
        "official_scorer_sha256": digest(f"{slot}-official-scorer"),
        "environment_sha256": digest("environment"),
        "system_checkpoint_sha256": (
            digest("equal-flop-control-checkpoint")
            if is_control
            else adaptive_checkpoint
        ),
        "fast_path_checkpoint_sha256": (
            digest("equal-flop-control-fast-path")
            if is_control
            else adaptive_checkpoint
        ),
        "system_config_sha256": (
            digest("equal-flop-control-config") if is_control else adaptive_config
        ),
        "completed_run_receipt_sha256": (
            digest("control-completed-run") if is_control else adaptive_run
        ),
        "source_disjoint": True,
        "terminal_public_board_accessed": False,
        "training_authorized": False,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def write_manifests(tmp_path: Path) -> tuple[list[Path], list[Path], list[Path]]:
    paths: dict[str, list[Path]] = {
        "forced_fast": [],
        "forced_slow": [],
        "equal_flop_fast_control": [],
    }
    for mode in paths:
        for slot in GATE_SLOTS:
            path = tmp_path / f"{mode}-{slot}.json"
            path.write_text(
                json.dumps(build_manifest(slot, mode), sort_keys=True) + "\n"
            )
            paths[mode].append(path)
    return (
        paths["forced_fast"],
        paths["forced_slow"],
        paths["equal_flop_fast_control"],
    )


def resign(payload: dict) -> None:
    payload["rows_sha256"] = canonical_sha256(payload["rows"])
    payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def test_oracle_analyzer_proves_positive_conditional_value(tmp_path: Path) -> None:
    fast, slow, control = write_manifests(tmp_path)
    result = analyze(fast, slow, control, bootstrap_replicates=500)
    assert result["decision"] == "oracle_slow_path_supported"
    assert result["next_falsification_gate_authorized"]
    assert not result["architecture_locked"]
    assert not result["training_authorized"]
    assert not result["terminal_public_board_accessed"]
    assert all(result["checks"].values())
    assert result["macro"] == pytest.approx(
        {
            "forced_fast_score": 50.0,
            "forced_slow_score": 60.0,
            "forced_equal_flop_control_score": 55.0,
            "oracle_score": 65.0,
            "mask_matched_control_score": 55.0,
            "oracle_vs_fast_points": 15.0,
            "oracle_vs_mask_matched_control_points": 10.0,
        }
    )
    assert all(row["slow_route_rate"] == 0.5 for row in result["benchmarks"].values())


def test_oracle_ties_stay_on_fast_path_and_do_not_fake_gain(tmp_path: Path) -> None:
    fast, slow, control = write_manifests(tmp_path)
    for path in slow:
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            row["official_score"] = 0.5
        resign(payload)
        path.write_text(json.dumps(payload) + "\n")
    result = analyze(fast, slow, control, bootstrap_replicates=200)
    assert result["decision"] == "oracle_slow_path_rejected"
    assert not result["next_falsification_gate_authorized"]
    assert all(row["slow_route_rate"] == 0.0 for row in result["benchmarks"].values())


def test_write_analysis_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    fast, slow, control = write_manifests(tmp_path)
    output = tmp_path / "oracle.json"
    payload = write_analysis(fast, slow, control, output, bootstrap_replicates=200)
    assert json.loads(output.read_text()) == payload
    assert (
        validate_analysis(payload, fast, slow, control, bootstrap_replicates=200)
        == payload
    )
    with pytest.raises(OracleEvaluationError, match="already exists"):
        write_analysis(fast, slow, control, output, bootstrap_replicates=200)


def test_tampered_analysis_cannot_replay(tmp_path: Path) -> None:
    fast, slow, control = write_manifests(tmp_path)
    payload = analyze(fast, slow, control, bootstrap_replicates=200)
    payload["macro"]["oracle_score"] -= 1
    with pytest.raises(OracleEvaluationError, match="analysis identity"):
        validate_analysis(payload, fast, slow, control, bootstrap_replicates=200)


def test_manifest_hash_and_row_hash_tampering_fail_closed() -> None:
    payload = build_manifest(GATE_SLOTS[0], "forced_fast")
    payload["rows"][0]["official_score"] = 1.0
    with pytest.raises(OracleEvaluationError, match="manifest identity"):
        validate_manifest(payload, "forced_fast")
    payload = build_manifest(GATE_SLOTS[0], "forced_fast")
    payload["rows"][0]["official_score"] = 1.0
    payload["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    with pytest.raises(OracleEvaluationError, match="row receipt"):
        validate_manifest(payload, "forced_fast")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("flops", "inference FLOPs differ"),
        ("prompt", "row pairing differs"),
        ("scorer", "benchmark binding differs"),
        ("infrastructure", "infrastructure is incomplete"),
        ("duplicate", "identities are duplicated"),
    ],
)
def test_pairing_flop_scorer_and_infrastructure_fail_closed(
    tmp_path: Path, mutation: str, match: str
) -> None:
    fast, slow, control = write_manifests(tmp_path)
    target = slow[0]
    payload = json.loads(target.read_text())
    if mutation == "flops":
        payload["rows"][0]["modeled_inference_flops"] += 1
    elif mutation == "prompt":
        payload["rows"][0]["prompt_sha256"] = digest("wrong-prompt")
    elif mutation == "scorer":
        payload["official_scorer_sha256"] = digest("wrong-scorer")
    elif mutation == "infrastructure":
        payload["rows"][0]["infrastructure_status"] = "failed"
    else:
        payload["rows"][1]["row_identity_sha256"] = payload["rows"][0][
            "row_identity_sha256"
        ]
        payload["identity_order_sha256"] = canonical_sha256(
            [row["row_identity_sha256"] for row in payload["rows"]]
        )
    resign(payload)
    target.write_text(json.dumps(payload) + "\n")
    with pytest.raises(OracleEvaluationError, match=match):
        analyze(fast, slow, control, bootstrap_replicates=200)


def test_missing_or_duplicate_gate_slot_fails_closed(tmp_path: Path) -> None:
    fast, slow, control = write_manifests(tmp_path)
    with pytest.raises(OracleEvaluationError, match="exactly five"):
        analyze(fast[:-1], slow, control, bootstrap_replicates=200)
    duplicate = copy.copy(fast)
    duplicate[-1] = duplicate[0]
    with pytest.raises(OracleEvaluationError, match="duplicate forced_fast"):
        analyze(duplicate, slow, control, bootstrap_replicates=200)
