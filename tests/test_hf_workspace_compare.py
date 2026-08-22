from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.evaluation.hf_smol_workspace_compare import compare_cross_family
from sai.evaluation.hf_workspace_compare import (
    HFWorkspaceComparisonError,
    compare,
    write_comparison,
)
from sai.training.hf_smol_workspace_screen import SCHEMA as SMOL_TRAINING_SCHEMA


def _workspace(
    mode: str, training_result: dict[str, object], training_path: Path
) -> dict[str, object]:
    common: dict[str, object] = {
        "parent_snapshot_tree_sha256": "1" * 64,
        "workspace_config_sha256": "2" * 64,
        "workspace_parameter_count": 19_938_304,
        "workspace_initial_state_sha256": "3" * 64,
        "training_stream_identity_sha256": "4" * 64,
        "training_source_manifest_sha256": "5" * 64,
        "training_sequences": 61_035,
        "training_utf8_bytes": 987_654,
        "optimizer": {"learning_rate": 3e-4},
        "code_sha256": "6" * 64,
        "environment_sha256": "7" * 64,
        "state_mode": mode,
        "matched_comparison": True,
        "source_disjoint_from_factor_training": True,
        "four_b_training_executed": False,
        "training_result_file_sha256": hashlib.sha256(
            training_path.read_bytes()
        ).hexdigest(),
        "training_receipt_sha256": training_result["receipt_sha256"],
    }
    common.update(
        {
            "training_run_sha256": ("8" if mode == "recurrent" else "a") * 64,
            "workspace_final_state_sha256": ("9" if mode == "recurrent" else "b") * 64,
        }
    )
    return common


def _write_training(path: Path, mode: str) -> dict[str, object]:
    evidence = {
        "schema": "sai-qwen35-0p8b-matched-workspace-screen-v1",
        "status": "complete",
        "state_mode": mode,
        "training_sequences": 61_035,
        "parent_state_unchanged": True,
        "four_b_training_executed": False,
        "architecture_improvement_demonstrated": False,
        "run_sha256": ("8" if mode == "recurrent" else "a") * 64,
        "workspace_final_state_sha256": ("9" if mode == "recurrent" else "b") * 64,
        "parent_snapshot_tree_sha256": "1" * 64,
        "workspace_config_sha256": "2" * 64,
        "workspace_parameter_count": 19_938_304,
        "workspace_initial_state_sha256": "3" * 64,
        "training_stream_identity_sha256": "4" * 64,
        "training_source_manifest_sha256": "5" * 64,
        "training_utf8_bytes": 987_654,
        "optimizer": {"learning_rate": 3e-4},
        "code_sha256": "6" * 64,
        "environment_sha256": "7" * 64,
    }
    evidence["receipt_sha256"] = canonical_sha256(evidence)
    path.write_text(json.dumps(evidence))
    return evidence


def _write_result(
    path: Path,
    *,
    benchmark: str,
    mode: str,
    correct: list[bool],
    training_result: dict[str, object] | None = None,
    training_path: Path | None = None,
) -> None:
    rows = [
        {
            "row_id": f"{benchmark}-{index:04d}",
            "domain": "reasoning",
            "answer_index": 0,
            "predicted_index": 0 if value else 1,
            "correct": value,
            "choice_scores": [{}, {}],
        }
        for index, value in enumerate(correct)
    ]
    decoding = {"mode": "likelihood"}
    scoring = {"mode": "normalized"}
    payload: dict[str, object] = {
        "schema": "sai-development-mc-likelihood-v1",
        "status": "complete",
        "benchmark": benchmark,
        "development_only": True,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "architecture_promotion_allowed": False,
        "bindings": {
            "benchmark_source_sha256": "c" * 64,
            "training_source_sha256": "d" * 64,
            "source_disjoint_receipt_sha256": "e" * 64,
            "identity_order_sha256": canonical_sha256([row["row_id"] for row in rows]),
            "checkpoint_sha256": ("f" if mode == "parent" else mode[0]) * 64,
            "config_sha256": "0" * 64,
            "tokenizer_sha256": "1" * 64,
            "evaluator_code_sha256": "2" * 64,
            "runtime_files_sha256": "3" * 64,
            "runtime_sha256": "4" * 64,
            "decoding_contract_sha256": canonical_sha256(decoding),
            "scoring_contract_sha256": canonical_sha256(scoring),
        },
        "decoding_contract": decoding,
        "scoring_contract": scoring,
        "coverage": {"expected_rows": len(rows), "scored_rows": len(rows)},
        "aggregate": {
            "correct": sum(correct),
            "rows": len(rows),
            "accuracy": sum(correct) / len(rows),
        },
        "domains": {},
        "rows": rows,
    }
    if mode == "parent":
        payload["parent_evidence"] = {"unchanged_parent": True}
    else:
        assert training_result is not None and training_path is not None
        payload["workspace_evidence"] = _workspace(mode, training_result, training_path)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))


def _inputs(tmp_path: Path, *, recurrent_gain: bool = True):
    paths = {
        mode: {
            benchmark: tmp_path / f"{mode}-{benchmark}.json"
            for benchmark in ("mmlu_pro", "musr")
        }
        for mode in ("parent", "recurrent", "reset_average")
    }
    training_paths = {
        mode: tmp_path / f"{mode}-training.json"
        for mode in ("recurrent", "reset_average")
    }
    training = {
        mode: _write_training(path, mode) for mode, path in training_paths.items()
    }
    parent = [index % 5 in {0, 1} for index in range(100)]
    reset = [index % 5 in {0, 1} for index in range(100)]
    recurrent = (
        [index % 5 in {0, 1, 2} for index in range(100)]
        if recurrent_gain
        else [index % 5 == 0 for index in range(100)]
    )
    for benchmark in ("mmlu_pro", "musr"):
        _write_result(
            paths["parent"][benchmark],
            benchmark=benchmark,
            mode="parent",
            correct=parent,
        )
        _write_result(
            paths["recurrent"][benchmark],
            benchmark=benchmark,
            mode="recurrent",
            correct=recurrent,
            training_result=training["recurrent"],
            training_path=training_paths["recurrent"],
        )
        _write_result(
            paths["reset_average"][benchmark],
            benchmark=benchmark,
            mode="reset_average",
            correct=reset,
            training_result=training["reset_average"],
            training_path=training_paths["reset_average"],
        )
    paths["training"] = training_paths
    return paths


def test_comparison_passes_only_clear_paired_recurrent_gain(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = compare(
        parent_paths=paths["parent"],
        recurrent_paths=paths["recurrent"],
        reset_paths=paths["reset_average"],
        recurrent_training_result=paths["training"]["recurrent"],
        reset_training_result=paths["training"]["reset_average"],
    )
    assert result["pass"] is True
    assert result["action"] == "authorize_sub4b_confirmation"
    assert result["macro"]["recurrent_minus_parent"] == pytest.approx(0.2)
    assert result["macro"]["recurrent_minus_reset_average"] == pytest.approx(0.2)
    assert all(result["checks"].values())
    assert result["four_b_training_authorized"] is False
    unsigned = dict(result)
    claimed = unsigned.pop("receipt_sha256")
    assert claimed == canonical_sha256(unsigned)
    output = tmp_path / "comparison.json"
    write_comparison(output, result)
    with pytest.raises(HFWorkspaceComparisonError, match="output target"):
        write_comparison(output, result)


def test_comparison_rejects_regression_and_matched_input_tamper(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, recurrent_gain=False)
    result = compare(
        parent_paths=paths["parent"],
        recurrent_paths=paths["recurrent"],
        reset_paths=paths["reset_average"],
        recurrent_training_result=paths["training"]["recurrent"],
        reset_training_result=paths["training"]["reset_average"],
    )
    assert result["pass"] is False
    assert result["action"] == "reject_recurrent_workspace"

    training_path = paths["training"]["reset_average"]
    training = json.loads(training_path.read_text())
    training["workspace_initial_state_sha256"] = "f" * 64
    training.pop("receipt_sha256")
    training["receipt_sha256"] = canonical_sha256(training)
    training_path.write_text(json.dumps(training))
    training_file_sha = hashlib.sha256(training_path.read_bytes()).hexdigest()
    for benchmark in ("mmlu_pro", "musr"):
        value = json.loads(paths["reset_average"][benchmark].read_text())
        value["workspace_evidence"]["training_result_file_sha256"] = training_file_sha
        value["workspace_evidence"]["training_receipt_sha256"] = training[
            "receipt_sha256"
        ]
        value.pop("receipt_sha256")
        value["receipt_sha256"] = canonical_sha256(value)
        paths["reset_average"][benchmark].write_text(json.dumps(value))
    with pytest.raises(HFWorkspaceComparisonError, match="matched workspace"):
        compare(
            parent_paths=paths["parent"],
            recurrent_paths=paths["recurrent"],
            reset_paths=paths["reset_average"],
            recurrent_training_result=paths["training"]["recurrent"],
            reset_training_result=paths["training"]["reset_average"],
        )


def test_comparison_rejects_resigned_row_identity_drift(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    value = json.loads(paths["recurrent"]["musr"].read_text())
    value["rows"][0]["row_id"] = "different-row"
    value.pop("receipt_sha256")
    value["receipt_sha256"] = canonical_sha256(value)
    paths["recurrent"]["musr"].write_text(json.dumps(value))
    with pytest.raises(HFWorkspaceComparisonError, match="paired row identity"):
        compare(
            parent_paths=paths["parent"],
            recurrent_paths=paths["recurrent"],
            reset_paths=paths["reset_average"],
            recurrent_training_result=paths["training"]["recurrent"],
            reset_training_result=paths["training"]["reset_average"],
        )


def test_comparison_reuses_exact_gate_with_cross_family_schema(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    for mode in ("recurrent", "reset_average"):
        training_path = paths["training"][mode]
        training = json.loads(training_path.read_text())
        training["schema"] = SMOL_TRAINING_SCHEMA
        training.pop("receipt_sha256")
        training["receipt_sha256"] = canonical_sha256(training)
        training_path.write_text(json.dumps(training))
        training_file_sha = hashlib.sha256(training_path.read_bytes()).hexdigest()
        for benchmark in ("mmlu_pro", "musr"):
            value = json.loads(paths[mode][benchmark].read_text())
            value["workspace_evidence"][
                "training_result_file_sha256"
            ] = training_file_sha
            value["workspace_evidence"]["training_receipt_sha256"] = training[
                "receipt_sha256"
            ]
            value["workspace_evidence"]["cross_family_confirmation"] = True
            value.pop("receipt_sha256")
            value["receipt_sha256"] = canonical_sha256(value)
            paths[mode][benchmark].write_text(json.dumps(value))
    qwen_factor = {
        "schema": "sai-qwen35-0p8b-matched-workspace-comparison-v1",
        "status": "complete",
        "checks": {"macro_gain": True, "paired_lcb": True},
        "pass": True,
        "action": "authorize_sub4b_confirmation",
        "architecture_locked": False,
        "four_b_training_executed": False,
        "four_b_training_authorized": False,
    }
    qwen_factor["receipt_sha256"] = canonical_sha256(qwen_factor)
    qwen_factor_path = tmp_path / "qwen-factor.json"
    qwen_factor_path.write_text(json.dumps(qwen_factor))
    result = compare_cross_family(
        qwen_factor_receipt=qwen_factor_path,
        parent_paths=paths["parent"],
        recurrent_paths=paths["recurrent"],
        reset_paths=paths["reset_average"],
        recurrent_training_result=paths["training"]["recurrent"],
        reset_training_result=paths["training"]["reset_average"],
    )
    assert result["schema"] == "sai-smollm3-3b-matched-workspace-comparison-v1"
    assert result["pass"] is True
    assert result["action"] == (
        "cross_family_factor_confirmed_await_user_4b_authorization"
    )
    assert result["four_b_training_authorized"] is False
    assert result["qwen_factor_evidence"] == {
        "path": str(qwen_factor_path.resolve()),
        "file_sha256": hashlib.sha256(qwen_factor_path.read_bytes()).hexdigest(),
        "receipt_sha256": qwen_factor["receipt_sha256"],
        "schema": qwen_factor["schema"],
        "pass": True,
        "action": "authorize_sub4b_confirmation",
    }

    qwen_factor["pass"] = False
    qwen_factor.pop("receipt_sha256")
    qwen_factor["receipt_sha256"] = canonical_sha256(qwen_factor)
    qwen_factor_path.write_text(json.dumps(qwen_factor))
    with pytest.raises(HFWorkspaceComparisonError, match="did not pass"):
        compare_cross_family(
            qwen_factor_receipt=qwen_factor_path,
            parent_paths=paths["parent"],
            recurrent_paths=paths["recurrent"],
            reset_paths=paths["reset_average"],
            recurrent_training_result=paths["training"]["recurrent"],
            reset_training_result=paths["training"]["reset_average"],
        )
