from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sai.evaluation.curriculum_benchmark_compare as paired_module
from sai.data.token_stream import canonical_sha256
from sai.evaluation.source_addition_benchmark_compare import (
    SourceAdditionBenchmarkComparisonError,
    compare_source_addition_benchmarks,
)


@pytest.fixture(autouse=True)
def _bounded_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paired_module, "EXPECTED_ROWS", {"mmlu_pro": 100, "musr": 100})


def _write(path: Path, payload: dict) -> Path:
    payload = copy.deepcopy(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _nll(tmp_path: Path, *, passed: bool = True) -> Path:
    checkpoint_fields = {
        "checkpoint_file_sha256": "1" * 64,
        "checkpoint_manifest_file_sha256": "2" * 64,
        "checkpoint_bundle_sha256": "3" * 64,
    }
    control_checkpoint = dict(checkpoint_fields)
    control_checkpoint["checkpoint_bundle_sha256"] = "4" * 64
    return _write(
        tmp_path / "nll.json",
        {
            "schema": "sai-source-addition-nll-comparison-v1",
            "status": "complete",
            "source_addition_supported_by_heldout_nll": passed,
            "real_source_disjoint_benchmark_confirmation_required": True,
            "source_addition_retained": False,
            "data_promotion_authorized": False,
            "architecture_promotion_authorized": False,
            "four_b_training_authorized": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "inputs": {
                "treatment_checkpoint": checkpoint_fields,
                "control_checkpoint": control_checkpoint,
                "treatment_training_source_sha256": "5" * 64,
                "control_training_source_sha256": "6" * 64,
            },
        },
    )


def _result(tmp_path: Path, benchmark: str, arm: str, correctness: list[bool]) -> Path:
    checkpoint = "3" * 64 if arm == "treatment" else "4" * 64
    rows = [
        {
            "row_id": f"{benchmark}-{index}",
            "domain": "a" if index < len(correctness) // 2 else "b",
            "answer_index": 0,
            "predicted_index": 0 if correct else 1,
            "correct": correct,
            "choice_scores": [{"normalized_log_likelihood": -1.0}],
        }
        for index, correct in enumerate(correctness)
    ]
    correct = sum(correctness)
    bindings = {
        field: "a" * 64
        for field in (
            "benchmark_source_sha256",
            "training_source_sha256",
            "source_disjoint_receipt_sha256",
            "identity_order_sha256",
            "config_sha256",
            "tokenizer_sha256",
            "evaluator_code_sha256",
            "runtime_files_sha256",
            "runtime_sha256",
            "decoding_contract_sha256",
            "scoring_contract_sha256",
        )
    }
    bindings["checkpoint_sha256"] = checkpoint
    bindings["training_source_sha256"] = "5" * 64 if arm == "treatment" else "6" * 64
    bindings["source_disjoint_receipt_sha256"] = (
        "7" * 64 if arm == "treatment" else "8" * 64
    )
    return _write(
        tmp_path / f"{arm}.{benchmark}.json",
        {
            "schema": "sai-development-mc-likelihood-v1",
            "status": "complete",
            "benchmark": benchmark,
            "development_only": True,
            "official_benchmark_result": False,
            "public_terminal_result": False,
            "architecture_promotion_allowed": False,
            "bindings": bindings,
            "coverage": {"expected_rows": len(rows), "scored_rows": len(rows)},
            "aggregate": {
                "correct": correct,
                "rows": len(rows),
                "accuracy": correct / len(rows),
            },
            "rows": rows,
        },
    )


def _inputs(tmp_path: Path, *, regression: bool = False) -> dict:
    control = [True] * 40 + [False] * 60
    treatment = [True] * 70 + [False] * 30
    if regression:
        treatment = [True] * 30 + [False] * 70
    return {
        "nll": _nll(tmp_path),
        "treatment": {
            benchmark: _result(tmp_path, benchmark, "treatment", treatment)
            for benchmark in ("mmlu_pro", "musr")
        },
        "control": {
            benchmark: _result(tmp_path, benchmark, "control", control)
            for benchmark in ("mmlu_pro", "musr")
        },
    }


def test_source_is_retained_only_after_strong_real_benchmark_gain(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = compare_source_addition_benchmarks(
        inputs["nll"],
        treatment_results=inputs["treatment"],
        control_results=inputs["control"],
    )
    assert payload["source_addition_retained"] is True
    assert payload["bootstrap"]["macro_delta_lcb_95"] > 0
    assert payload["architecture_promotion_authorized"] is False
    assert payload["four_b_training_authorized"] is False


def test_benchmark_regression_vetoes_source_retention(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, regression=True)
    payload = compare_source_addition_benchmarks(
        inputs["nll"],
        treatment_results=inputs["treatment"],
        control_results=inputs["control"],
    )
    assert payload["source_addition_retained"] is False
    assert payload["checks"]["every_benchmark_nonnegative"] is False


def test_failed_nll_or_wrong_checkpoint_cannot_open_benchmark_gate(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["nll"] = _nll(tmp_path, passed=False)
    with pytest.raises(SourceAdditionBenchmarkComparisonError):
        compare_source_addition_benchmarks(
            inputs["nll"],
            treatment_results=inputs["treatment"],
            control_results=inputs["control"],
        )

    inputs = _inputs(tmp_path)
    path = inputs["treatment"]["musr"]
    payload = json.loads(path.read_text())
    payload["bindings"]["checkpoint_sha256"] = "9" * 64
    _write(
        path, {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(SourceAdditionBenchmarkComparisonError, match="checkpoint"):
        compare_source_addition_benchmarks(
            inputs["nll"],
            treatment_results=inputs["treatment"],
            control_results=inputs["control"],
        )
