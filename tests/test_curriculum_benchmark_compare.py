import copy
import json
from pathlib import Path

import pytest

import sai.evaluation.curriculum_benchmark_compare as comparison_module
from sai.data.token_stream import canonical_sha256
from sai.evaluation.curriculum_benchmark_compare import (
    CurriculumBenchmarkComparisonError,
    compare_curriculum_benchmarks,
)


@pytest.fixture(autouse=True)
def _bounded_synthetic_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        comparison_module, "EXPECTED_ROWS", {"mmlu_pro": 100, "musr": 100}
    )


def _write(path: Path, payload: dict) -> Path:
    payload = copy.deepcopy(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _order(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "order.json",
        {
            "schema": "sai-curriculum-order-training-comparison-v1",
            "status": "complete",
            "curriculum_order_supported_by_heldout_nll": True,
            "heldout_phase_no_regression": True,
            "same_documents_tokens_targets_masks": True,
            "only_training_sequence_order_changed": True,
            "same_model_initialization_optimizer_budget_compute": True,
            "development_population_disjoint_from_training": True,
            "real_benchmark_gate_required": True,
            "scientific_promotion_authorized": False,
            "four_b_training_authorized": False,
            "arms": {
                "curriculum": {"checkpoint_sha256": "1" * 64},
                "order_control": {"checkpoint_sha256": "2" * 64},
            },
        },
    )


def _result(tmp_path: Path, benchmark: str, arm: str, correctness: list[bool]) -> Path:
    checkpoint = "1" * 64 if arm == "curriculum" else "2" * 64
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


def _inputs(tmp_path: Path, *, regression: bool = False):
    control = [True] * 40 + [False] * 60
    curriculum = [True] * 70 + [False] * 30
    if regression:
        curriculum = [True] * 30 + [False] * 70
    return {
        "order": _order(tmp_path),
        "curriculum": {
            benchmark: _result(tmp_path, benchmark, "curriculum", curriculum)
            for benchmark in ("mmlu_pro", "musr")
        },
        "control": {
            benchmark: _result(tmp_path, benchmark, "control", control)
            for benchmark in ("mmlu_pro", "musr")
        },
    }


def test_real_benchmark_confirmation_passes_strong_paired_gain(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = compare_curriculum_benchmarks(
        inputs["order"],
        curriculum_results=inputs["curriculum"],
        control_results=inputs["control"],
    )
    assert payload["curriculum_order_supported_by_real_development_benchmarks"]
    assert payload["data_order_retention_authorized"]
    assert payload["bootstrap"]["macro_delta_lcb_95"] > 0
    assert payload["architecture_promotion_authorized"] is False
    assert payload["four_b_training_authorized"] is False


def test_benchmark_regression_vetoes_retention(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, regression=True)
    payload = compare_curriculum_benchmarks(
        inputs["order"],
        curriculum_results=inputs["curriculum"],
        control_results=inputs["control"],
    )
    assert payload["data_order_retention_authorized"] is False
    assert payload["checks"]["every_benchmark_nonnegative"] is False


def test_domain_regression_vetoes_favorable_aggregate(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    curriculum = [True] * 30 + [False] * 20 + [True] * 50
    inputs["curriculum"] = {
        benchmark: _result(tmp_path, benchmark, "curriculum", curriculum)
        for benchmark in ("mmlu_pro", "musr")
    }
    payload = compare_curriculum_benchmarks(
        inputs["order"],
        curriculum_results=inputs["curriculum"],
        control_results=inputs["control"],
    )
    assert payload["unweighted_macro_accuracy_delta"] > 0
    assert payload["checks"]["every_domain_delta_at_least_minus_1pp"] is False
    assert payload["data_order_retention_authorized"] is False


def test_checkpoint_or_row_pairing_tamper_fails(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    path = inputs["curriculum"]["musr"]
    payload = json.loads(path.read_text())
    payload["rows"][0]["row_id"] = "different"
    _write(
        path, {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(CurriculumBenchmarkComparisonError):
        compare_curriculum_benchmarks(
            inputs["order"],
            curriculum_results=inputs["curriculum"],
            control_results=inputs["control"],
        )
