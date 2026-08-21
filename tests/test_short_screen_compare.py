from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.evaluation.short_screen_compare import (
    BENCHMARKS,
    FAMILIES,
    ShortScreenComparisonError,
    compare,
    write_comparison,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _result(benchmark: str, family: str, correctness: list[bool]) -> dict[str, object]:
    rows = [
        {
            "row_id": f"row-{index}",
            "domain": "logic",
            "answer_index": 0,
            "predicted_index": 0 if correct else 1,
            "correct": correct,
            "choice_scores": [],
        }
        for index, correct in enumerate(correctness)
    ]
    correct = sum(correctness)
    shared = {
        "benchmark_source_sha256": "1" * 64,
        "training_source_sha256": "2" * 64,
        "source_disjoint_receipt_sha256": "3" * 64,
        "identity_order_sha256": "4" * 64,
        "tokenizer_sha256": "5" * 64,
        "evaluator_code_sha256": "6" * 64,
        "runtime_files_sha256": "7" * 64,
        "runtime_sha256": "8" * 64,
        "decoding_contract_sha256": "9" * 64,
        "scoring_contract_sha256": "a" * 64,
        "checkpoint_sha256": hashlib.sha256(family.encode()).hexdigest(),
    }
    value: dict[str, object] = {
        "schema": "sai-development-mc-likelihood-v1",
        "status": "complete",
        "benchmark": benchmark,
        "development_only": True,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "architecture_promotion_allowed": False,
        "bindings": shared,
        "coverage": {"expected_rows": len(rows), "scored_rows": len(rows)},
        "aggregate": {
            "correct": correct,
            "rows": len(rows),
            "accuracy": correct / len(rows),
        },
        "rows": rows,
    }
    value["receipt_sha256"] = _canonical_sha256(value)
    return value


def _matrix(tmp_path: Path) -> dict[str, dict[str, Path]]:
    scores = {
        "gated_gqa": [True, True, False, False],
        "gdn_hybrid": [True, False, True, False],
        "kda_mla_hybrid": [False, False, True, False],
    }
    paths = {}
    for family in FAMILIES:
        paths[family] = {}
        for benchmark in BENCHMARKS:
            path = tmp_path / f"{family}.{benchmark}.json"
            path.write_text(json.dumps(_result(benchmark, family, scores[family])))
            paths[family][benchmark] = path
    return paths


def test_compares_exact_family_matrix_and_writes_once(tmp_path: Path) -> None:
    paths = _matrix(tmp_path)
    unicode_path = paths["gated_gqa"]["musr"]
    unicode_result = json.loads(unicode_path.read_text())
    unicode_result["rows"][0]["domain"] = "lógica"
    unsigned = dict(unicode_result)
    unsigned.pop("receipt_sha256")
    unicode_result["receipt_sha256"] = _canonical_sha256(unsigned)
    unicode_path.write_text(json.dumps(unicode_result, ensure_ascii=False))
    for family in FAMILIES[1:]:
        path = paths[family]["musr"]
        result = json.loads(path.read_text())
        result["rows"][0]["domain"] = "lógica"
        unsigned = dict(result)
        unsigned.pop("receipt_sha256")
        result["receipt_sha256"] = _canonical_sha256(unsigned)
        path.write_text(json.dumps(result, ensure_ascii=False))
    payload = compare(paths)
    musr = payload["benchmarks"]["musr"]
    assert musr["families"]["gated_gqa"]["accuracy"] == 0.5
    pair = musr["pairwise"]["gated_gqa_minus_kda_mla_hybrid"]
    assert pair["left_only_correct"] == 2
    assert pair["right_only_correct"] == 1
    assert pair["same_outcome"] == 1
    assert pair["paired_interval"]["delta_percentage_points"] == 25.0
    assert payload["iso_flop_comparison"] is False
    unsigned = dict(payload)
    receipt = unsigned.pop("receipt_sha256")
    assert receipt == _canonical_sha256(unsigned)
    output = tmp_path / "comparison.json"
    write_comparison(output, payload)
    assert json.loads(output.read_text()) == payload
    with pytest.raises(ShortScreenComparisonError, match="output path differs"):
        write_comparison(output, payload)


@pytest.mark.parametrize("tamper", ["receipt", "binding", "row", "checkpoint"])
def test_rejects_tampered_or_unpaired_results(tmp_path: Path, tamper: str) -> None:
    paths = _matrix(tmp_path)
    path = paths["gdn_hybrid"]["musr"]
    value = json.loads(path.read_text())
    if tamper == "receipt":
        value["receipt_sha256"] = "0" * 64
    elif tamper == "binding":
        value["bindings"]["scoring_contract_sha256"] = "b" * 64
        unsigned = dict(value)
        unsigned.pop("receipt_sha256")
        value["receipt_sha256"] = _canonical_sha256(unsigned)
    elif tamper == "row":
        value["rows"][0]["row_id"] = "wrong"
        unsigned = dict(value)
        unsigned.pop("receipt_sha256")
        value["receipt_sha256"] = _canonical_sha256(unsigned)
    else:
        value["bindings"]["checkpoint_sha256"] = hashlib.sha256(
            b"gated_gqa"
        ).hexdigest()
        unsigned = dict(value)
        unsigned.pop("receipt_sha256")
        value["receipt_sha256"] = _canonical_sha256(unsigned)
    path.write_text(json.dumps(value))
    with pytest.raises(ShortScreenComparisonError):
        compare(paths)
