from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.evaluation.development_mc import DISJOINT_RECEIPT_SCHEMA
from sai.evaluation.development_mc_shards import (
    DevelopmentMCShardError,
    build_shards,
    merge_shards,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _source_and_receipt(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "mmlu.jsonl"
    rows = [
        {
            "benchmark": "mmlu_pro",
            "row_id": f"row-{index}",
            "domain": "logic" if index < 3 else "math",
            "question": f"Question {index}?",
            "choices": ["yes", "no"],
            "answer_index": index % 2,
        }
        for index in range(5)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    training_sha256 = "a" * 64
    receipt = tmp_path / "disjoint.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": DISJOINT_RECEIPT_SCHEMA,
                "benchmark": "mmlu_pro",
                "benchmark_source_sha256": source_sha256,
                "training_source_sha256": training_sha256,
                "source_disjoint": True,
                "method": "identity-and-contamination-audit",
                "evidence_sha256": "b" * 64,
            }
        )
    )
    return source, receipt, training_sha256


def _shard_result(shard: dict[str, object], index: int) -> dict[str, object]:
    rows = [
        json.loads(line) for line in Path(str(shard["source"])).read_text().splitlines()
    ]
    scored_rows = [
        {
            "row_id": row["row_id"],
            "domain": row["domain"],
            "answer_index": row["answer_index"],
            "predicted_index": (
                row["answer_index"]
                if int(row["row_id"][4:]) % 2 == 0
                else 1 - row["answer_index"]
            ),
            "correct": int(row["row_id"][4:]) % 2 == 0,
            "choice_scores": [{}, {}],
        }
        for row in rows
    ]
    correct = sum(row["correct"] for row in scored_rows)
    decoding_contract = {"mode": "likelihood"}
    scoring_contract = {"mode": "normalized"}
    value: dict[str, object] = {
        "schema": "sai-development-mc-likelihood-v1",
        "status": "complete",
        "benchmark": "mmlu_pro",
        "development_only": True,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "architecture_promotion_allowed": False,
        "bindings": {
            "benchmark_source_sha256": shard["source_sha256"],
            "training_source_sha256": "a" * 64,
            "source_disjoint_receipt_sha256": shard["disjoint_receipt_sha256"],
            "identity_order_sha256": shard["identity_order_sha256"],
            "checkpoint_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "tokenizer_sha256": "e" * 64,
            "evaluator_code_sha256": "f" * 64,
            "runtime_files_sha256": "1" * 64,
            "runtime_sha256": "2" * 64,
            "decoding_contract_sha256": _canonical_sha256(decoding_contract),
            "scoring_contract_sha256": _canonical_sha256(scoring_contract),
        },
        "decoding_contract": decoding_contract,
        "scoring_contract": scoring_contract,
        "coverage": {"expected_rows": len(rows), "scored_rows": len(rows)},
        "aggregate": {
            "correct": correct,
            "rows": len(rows),
            "accuracy": correct / len(rows),
        },
        "domains": {},
        "rows": scored_rows,
        "parent_evidence": {"unchanged_parent": True},
        "workspace_evidence": {
            "state_mode": "recurrent",
            "training_run_sha256": "3" * 64,
        },
    }
    value["receipt_sha256"] = _canonical_sha256(value)
    return value


def _build_with_results(tmp_path: Path):
    source, receipt, training_sha256 = _source_and_receipt(tmp_path)
    root = tmp_path / "shards"
    manifest = build_shards(
        benchmark="mmlu_pro",
        source_path=source,
        full_disjoint_receipt_path=receipt,
        training_source_sha256=training_sha256,
        shard_count=3,
        output_root=root,
    )
    result_paths = []
    for index, shard in enumerate(manifest["shards"]):
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(_shard_result(shard, index)))
        result_paths.append(path)
    return source, receipt, root / "manifest.json", manifest, result_paths


def test_builds_exact_contiguous_shards_and_merges_standard_result(
    tmp_path: Path,
) -> None:
    source, receipt, manifest_path, manifest, result_paths = _build_with_results(
        tmp_path
    )
    assert [(row["start"], row["end"]) for row in manifest["shards"]] == [
        (0, 1),
        (1, 3),
        (3, 5),
    ]
    assert (
        manifest["full_source_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert (
        manifest["full_disjoint_receipt_sha256"]
        == hashlib.sha256(receipt.read_bytes()).hexdigest()
    )
    unsigned = dict(manifest)
    receipt_sha256 = unsigned.pop("receipt_sha256")
    assert receipt_sha256 == _canonical_sha256(unsigned)

    merged = merge_shards(manifest_path, result_paths)
    assert merged["coverage"] == {"expected_rows": 5, "scored_rows": 5}
    assert merged["aggregate"] == {"correct": 3, "rows": 5, "accuracy": 0.6}
    assert [row["row_id"] for row in merged["rows"]] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
        "row-4",
    ]
    assert (
        merged["bindings"]["benchmark_source_sha256"] == manifest["full_source_sha256"]
    )
    assert (
        merged["bindings"]["source_disjoint_receipt_sha256"]
        == manifest["full_disjoint_receipt_sha256"]
    )
    assert merged["parent_evidence"] == {"unchanged_parent": True}
    assert merged["workspace_evidence"] == {
        "state_mode": "recurrent",
        "training_run_sha256": "3" * 64,
    }
    unsigned = dict(merged)
    merged_receipt = unsigned.pop("receipt_sha256")
    assert merged_receipt == _canonical_sha256(unsigned)


def test_merge_rejects_resigned_binding_and_mutated_shard_source(
    tmp_path: Path,
) -> None:
    _, _, manifest_path, _, result_paths = _build_with_results(tmp_path)
    value = json.loads(result_paths[1].read_text())
    value["bindings"]["runtime_sha256"] = "9" * 64
    unsigned = dict(value)
    unsigned.pop("receipt_sha256")
    value["receipt_sha256"] = _canonical_sha256(unsigned)
    result_paths[1].write_text(json.dumps(value))
    with pytest.raises(DevelopmentMCShardError, match="execution binding"):
        merge_shards(manifest_path, result_paths)

    value = json.loads(result_paths[1].read_text())
    value["bindings"]["runtime_sha256"] = "2" * 64
    value["workspace_evidence"]["state_mode"] = "reset_average"
    unsigned = dict(value)
    unsigned.pop("receipt_sha256")
    value["receipt_sha256"] = _canonical_sha256(unsigned)
    result_paths[1].write_text(json.dumps(value))
    with pytest.raises(DevelopmentMCShardError, match="workspace evidence"):
        merge_shards(manifest_path, result_paths)

    source = manifest_path.parent / "shard_00.jsonl"
    source.write_text(source.read_text() + "{}\n")
    with pytest.raises(DevelopmentMCShardError, match="row differs"):
        merge_shards(manifest_path, result_paths)


def test_build_is_atomic_no_overwrite(tmp_path: Path) -> None:
    source, receipt, training_sha256 = _source_and_receipt(tmp_path)
    root = tmp_path / "shards"
    build_shards(
        benchmark="mmlu_pro",
        source_path=source,
        full_disjoint_receipt_path=receipt,
        training_source_sha256=training_sha256,
        shard_count=2,
        output_root=root,
    )
    with pytest.raises(DevelopmentMCShardError, match="output root differs"):
        build_shards(
            benchmark="mmlu_pro",
            source_path=source,
            full_disjoint_receipt_path=receipt,
            training_source_sha256=training_sha256,
            shard_count=2,
            output_root=root,
        )
