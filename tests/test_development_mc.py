from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sai.evaluation.development_mc import (
    DISJOINT_RECEIPT_SCHEMA,
    DevelopmentMCError,
    evaluate_development_mc,
    write_development_mc,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


class YesBiasedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        assert torch.equal(segment_ids, torch.zeros_like(input_ids))
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        preferred = {
            ord(":"): ord(" "),
            ord(" "): ord("y"),
            ord("y"): ord("e"),
            ord("e"): ord("s"),
        }
        for source, target in preferred.items():
            logits[:, :, target] += (input_ids == source).float() * 4.0
        return logits + self.anchor * 0.0


class SpecializedYesBiasedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[int, int]] = []

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor):
        raise AssertionError("the full-logit path must not run")

    def choice_logits(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        start_position: int,
        token_count: int,
    ) -> torch.Tensor:
        assert torch.equal(segment_ids, torch.zeros_like(input_ids))
        self.calls.append((start_position, token_count))
        logits = torch.zeros((token_count, 128), device=input_ids.device)
        targets = [ord(" "), ord("y"), ord("e"), ord("s")]
        for index, target in enumerate(targets[:token_count]):
            logits[index, target] = 4.0
        return logits + self.anchor * 0.0


def _artifact(tmp_path: Path, name: str, text: str = "artifact") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _population(
    tmp_path: Path, benchmark: str = "mmlu_pro"
) -> tuple[Path, list[dict[str, object]]]:
    common: dict[str, object] = {
        "benchmark": benchmark,
        "row_id": "row-001",
        "domain": "logic",
        "question": "Is the statement valid?",
        "choices": ["yes", "no"],
        "answer_index": 0,
    }
    if benchmark == "musr":
        common["context"] = "A witness observed the event."
    source = tmp_path / f"{benchmark}.jsonl"
    source.write_text(json.dumps(common) + "\n")
    return source, [common]


def _receipt(tmp_path: Path, source: Path, benchmark: str) -> tuple[Path, str]:
    training_sha256 = "a" * 64
    path = tmp_path / "disjoint.json"
    path.write_text(
        json.dumps(
            {
                "schema": DISJOINT_RECEIPT_SCHEMA,
                "benchmark": benchmark,
                "benchmark_source_sha256": _sha256_file(source),
                "training_source_sha256": training_sha256,
                "source_disjoint": True,
                "method": "identity-and-contamination-audit",
                "evidence_sha256": "b" * 64,
            }
        )
    )
    return path, training_sha256


def _evaluate(tmp_path: Path, benchmark: str = "mmlu_pro") -> dict[str, object]:
    source, rows = _population(tmp_path, benchmark)
    receipt, training_sha256 = _receipt(tmp_path, source, benchmark)
    artifacts = [
        _artifact(tmp_path, "checkpoint.pt"),
        _artifact(tmp_path, "config.json"),
        _artifact(tmp_path, "tokenizer.json"),
        _artifact(tmp_path, "runtime.py"),
    ]
    return evaluate_development_mc(
        YesBiasedModel(),
        CharacterTokenizer(),
        benchmark=benchmark,
        source_path=source,
        disjoint_receipt_path=receipt,
        training_source_sha256=training_sha256,
        checkpoint_paths=[artifacts[0]],
        config_paths=[artifacts[1]],
        tokenizer_paths=[artifacts[2]],
        runtime_paths=[artifacts[3]],
        expected_rows=1,
        expected_identity_order_sha256=_canonical_sha256([rows[0]["row_id"]]),
        max_sequence_tokens=256,
    )


@pytest.mark.parametrize("benchmark", ["mmlu_pro", "musr"])
def test_exact_supported_schema_scores_normalized_choices_and_binds_artifacts(
    tmp_path: Path, benchmark: str
) -> None:
    result = _evaluate(tmp_path, benchmark)
    assert result["aggregate"] == {"correct": 1, "rows": 1, "accuracy": 1.0}
    assert result["domains"] == {"logic": {"correct": 1, "rows": 1, "accuracy": 1.0}}
    assert result["rows"][0]["predicted_index"] == 0
    assert result["rows"][0]["choice_scores"][0]["choice_token_count"] == 4
    assert result["official_benchmark_result"] is False
    assert result["public_terminal_result"] is False
    assert result["architecture_promotion_allowed"] is False
    unsigned = dict(result)
    receipt_sha256 = unsigned.pop("receipt_sha256")
    assert receipt_sha256 == _canonical_sha256(unsigned)
    assert set(result["bindings"]) == {
        "benchmark_source_sha256",
        "training_source_sha256",
        "source_disjoint_receipt_sha256",
        "identity_order_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "evaluator_code_sha256",
        "runtime_files_sha256",
        "runtime_sha256",
        "decoding_contract_sha256",
        "scoring_contract_sha256",
    }


def test_exact_duplicate_distractors_preserve_official_choice_indices(
    tmp_path: Path,
) -> None:
    source, rows = _population(tmp_path)
    rows[0]["choices"] = ["yes", "no", "yes"]
    rows[0]["answer_index"] = 2
    source.write_text(json.dumps(rows[0]) + "\n")
    receipt, training_sha256 = _receipt(tmp_path, source, "mmlu_pro")
    artifacts = [
        _artifact(tmp_path, "checkpoint.pt"),
        _artifact(tmp_path, "config.json"),
        _artifact(tmp_path, "tokenizer.json"),
        _artifact(tmp_path, "runtime.py"),
    ]
    result = evaluate_development_mc(
        YesBiasedModel(),
        CharacterTokenizer(),
        benchmark="mmlu_pro",
        source_path=source,
        disjoint_receipt_path=receipt,
        training_source_sha256=training_sha256,
        checkpoint_paths=[artifacts[0]],
        config_paths=[artifacts[1]],
        tokenizer_paths=[artifacts[2]],
        runtime_paths=[artifacts[3]],
        expected_rows=1,
        expected_identity_order_sha256=_canonical_sha256([rows[0]["row_id"]]),
        max_sequence_tokens=256,
    )
    assert result["aggregate"] == {"correct": 1, "rows": 1, "accuracy": 1.0}
    assert len(result["rows"][0]["choice_scores"]) == 3
    assert result["rows"][0]["predicted_index"] == 0
    assert result["rows"][0]["answer_index"] == 2
    assert result["rows"][0]["correct"] is True


def test_fail_closed_on_schema_identity_and_disjointness(tmp_path: Path) -> None:
    source, rows = _population(tmp_path)
    row = rows[0]
    row["answer"] = row.pop("answer_index")
    source.write_text(json.dumps(row) + "\n")
    receipt, training_sha256 = _receipt(tmp_path, source, "mmlu_pro")
    artifact = _artifact(tmp_path, "artifact")
    with pytest.raises(DevelopmentMCError, match="schema is unsupported"):
        evaluate_development_mc(
            YesBiasedModel(),
            CharacterTokenizer(),
            benchmark="mmlu_pro",
            source_path=source,
            disjoint_receipt_path=receipt,
            training_source_sha256=training_sha256,
            checkpoint_paths=[artifact],
            config_paths=[artifact],
            tokenizer_paths=[artifact],
            runtime_paths=[artifact],
            expected_rows=1,
            expected_identity_order_sha256=_canonical_sha256(["row-001"]),
            max_sequence_tokens=256,
        )

    source, rows = _population(tmp_path)
    receipt, training_sha256 = _receipt(tmp_path, source, "mmlu_pro")
    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["source_disjoint"] = False
    receipt.write_text(json.dumps(receipt_payload))
    with pytest.raises(DevelopmentMCError, match="source-disjoint evidence"):
        evaluate_development_mc(
            YesBiasedModel(),
            CharacterTokenizer(),
            benchmark="mmlu_pro",
            source_path=source,
            disjoint_receipt_path=receipt,
            training_source_sha256=training_sha256,
            checkpoint_paths=[artifact],
            config_paths=[artifact],
            tokenizer_paths=[artifact],
            runtime_paths=[artifact],
            expected_rows=1,
            expected_identity_order_sha256=_canonical_sha256(["row-001"]),
            max_sequence_tokens=256,
        )


def test_result_write_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    result = _evaluate(tmp_path)
    output = tmp_path / "result.json"
    write_development_mc(output, result)
    assert json.loads(output.read_text())["schema"] == result["schema"]
    with pytest.raises(DevelopmentMCError, match="already exists"):
        write_development_mc(output, result)


def test_specialized_choice_logits_preserve_exact_scoring_contract(
    tmp_path: Path,
) -> None:
    source, rows = _population(tmp_path)
    receipt, training_sha256 = _receipt(tmp_path, source, "mmlu_pro")
    artifact = _artifact(tmp_path, "artifact")
    model = SpecializedYesBiasedModel()
    result = evaluate_development_mc(
        model,
        CharacterTokenizer(),
        benchmark="mmlu_pro",
        source_path=source,
        disjoint_receipt_path=receipt,
        training_source_sha256=training_sha256,
        checkpoint_paths=[artifact],
        config_paths=[artifact],
        tokenizer_paths=[artifact],
        runtime_paths=[artifact],
        expected_rows=1,
        expected_identity_order_sha256=_canonical_sha256([rows[0]["row_id"]]),
        max_sequence_tokens=256,
    )
    assert result["aggregate"]["accuracy"] == 1.0
    assert model.calls == [(40, 4), (40, 3)]
