from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import sai.data.authored_review_context as context
from sai.data.authored_review_context import AuthoredReviewContextError, run, validate

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts" / "authored-curriculum-sources-r1"
JOB = ROOT / "jobs" / "sai-authored-review-context-cpu.sbatch"


class _Tokenizer:
    def __len__(self) -> int:
        return 100

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        return torch.arange(100 + len(messages[1]["content"]) // 16).unsqueeze(0)


class _BatchEncodingTokenizer(_Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": super().apply_chat_template(messages, **kwargs)}


def _kwargs(tmp_path: Path) -> dict:
    packet = ARTIFACT / "authored-curriculum-blind-review.jsonl"
    receipt = ARTIFACT / "authored-curriculum-review-receipt.json"
    return {
        "reviewer": "qwen35_9b",
        "model_root": tmp_path / "model",
        "manifest": tmp_path / "manifest.json",
        "restoration_receipt": tmp_path / "restoration.json",
        "review_packet": packet,
        "review_packet_receipt": receipt,
        "expected_review_packet_sha256": hashlib.sha256(
            packet.read_bytes()
        ).hexdigest(),
        "expected_review_packet_receipt_sha256": hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest(),
        "concept_list": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "annotation_policy": ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json",
        "output": tmp_path / "context.json",
    }


def test_context_receipt_covers_all_blind_rows_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _snapshot(root, *, manifest_path, receipt_path, spec):
        assert root == tmp_path / "model"
        assert manifest_path == tmp_path / "manifest.json"
        assert receipt_path == tmp_path / "restoration.json"
        assert spec == context.REVIEWERS["qwen35_9b"]
        return {"tree": "exact"}

    monkeypatch.setattr(context, "validate_external_snapshot", _snapshot)
    monkeypatch.setattr(context, "_tokenizer", lambda path: (_Tokenizer(), "test"))
    kwargs = _kwargs(tmp_path)
    payload = run(**kwargs)
    assert payload["status"] == "pass"
    assert payload["row_count"] == 127
    assert payload["model_loaded"] is False
    assert payload["gpu_jobs_submitted"] == 0
    assert validate(**kwargs) == payload
    kwargs["output"].chmod(0o644)
    kwargs["output"].write_text(
        kwargs["output"].read_text().replace('"pass"', '"fail"', 1)
    )
    with pytest.raises(AuthoredReviewContextError, match="receipt differs"):
        validate(**kwargs)


def test_context_accepts_exact_input_only_batch_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        context, "validate_external_snapshot", lambda *a, **k: {"tree": "exact"}
    )
    monkeypatch.setattr(
        context, "_tokenizer", lambda path: (_BatchEncodingTokenizer(), "test")
    )
    assert run(**_kwargs(tmp_path))["row_count"] == 127


def test_context_rejects_over_budget_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TooLong(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return torch.arange(context.MAX_INPUT_TOKENS + 1).unsqueeze(0)

    monkeypatch.setattr(
        context, "validate_external_snapshot", lambda *a, **k: {"tree": "exact"}
    )
    monkeypatch.setattr(context, "_tokenizer", lambda path: (_TooLong(), "test"))
    with pytest.raises(AuthoredReviewContextError, match="exceeds"):
        run(**_kwargs(tmp_path))


def test_context_job_is_cpu_only_sealed_and_nontraining() -> None:
    script = JOB.read_text()
    assert "#SBATCH --gres" not in script
    assert "#SBATCH --no-requeue" in script
    assert 'export PYTHONPATH="$SAI_ROOT/src"' in script
    assert "export GIT_OPTIONAL_LOCKS=0" in script
    assert "sai.data.authored_review_context" in script
    assert "optimizer" not in script.lower()
    assert "torchrun" not in script
