from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sai.evaluation.hf_parent import (
    EXPECTED_EOS_TOKEN_ID,
    EXPECTED_MODEL_CLASS,
    EXPECTED_MODEL_VOCAB_SIZE,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_TOKENIZER_BASE_VOCAB_SIZE,
    EXPECTED_TOKENIZER_LENGTH,
    HFParentError,
    HFTextLogitAdapter,
    _loading_strings,
)


class _Output:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _DummyParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        assert attention_mask.eq(1).all()
        assert use_cache is False
        assert logits_to_keep == 0
        return _Output(torch.zeros((*input_ids.shape, 7)))


def test_parent_identity_constants_are_exact() -> None:
    assert EXPECTED_MODEL_CLASS == "Qwen3_5ForCausalLM"
    assert EXPECTED_PARAMETER_COUNT == 752_393_024
    assert EXPECTED_MODEL_VOCAB_SIZE == 248_320
    assert EXPECTED_TOKENIZER_BASE_VOCAB_SIZE == 248_044
    assert EXPECTED_TOKENIZER_LENGTH == 248_077
    assert EXPECTED_EOS_TOKEN_ID == 248_046


def test_loading_key_evidence_accepts_transformers_sets_but_not_bad_content() -> None:
    assert _loading_strings(set(), "keys") == []
    assert _loading_strings({"b", "a"}, "keys") == ["a", "b"]
    with pytest.raises(HFParentError, match="evidence differs"):
        _loading_strings("a", "keys")
    with pytest.raises(HFParentError, match="evidence differs"):
        _loading_strings({1}, "keys")


def test_text_adapter_exposes_logits_and_rejects_segments() -> None:
    adapter = HFTextLogitAdapter(_DummyParent())
    input_ids = torch.tensor([[1, 2, 3]])
    segment_ids = torch.zeros_like(input_ids)
    assert adapter(input_ids, segment_ids).shape == (1, 3, 7)
    with pytest.raises(HFParentError, match="unsegmented"):
        adapter(input_ids, torch.tensor([[0, 1, 1]]))


def test_mechanics_job_is_one_h100_offline_and_no_retry() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-qwen35-0p8b-mechanics-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "evc50" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
    assert "retry" not in job.lower()


def test_parent_mc_job_is_one_h100_offline_and_no_retry() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-qwen35-0p8b-development-mc-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "MECHANICS_RECEIPT_SHA256" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
    assert "retry" not in job.lower()
