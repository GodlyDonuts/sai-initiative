from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sai.evaluation.hf_smol_parent import SmolParentError
from sai.evaluation.hf_smol_parent_mc import SmolTextLogitAdapter


class _Output:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _DummySmol(nn.Module):
    def forward(self, *, input_ids, attention_mask, use_cache, logits_to_keep):
        assert bool(attention_mask.eq(1).all())
        assert use_cache is False
        assert logits_to_keep == 0
        return _Output(torch.zeros((*input_ids.shape, 9)))


def test_smol_text_adapter_exposes_logits_and_rejects_segments() -> None:
    adapter = SmolTextLogitAdapter(_DummySmol())
    inputs = torch.tensor([[1, 2, 3]])
    assert adapter(inputs, torch.zeros_like(inputs)).shape == (1, 3, 9)
    with pytest.raises(SmolParentError, match="unsegmented"):
        adapter(inputs, torch.tensor([[0, 1, 1]]))


def test_smol_mc_worker_is_one_h100_offline_and_no_retry() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs/sai-smollm3-3b-development-mc-single-h100.sbatch"
    ).read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "evc50" in job
    assert "MODEL_MANIFEST" in job and "RESTORATION_RECEIPT" in job
    assert "HF_HUB_OFFLINE=1" in job
    assert "TRANSFORMERS_OFFLINE=1" in job
    assert "sai.evaluation.hf_smol_parent_mc" in job
    assert "retry" not in job.lower()
