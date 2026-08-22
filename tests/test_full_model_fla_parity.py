from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sai.model.fla_backend import FLA_VERSION, FlaBackendOperators
from sai.model.reference import causal_delta_recurrence
from sai.training.full_model_fla_parity import (
    FullModelFlaParityError,
    run_full_delta_mixer_parity,
)


def _spans(cu_seqlens: torch.Tensor | None, length: int) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, length)]
    offsets = cu_seqlens.detach().cpu().tolist()
    return list(zip(offsets[:-1], offsets[1:], strict=True))


def _oracle_conv(**kwargs):
    value = kwargs["x"]
    weight = kwargs["weight"].unsqueeze(1)
    outputs = []
    for start, stop in _spans(kwargs["cu_seqlens"], value.shape[1]):
        segment = value[:, start:stop]
        convolved = F.conv1d(
            segment.transpose(1, 2),
            weight,
            padding=weight.shape[-1] - 1,
            groups=value.shape[-1],
        )[..., : stop - start]
        outputs.append(F.silu(convolved.transpose(1, 2)))
    return torch.cat(outputs, dim=1)


def _oracle_delta(*, perturb: float = 0.0) -> Callable[..., tuple[torch.Tensor, None]]:
    def execute(**kwargs):
        query = kwargs["q"]
        key = kwargs["k"]
        value = kwargs["v"]
        log_decay = kwargs["g"]
        beta = kwargs["beta"].unsqueeze(-1)
        assert kwargs["scale"] == 1.0
        assert kwargs["use_qk_l2norm_in_kernel"] is False
        outputs = []
        for start, stop in _spans(kwargs["cu_seqlens"], value.shape[1]):
            alpha = log_decay[:, start:stop].exp()
            if alpha.ndim == 3:
                alpha = alpha.unsqueeze(-1)
            output, _ = causal_delta_recurrence(
                query[:, start:stop],
                key[:, start:stop],
                value[:, start:stop],
                alpha,
                beta[:, start:stop],
            )
            outputs.append(output)
        combined = torch.cat(outputs, dim=1)
        return combined + perturb, None

    return execute


def _operators(*, perturb: float = 0.0) -> FlaBackendOperators:
    delta = _oracle_delta(perturb=perturb)
    return FlaBackendOperators(
        gated_delta_chunk=delta,
        kda_chunk=delta,
        causal_conv1d=_oracle_conv,
        version=FLA_VERSION,
    )


def test_full_delta_mixer_mapping_matches_reference_for_all_boundaries() -> None:
    report = run_full_delta_mixer_parity(
        device="cpu", operators=_operators(), seed=20260821
    )

    assert report["status"] == "test_oracle_passed"
    assert report["production_cuda_qualified"] is False
    assert len(report["cases"]) == 8
    assert {(case["family"], case["sequence_length"]) for case in report["cases"]} == {
        (family, length) for family in ("gdn", "kda") for length in (1, 63, 64, 65)
    }
    assert all(case["passed"] for case in report["cases"])
    assert all(
        len(case["causal_convolution_forward"]) == 3
        and all(metric["passed"] for metric in case["causal_convolution_forward"])
        for case in report["cases"]
    )
    assert all(case["forward"]["elements_compared"] > 0 for case in report["cases"])
    assert all(
        case["forward"]["elements_outside_tolerance"] == 0 for case in report["cases"]
    )
    assert all(len(case["forward"]["worst_index"]) == 3 for case in report["cases"])
    assert all(case["fla_mapping"]["q_shape"][-1] == 16 for case in report["cases"])
    assert {
        case["sequence_length"]: case["packed_cu_seqlens"]
        for case in report["cases"]
        if case["family"] == "gdn"
    } == {
        1: [0, 1, 2],
        63: [0, 63, 126],
        64: [0, 63, 64, 128],
        65: [0, 1, 64, 65, 130],
    }
    assert all(case["equal_segment_id_across_row_boundary"] for case in report["cases"])
    assert all(case["fla_mapping"]["scale"] == 1.0 for case in report["cases"])
    assert all(
        case["fla_mapping"]["family_specific_flags_passed"] for case in report["cases"]
    )
    assert all(
        set(case["parameter_gradients"]) == set(case["expected_parameter_names"])
        for case in report["cases"]
    )
    assert report["limitations"] == [
        "small_reference_geometry_only",
        "not_the_exact_b8_x_2048_canary",
        "mechanics_only_not_model_quality_evidence",
        "no_optimizer_or_parameter_update",
    ]


def test_full_delta_mixer_parity_fails_closed_on_operator_drift() -> None:
    report = run_full_delta_mixer_parity(
        device="cpu", operators=_operators(perturb=0.25), seed=20260821
    )

    assert report["status"] == "parity_failed"
    assert report["production_cuda_qualified"] is False
    assert any(not case["passed"] for case in report["cases"])
    assert report["training_authorized"] is False
    assert report["architecture_promoted"] is False


def test_production_path_refuses_cpu_and_injected_version_drift() -> None:
    with pytest.raises(FullModelFlaParityError, match="requires CUDA"):
        run_full_delta_mixer_parity(device="cpu")

    bad = _operators()
    bad = FlaBackendOperators(
        gated_delta_chunk=bad.gated_delta_chunk,
        kda_chunk=bad.kda_chunk,
        causal_conv1d=bad.causal_conv1d,
        version="unqualified",
    )
    with pytest.raises(FullModelFlaParityError, match="version differs"):
        run_full_delta_mixer_parity(device="cpu", operators=bad)


def test_job_is_one_h100_no_training_no_requeue() -> None:
    job = (
        Path(__file__).parents[1]
        / "jobs"
        / "sai-full-model-fla-parity-single-h100.sbatch"
    ).read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "--no-requeue" in job
    assert "--time=00:30:00" in job
    assert "sai.training.full_model_fla_parity" in job
    assert "optimizer" not in job.lower()
    assert "backward" not in job.lower()
    assert "4b" not in job.lower()
    assert "retry" not in job.lower()
