from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from sai.data.token_stream import canonical_sha256
from sai.model.fla_backend import FLA_VERSION, FlaBackendOperators
from sai.training.fla_semantic_parity import (
    RECURRENCE_THRESHOLDS,
    FlaSemanticParityError,
    _reference_recurrence,
)
from sai.training.fla_semantic_parity_v3 import (
    LANE_A_THRESHOLD,
    PRODUCTION_SEEDS,
    _bf16_half_ulp,
    _noninferiority_metric,
    _ordered_bf16,
    _reference_conv_activation,
    run_semantic_parity_v3,
)


def _offsets(kwargs: dict, total: int) -> list[int]:
    cu_seqlens = kwargs.get("cu_seqlens")
    return [0, total] if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()


def _oracle_conv(*, perturb: float = 0.0) -> Callable[..., torch.Tensor]:
    def execute(**kwargs):
        output = _reference_conv_activation(
            kwargs["x"],
            kwargs["weight"],
            _offsets(kwargs, kwargs["x"].shape[1]),
            activation=kwargs["activation"],
        )
        return output + perturb

    return execute


def _oracle_delta(*, perturb: float = 0.0) -> Callable[..., tuple[torch.Tensor, None]]:
    def execute(**kwargs):
        output = _reference_recurrence(
            kwargs["q"],
            kwargs["k"],
            kwargs["v"],
            kwargs["g"],
            kwargs["beta"],
            _offsets(kwargs, kwargs["q"].shape[1]),
        )
        return output + perturb, None

    return execute


def _operators(
    *, conv_perturb: float = 0.0, kda_perturb: float = 0.0
) -> FlaBackendOperators:
    return FlaBackendOperators(
        gated_delta_chunk=_oracle_delta(),
        kda_chunk=_oracle_delta(perturb=kda_perturb),
        causal_conv1d=_oracle_conv(perturb=conv_perturb),
        version=FLA_VERSION,
    )


def test_v3_dual_lane_oracle_passes_every_unaveraged_veto() -> None:
    report = run_semantic_parity_v3(
        seed=PRODUCTION_SEEDS[0], device="cpu", operators=_operators()
    )

    assert report["schema"] == "sai-fla-semantic-parity-v3"
    assert report["status"] == "test_oracle_passed"
    assert report["production_cuda_qualified"] is False
    assert report["mechanics_canary_admitted"] is False
    assert report["acceptance"]["lane_a"] == {
        "strict_relative_rmse_less_than": LANE_A_THRESHOLD,
        "fp32_activation": "swish",
        "fp16_activation": None,
        "bf16_qualification": False,
    }
    assert report["acceptance"]["lane_b"]["fitted_scalar_threshold"] is None
    assert report["acceptance"]["packed_recurrence"] == RECURRENCE_THRESHOLDS
    assert len(report["cases"]) == 8
    assert all(case["passed"] for case in report["cases"])
    for case in report["cases"]:
        assert case["structural_mapping"]["passed"]
        assert case["structural_mapping"]["explicit_scale_one"]
        assert case["structural_mapping"]["equal_id_across_row_boundary"]
        assert len(case["lane_a_upstream_fp32_fp16"]) == 3
        assert len(case["lane_b_bf16_noninferiority"]) == 3
        for path in case["lane_a_upstream_fp32_fp16"]:
            assert path["passed"]
            assert [probe["dtype"] for probe in path["probes"]] == [
                "torch.float32",
                "torch.float16",
            ]
            assert all(not probe["bf16_qualification"] for probe in path["probes"])
            assert all(
                set(probe["metrics"]) == {"y", "dx", "dw"} for probe in path["probes"]
            )
        for path in case["lane_b_bf16_noninferiority"]:
            assert path["passed"]
            assert path["quantized_once_to"] == "torch.bfloat16"
            assert path["oracle_dtype"] == "torch.float64"
            assert all(metric["passed"] for metric in path["metrics"].values())
        assert case["packed_recurrence"]["passed"]
    assert report["family_results"] == {
        "gdn": {
            "status": "test_oracle_passed",
            "production_semantics_qualified": False,
            "passed_cases": 4,
            "required_cases": 4,
        },
        "kda": {
            "status": "test_oracle_passed",
            "production_semantics_qualified": False,
            "passed_cases": 4,
            "required_cases": 4,
        },
    }
    assert report["optimizer_steps"] == 0
    assert report["training_gpu_jobs_submitted"] == 0
    assert report["training_authorized"] is False
    assert report["architecture_promoted"] is False
    assert report["four_b_training_authorized"] is False
    assert report["long_training_authorized"] is False
    assert report["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "receipt_sha256"}
    )


def test_ordered_bf16_handles_signed_zero_subnormals_and_sign() -> None:
    minimum_subnormal = torch.tensor(float.fromhex("0x1p-133"), dtype=torch.bfloat16)
    values = torch.stack(
        (
            -torch.tensor(1.0, dtype=torch.bfloat16),
            -minimum_subnormal,
            torch.tensor(-0.0, dtype=torch.bfloat16),
            torch.tensor(0.0, dtype=torch.bfloat16),
            minimum_subnormal,
            torch.tensor(1.0, dtype=torch.bfloat16),
        )
    )
    ordered = _ordered_bf16(values)

    assert ordered.tolist() == sorted(ordered.tolist())
    assert ordered[2].item() == ordered[3].item()
    assert ordered[3].item() - ordered[1].item() == 1
    assert ordered[4].item() - ordered[3].item() == 1
    half_ulp = _bf16_half_ulp(values.double())
    assert torch.isfinite(half_ulp).all()
    assert half_ulp[1].item() == half_ulp[2].item() == half_ulp[4].item()
    metric = _noninferiority_metric(values, values, values.double())
    assert metric["passed"]


def test_bf16_lane_rejects_elementwise_and_ulp_inferiority() -> None:
    oracle = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    torch_value = oracle.to(torch.bfloat16)
    fla = torch.tensor([2.0, -2.0, 1.0], dtype=torch.bfloat16)

    metric = _noninferiority_metric(fla, torch_value, oracle)

    assert metric["elementwise_envelope_passed"] is False
    assert metric["rms_envelope_passed"] is False
    assert metric["ordered_bf16_ulp_passed"] is False
    assert metric["passed"] is False


def test_one_kda_recurrence_failure_remains_family_separated() -> None:
    report = run_semantic_parity_v3(
        seed=PRODUCTION_SEEDS[0],
        device="cpu",
        operators=_operators(kda_perturb=0.1),
    )

    assert report["status"] == "one_or_more_families_failed"
    assert report["family_results"]["gdn"]["status"] == "test_oracle_passed"
    assert report["family_results"]["gdn"]["passed_cases"] == 4
    assert report["family_results"]["kda"]["status"] == "semantic_parity_failed"
    assert report["family_results"]["kda"]["passed_cases"] == 0
    assert report["production_cuda_qualified"] is False
    assert report["mechanics_canary_admitted"] is False


def test_production_refuses_nonfrozen_seed_cpu_and_version_drift() -> None:
    with pytest.raises(FlaSemanticParityError, match="prospectively frozen"):
        run_semantic_parity_v3(seed=20260826, device="cuda")
    with pytest.raises(FlaSemanticParityError, match="requires CUDA"):
        run_semantic_parity_v3(seed=PRODUCTION_SEEDS[0], device="cpu")
    bad = _operators()
    bad = FlaBackendOperators(
        gated_delta_chunk=bad.gated_delta_chunk,
        kda_chunk=bad.kda_chunk,
        causal_conv1d=bad.causal_conv1d,
        version="unqualified",
    )
    with pytest.raises(FlaSemanticParityError, match="version differs"):
        run_semantic_parity_v3(seed=PRODUCTION_SEEDS[0], device="cpu", operators=bad)


def test_v3_job_is_single_h100_create_only_and_never_submits_training() -> None:
    job = (
        Path(__file__).parents[1]
        / "jobs"
        / "sai-fla-semantic-parity-v3-single-h100.sbatch"
    ).read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert ': "${SEED:?SEED is required}"' in job
    assert "20260827|20260828|20260829" in job
    assert "ca910f8" in job
    assert "sai.training.fla_semantic_parity_v3" in job
    assert '--fla-root "$FLA_ROOT"' in job
    assert 'git -C "$FLA_ROOT" status --short' in job
    executable = "\n".join(
        line for line in job.splitlines() if not line.startswith("#SBATCH")
    )
    assert "sbatch " not in executable.lower()
    assert "scancel" not in executable.lower()
    assert "retry" not in job.lower()
    assert "4b" not in job.lower()
