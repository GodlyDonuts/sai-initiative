from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from sai.data.token_stream import canonical_sha256
from sai.model.fla_backend import FLA_VERSION, FlaBackendOperators
from sai.training.fla_semantic_parity import (
    CALIBRATION_SEEDS,
    CONV_THRESHOLDS,
    PRODUCTION_SEEDS,
    RECURRENCE_THRESHOLDS,
    FlaSemanticParityError,
    _reference_conv,
    _reference_recurrence,
    run_semantic_parity,
)


def _offsets(kwargs: dict, total: int) -> list[int]:
    cu_seqlens = kwargs.get("cu_seqlens")
    return [0, total] if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()


def _oracle_conv(**kwargs):
    return _reference_conv(
        kwargs["x"],
        kwargs["weight"],
        _offsets(kwargs, kwargs["x"].shape[1]),
    )


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


def _operators(*, kda_perturb: float = 0.0) -> FlaBackendOperators:
    return FlaBackendOperators(
        gated_delta_chunk=_oracle_delta(),
        kda_chunk=_oracle_delta(perturb=kda_perturb),
        causal_conv1d=_oracle_conv,
        version=FLA_VERSION,
    )


def test_semantic_v2_oracle_passes_every_tensor_without_averaging() -> None:
    report = run_semantic_parity(
        seed=PRODUCTION_SEEDS[0], device="cpu", operators=_operators()
    )

    assert report["schema"] == "sai-fla-semantic-parity-v2"
    assert report["status"] == "test_oracle_passed"
    assert report["production_cuda_qualified"] is False
    assert report["thresholds"] == {
        "strict_less_than": True,
        "causal_convolution": CONV_THRESHOLDS,
        "packed_recurrence": RECURRENCE_THRESHOLDS,
    }
    assert len(report["cases"]) == 8
    assert all(case["passed"] for case in report["cases"])
    assert all(case["structural_mapping"]["passed"] for case in report["cases"])
    assert all(
        len(case["causal_convolution"]) == 3
        and all(probe["passed"] for probe in case["causal_convolution"])
        for case in report["cases"]
    )
    assert all(case["packed_recurrence"]["passed"] for case in report["cases"])
    for case in report["cases"]:
        for probe in case["causal_convolution"]:
            assert set(probe["metrics"]) == set(CONV_THRESHOLDS)
            assert all(metric["all_finite"] for metric in probe["metrics"].values())
        assert set(case["packed_recurrence"]["metrics"]) == set(RECURRENCE_THRESHOLDS)
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
    assert report["training_authorized"] is False
    assert report["architecture_promoted"] is False
    assert report["four_b_training_authorized"] is False
    assert report["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "receipt_sha256"}
    )


def test_one_kda_primitive_failure_does_not_fail_or_qualify_gdn() -> None:
    report = run_semantic_parity(
        seed=PRODUCTION_SEEDS[0],
        device="cpu",
        operators=_operators(kda_perturb=0.1),
    )

    assert report["status"] == "one_or_more_families_failed"
    assert report["production_cuda_qualified"] is False
    assert report["family_results"]["gdn"] == {
        "status": "test_oracle_passed",
        "production_semantics_qualified": False,
        "passed_cases": 4,
        "required_cases": 4,
    }
    assert report["family_results"]["kda"]["status"] == "semantic_parity_failed"
    assert report["family_results"]["kda"]["passed_cases"] == 0


def test_production_refuses_calibration_seed_cpu_and_version_drift() -> None:
    with pytest.raises(FlaSemanticParityError, match="prospectively frozen"):
        run_semantic_parity(seed=CALIBRATION_SEEDS[0], device="cuda")
    with pytest.raises(FlaSemanticParityError, match="requires CUDA"):
        run_semantic_parity(seed=PRODUCTION_SEEDS[0], device="cpu")
    bad = _operators()
    bad = FlaBackendOperators(
        gated_delta_chunk=bad.gated_delta_chunk,
        kda_chunk=bad.kda_chunk,
        causal_conv1d=bad.causal_conv1d,
        version="unqualified",
    )
    with pytest.raises(FlaSemanticParityError, match="version differs"):
        run_semantic_parity(seed=PRODUCTION_SEEDS[0], device="cpu", operators=bad)


def test_semantic_job_is_one_h100_and_never_submits_training() -> None:
    job = (
        Path(__file__).parents[1]
        / "jobs"
        / "sai-fla-semantic-parity-single-h100.sbatch"
    ).read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --no-requeue" in job
    assert ': "${SEED:?SEED is required}"' in job
    assert "20260824|20260825|20260826" in job
    assert "sai.training.fla_semantic_parity" in job
    executable = "\n".join(
        line for line in job.splitlines() if not line.startswith("#SBATCH")
    )
    assert "sbatch " not in executable.lower()
    assert "scancel" not in executable.lower()
    assert "optimizer" not in job.lower()
    assert "training_authorized" not in job
    assert "4b" not in job.lower()
    assert "retry" not in job.lower()
