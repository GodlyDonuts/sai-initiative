from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from sai.training.fla_parity import (
    FlaOperators,
    FlaParityError,
    canonical_sha256,
    run_parity_mechanics,
    validate_receipt,
)


def _mock_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del output_final_state
    assert scale == 1.0
    assert use_qk_l2norm_in_kernel is True
    offsets = (
        [0, q.shape[1]]
        if cu_seqlens is None
        else [int(value) for value in cu_seqlens.cpu().tolist()]
    )
    outputs = []
    states = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        state = torch.zeros(
            q.shape[2], q.shape[3], v.shape[3], device=q.device, dtype=torch.float32
        )
        for index in range(start, stop):
            query = F.normalize(q[0, index].float(), dim=-1)
            key = F.normalize(k[0, index].float(), dim=-1)
            decay = g[0, index].float().exp()
            if decay.ndim == 1:
                decay = decay.unsqueeze(-1)
            state = state * decay.unsqueeze(-1)
            prediction = torch.einsum("hk,hkv->hv", key, state)
            error = beta[0, index].float().unsqueeze(-1) * (
                v[0, index].float() - prediction
            )
            state = state + torch.einsum("hk,hv->hkv", key, error)
            outputs.append(torch.einsum("hk,hkv->hv", query, state))
        states.append(state)
    return (
        torch.stack(outputs).unsqueeze(0).to(q.dtype),
        torch.stack(states),
    )


def _operators(*, corrupt_chunk: bool = False) -> FlaOperators:
    def recurrent(**kwargs):
        return _mock_recurrence(**kwargs)

    def chunk(**kwargs):
        output, state = _mock_recurrence(**kwargs)
        if corrupt_chunk and kwargs["cu_seqlens"] is not None:
            output = output.clone()
            output[:, -1] += 2
        return output, state

    return FlaOperators(
        gated_delta_chunk=chunk,
        gated_delta_recurrent=recurrent,
        kda_chunk=chunk,
        kda_recurrent=recurrent,
        source="unit-test-reference",
        version="0",
        mock=True,
    )


@pytest.fixture(scope="module")
def receipt() -> dict:
    return run_parity_mechanics(_operators(), "cpu", seed=7)


def _resign(payload: dict) -> None:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )


def test_mock_contract_exercises_both_operators_packing_and_backward(
    receipt: dict,
) -> None:
    assert validate_receipt(receipt) == receipt
    assert receipt["status"] == "mock_mechanics_passed"
    assert not receipt["parity_qualified"]
    assert not receipt["production_cuda_qualified"]
    assert [case["family"] for case in receipt["cases"]] == [
        "gated_delta",
        "kda",
    ]
    assert all(
        case["packed_cu_seqlens"] == [0, 67, 98, 103] for case in receipt["cases"]
    )
    assert all(case["backward_calls"] == 2 for case in receipt["cases"])
    assert all(case["all_gradients_finite"] for case in receipt["cases"])
    assert all(case["bf16_forward_backward_mechanics"] for case in receipt["cases"])
    assert receipt["optimizer_steps"] == 0
    assert receipt["gpu_allocation_consumed"] is False
    assert receipt["training_gpu_jobs_submitted"] == 0
    assert not receipt["training_authorized"]
    assert not receipt["architecture_promoted"]
    assert not receipt["four_b_training_authorized"]


def test_corrupt_packed_chunk_fails_parity_without_authorizing_anything() -> None:
    receipt = run_parity_mechanics(_operators(corrupt_chunk=True), "cpu", seed=7)
    assert validate_receipt(receipt) == receipt
    assert receipt["status"] == "parity_failed"
    assert not receipt["parity_qualified"]
    assert not receipt["training_authorized"]
    assert not receipt["four_b_training_authorized"]
    assert any(not case["passed"] for case in receipt["cases"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("training_authorized",), True),
        (("architecture_promoted",), True),
        (("four_b_training_authorized",), True),
        (("optimizer_steps",), 1),
        (("cases", 0, "packed_cu_seqlens"), [0, 103]),
        (("cases", 0, "metrics", "packed_output_chunk_vs_recurrent", "passed"), False),
        (("environment", "device_type"), "cuda"),
        (("operators", "mock"), False),
    ],
)
def test_resigned_tampering_and_false_promotion_fail_closed(
    receipt: dict, path: tuple[object, ...], value: object
) -> None:
    changed = copy.deepcopy(receipt)
    cursor = changed
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value
    _resign(changed)
    with pytest.raises(FlaParityError):
        validate_receipt(changed)
