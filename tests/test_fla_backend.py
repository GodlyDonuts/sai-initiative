from __future__ import annotations

import torch

from sai.model.fla_backend import (
    FLA_VERSION,
    FlaBackendOperators,
    fla_causal_conv1d,
    fla_delta_recurrence,
    packed_cu_seqlens,
)


class Calls:
    def __init__(self) -> None:
        self.delta: list[dict] = []
        self.conv: list[dict] = []

    def chunk(self, **kwargs):
        self.delta.append(kwargs)
        return kwargs["v"], None

    def causal_conv(self, **kwargs):
        self.conv.append(kwargs)
        return kwargs["x"]


def operators(calls: Calls) -> FlaBackendOperators:
    return FlaBackendOperators(
        gated_delta_chunk=calls.chunk,
        kda_chunk=calls.chunk,
        causal_conv1d=calls.causal_conv,
        version=FLA_VERSION,
    )


def test_packed_offsets_never_merge_equal_ids_across_rows() -> None:
    segment_ids = torch.tensor(
        [[41, 41, 41, 41, 97, 97, 97], [97, 11, 11, 13, 13, 13, 13]]
    )
    assert packed_cu_seqlens(segment_ids).tolist() == [0, 4, 7, 8, 10, 14]


def test_delta_adapter_materializes_exact_sai_mapping_and_packing() -> None:
    calls = Calls()
    segment_ids = torch.tensor([[0, 0, 1], [1, 2, 2]])
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 3, 2, 4)
    v = torch.randn(2, 3, 2, 5)
    alpha = torch.sigmoid(torch.randn(2, 3, 2, 4))
    beta = torch.sigmoid(torch.randn(2, 3, 2, 1))
    output = fla_delta_recurrence(
        q,
        k,
        v,
        alpha,
        beta,
        segment_ids,
        channel_wise_decay=True,
        operators=operators(calls),
    )
    torch.testing.assert_close(output, v)
    call = calls.delta[0]
    assert call["q"].shape == (1, 6, 2, 4)
    torch.testing.assert_close(call["q"].float().norm(dim=-1), torch.ones(1, 6, 2))
    torch.testing.assert_close(call["g"].exp(), alpha.reshape(1, 6, 2, 4))
    assert call["beta"].shape == (1, 6, 2)
    assert call["scale"] == 1.0
    assert call["use_qk_l2norm_in_kernel"] is False
    assert call["use_gate_in_kernel"] is False
    assert call["cu_seqlens"].tolist() == [0, 2, 3, 4, 6]


def test_convolution_adapter_reuses_weights_and_exact_offsets() -> None:
    calls = Calls()
    value = torch.randn(2, 4, 6)
    weight = torch.randn(6, 1, 4)
    segment_ids = torch.tensor([[5, 5, 7, 7], [7, 7, 7, 9]])
    output = fla_causal_conv1d(value, weight, segment_ids, operators=operators(calls))
    torch.testing.assert_close(output, value)
    call = calls.conv[0]
    assert call["x"].shape == (1, 8, 6)
    assert call["weight"].data_ptr() == weight[:, 0, :].data_ptr()
    assert call["activation"] == "silu"
    assert call["cu_seqlens"].tolist() == [0, 2, 4, 7, 8]
