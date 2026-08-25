from __future__ import annotations

from sai.training.one_b_olmo_config import _model


def test_olmo_model_geometry_matches_effective_swiglu_contract() -> None:
    model = _model(eos_token_id=7, pad_token_id=8)
    assert model["d_model"] == 2_048
    assert model["n_layers"] == 16
    assert model["mlp_hidden_size"] == 11_008
    assert model["activation_type"] == "swiglu"
    assert model["vocab_size"] == model["embedding_size"] == 48_000
    assert model["eos_token_id"] == 7
    assert model["pad_token_id"] == 8
    assert model["norm_after"] is True
    assert model["attention_layer_norm"] is True
