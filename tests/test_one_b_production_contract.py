from __future__ import annotations

import copy

import pytest

from sai.training.one_b_production_contract import (
    OneBProductionContractError,
    build_contract,
    validate_contract,
)


def test_exact_olmo2_derived_one_b_ledger_and_terminal_batch() -> None:
    value = validate_contract(build_contract())
    assert value["parameter_ledger"]["total"] == 1_006_241_792
    assert value["model"]["max_sequence_length"] == 4_096
    assert value["model"]["weight_tying"] is False
    assert value["distributed"]["terminal_partial_batch_sequences"] == 324
    assert value["architecture_novelty_claimed"] is False
    assert value["model_training_started"] is False


def test_contract_rejects_architecture_drift() -> None:
    value = copy.deepcopy(build_contract())
    value["model"]["n_layers"] = 17
    with pytest.raises(OneBProductionContractError, match="differs"):
        validate_contract(value)
