from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.model.fla_backend import FLA_VERSION, FlaBackendOperators
from sai.training.fla_semantic_aggregate import (
    FlaSemanticAggregateError,
    aggregate_seed_receipts,
)
from sai.training.fla_semantic_parity import (
    PRODUCTION_SEEDS,
    _reference_conv,
    _reference_recurrence,
    run_semantic_parity,
)


def _offsets(kwargs: dict, total: int) -> list[int]:
    value = kwargs.get("cu_seqlens")
    return [0, total] if value is None else value.detach().cpu().tolist()


def _oracle_conv(**kwargs):
    return _reference_conv(
        kwargs["x"], kwargs["weight"], _offsets(kwargs, kwargs["x"].shape[1])
    )


def _oracle_delta(**kwargs):
    output = _reference_recurrence(
        kwargs["q"],
        kwargs["k"],
        kwargs["v"],
        kwargs["g"],
        kwargs["beta"],
        _offsets(kwargs, kwargs["q"].shape[1]),
    )
    return output, None


def _operators() -> FlaBackendOperators:
    return FlaBackendOperators(
        gated_delta_chunk=_oracle_delta,
        kda_chunk=_oracle_delta,
        causal_conv1d=_oracle_conv,
        version=FLA_VERSION,
    )


def _resign(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


def _production_payload(seed: int) -> dict:
    payload = run_semantic_parity(seed=seed, device="cpu", operators=_operators())
    payload["status"] = "all_families_production_qualified"
    payload["production_cuda_qualified"] = True
    payload["dtype"] = "torch.bfloat16"
    payload["environment"] = {
        "device_type": "cuda",
        "torch": "2.9.0+cu129",
        "torch_cuda": "12.9",
        "cuda_available": True,
        "cuda_bf16_supported": True,
        "cuda_device_name": "NVIDIA H100 PCIe",
        "cuda_capability": [9, 0],
    }
    payload["operators"] = {
        "causal_conv1d": "fla.modules.convolution.causal_conv1d",
        "gated_delta_chunk": "fla.ops.gated_delta_rule.chunk_gated_delta_rule",
        "kda_chunk": "fla.ops.kda.chunk_kda",
    }
    for family in ("gdn", "kda"):
        payload["family_results"][family] = {
            "status": "production_semantics_qualified",
            "production_semantics_qualified": True,
            "passed_cases": 4,
            "required_cases": 4,
        }
    return _resign(payload)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _population(tmp_path: Path) -> list[Path]:
    return [
        _write(tmp_path / f"seed-{seed}.json", _production_payload(seed))
        for seed in PRODUCTION_SEEDS
    ]


def test_aggregate_reopens_and_qualifies_exact_three_seed_population(
    tmp_path: Path,
) -> None:
    aggregate = aggregate_seed_receipts(_population(tmp_path))

    assert aggregate["schema"] == "sai-fla-semantic-parity-aggregate-v2"
    assert aggregate["status"] == "all_families_production_qualified"
    assert aggregate["production_cuda_qualified"] is True
    assert aggregate["seeds"] == list(PRODUCTION_SEEDS)
    assert [value["seed"] for value in aggregate["inputs"]] == list(PRODUCTION_SEEDS)
    assert all(len(value["file_sha256"]) == 64 for value in aggregate["inputs"])
    for family in ("gdn", "kda"):
        assert aggregate["family_results"][family] == {
            "status": "production_semantics_qualified",
            "production_semantics_qualified": True,
            "passed_seeds": 3,
            "required_seeds": 3,
            "failed_seeds": [],
        }
    assert aggregate["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "receipt_sha256"}
    )
    assert aggregate["training_authorized"] is False
    assert aggregate["architecture_promoted"] is False
    assert aggregate["four_b_training_authorized"] is False


def test_missing_tampered_and_duplicate_receipts_fail_closed(tmp_path: Path) -> None:
    population = _population(tmp_path)
    with pytest.raises(FlaSemanticAggregateError, match="exactly three"):
        aggregate_seed_receipts(population[:2])
    with pytest.raises(FlaSemanticAggregateError, match="missing or unsafe"):
        aggregate_seed_receipts([*population[:2], tmp_path / "missing.json"])

    tampered = json.loads(population[0].read_text())
    tampered["dtype"] = "torch.float32"
    population[0].write_text(json.dumps(tampered))
    with pytest.raises(FlaSemanticAggregateError, match="hash differs"):
        aggregate_seed_receipts(population)

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    duplicate = [
        _write(duplicate_root / "first.json", _production_payload(PRODUCTION_SEEDS[0])),
        _write(
            duplicate_root / "second.json", _production_payload(PRODUCTION_SEEDS[0])
        ),
        _write(duplicate_root / "third.json", _production_payload(PRODUCTION_SEEDS[2])),
    ]
    with pytest.raises(FlaSemanticAggregateError, match="seed population"):
        aggregate_seed_receipts(duplicate)


@pytest.mark.parametrize("drift", ["environment", "operator"])
def test_cross_seed_runtime_identity_drift_fails_closed(
    tmp_path: Path, drift: str
) -> None:
    population = _population(tmp_path)
    changed = json.loads(population[1].read_text())
    if drift == "environment":
        changed["environment"]["cuda_device_name"] = "different H100"
    else:
        changed["operators"]["kda_chunk"] = "different.module.chunk_kda"
    _write(population[1], _resign(changed))
    with pytest.raises(FlaSemanticAggregateError, match=f"{drift} identity drifted"):
        aggregate_seed_receipts(population)


def test_one_kda_seed_veto_is_preserved_without_revoking_gdn(tmp_path: Path) -> None:
    population = _population(tmp_path)
    failed = json.loads(population[2].read_text())
    case = next(
        item
        for item in failed["cases"]
        if item["family"] == "kda" and item["sequence_length"] == 64
    )
    metric = case["packed_recurrence"]["metrics"]["o"]
    metric["relative_root_mean_square_error"] = 0.006
    metric["passed"] = False
    case["packed_recurrence"]["passed"] = False
    case["passed"] = False
    failed["family_results"]["kda"] = {
        "status": "semantic_parity_failed",
        "production_semantics_qualified": False,
        "passed_cases": 3,
        "required_cases": 4,
    }
    failed["status"] = "one_or_more_families_failed"
    failed["production_cuda_qualified"] = False
    _write(population[2], _resign(failed))

    aggregate = aggregate_seed_receipts(population)
    assert aggregate["production_cuda_qualified"] is False
    assert aggregate["family_results"]["gdn"]["production_semantics_qualified"]
    assert aggregate["family_results"]["kda"] == {
        "status": "semantic_parity_failed",
        "production_semantics_qualified": False,
        "passed_seeds": 2,
        "required_seeds": 3,
        "failed_seeds": [PRODUCTION_SEEDS[2]],
    }
