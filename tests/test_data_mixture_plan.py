from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sai.data.mixture_plan import (
    SCHEMA,
    DataMixturePlanError,
    validate_payload,
    validate_plan,
)
from sai.data.token_stream import canonical_sha256


def _source(
    index: int,
    *,
    source_class: str,
    domain: str,
    planned_tokens: int,
    minimum_phase: str = "grounding",
    rehearsal_required: bool = True,
) -> dict:
    return {
        "source_id": f"source-{index}",
        "source_class": source_class,
        "revision": f"{index + 1:040x}",
        "license": "ODC-BY-1.0",
        "domain": domain,
        "source_manifest_sha256": f"{index + 11:064x}",
        "selection_policy_sha256": f"{index + 21:064x}",
        "decontamination_receipt_sha256": f"{index + 31:064x}",
        "minimum_phase": minimum_phase,
        "rehearsal_required": rehearsal_required,
        "planned_tokens": planned_tokens,
    }


def _payload() -> dict:
    sources = [
        _source(
            0,
            source_class="educational_web",
            domain="english",
            planned_tokens=64,
        ),
        _source(1, source_class="code", domain="code", planned_tokens=32),
        _source(2, source_class="mathematics", domain="math", planned_tokens=24),
        _source(
            3,
            source_class="science_technical",
            domain="science",
            planned_tokens=32,
        ),
        _source(
            4,
            source_class="science_technical",
            domain="technical",
            planned_tokens=8,
            minimum_phase="reasoning",
            rehearsal_required=False,
        ),
    ]
    allocations = (
        (16, 8, 8, 8, 0),
        (16, 8, 8, 8, 0),
        (16, 8, 4, 8, 4),
        (16, 8, 4, 8, 4),
    )
    phases = []
    cumulative = 0
    for index, (phase, allocation) in enumerate(
        zip(
            ("grounding", "integration", "reasoning", "specialization"),
            allocations,
            strict=True,
        )
    ):
        tokens = sum(allocation)
        cumulative += tokens
        phases.append(
            {
                "phase": phase,
                "index": index,
                "tokens": tokens,
                "cumulative_tokens": cumulative,
                "by_source": {
                    source["source_id"]: value
                    for source, value in zip(sources, allocation, strict=True)
                },
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "sequence_length": 8,
        "sequences_per_update": 1,
        "total_tokens": 160,
        "sources": sources,
        "phases": phases,
        "controls": {
            "same_sequence_multiset_order_control": True,
            "tokenizer_factor_isolated": True,
            "architecture_factor_isolated": True,
            "terminal_benchmarks_used_for_tuning": False,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _resign(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


def test_validates_exact_multisource_progressive_plan(tmp_path: Path) -> None:
    payload = _payload()
    assert validate_payload(payload) == payload
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    assert validate_plan(path) == payload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["sources"][4].update(domain="science"),
        lambda value: value["sources"][0].update(planned_tokens=63),
        lambda value: value["phases"][0]["by_source"].update({"source-4": 1}),
        lambda value: value["phases"][1]["by_source"].update({"source-0": 0}),
        lambda value: value["phases"][2].update(cumulative_tokens=119),
        lambda value: value["controls"].update(
            terminal_benchmarks_used_for_tuning=True
        ),
        lambda value: value.update(four_b_training_authorized=True),
    ],
)
def test_rejects_resigned_structural_or_authorization_drift(mutate) -> None:
    payload = deepcopy(_payload())
    mutate(payload)
    _resign(payload)
    with pytest.raises(DataMixturePlanError):
        validate_payload(payload)


def test_rejects_hash_tamper_and_unsafe_path(tmp_path: Path) -> None:
    payload = _payload()
    payload["total_tokens"] += 8
    with pytest.raises(DataMixturePlanError, match="receipt hash"):
        validate_payload(payload)

    target = tmp_path / "target.json"
    target.write_text(json.dumps(_payload()))
    link = tmp_path / "plan.json"
    link.symlink_to(target)
    with pytest.raises(DataMixturePlanError, match="unsafe"):
        validate_plan(link)
