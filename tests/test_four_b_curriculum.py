from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sai.data.four_b_curriculum import (
    FourBCurriculumError,
    validate_file,
    validate_payload,
)
from sai.data.token_stream import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "docs" / "SAI_4B_120B_CURRICULUM_CANDIDATE.json"


def _payload() -> dict:
    return json.loads(CANDIDATE.read_text())


def _resign(payload: dict) -> None:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )


def test_frozen_candidate_replays_exactly() -> None:
    payload = validate_file(CANDIDATE)
    assert payload["target"]["total_tokens"] == 120_000_000_000
    assert sum(row["sequences"] for row in payload["phases"]) == 58_593_750
    assert payload["total_by_pool"]["foundation"]["tokens"] == 68_100_001_792


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(data_ready=True),
        lambda value: value["target"].update(total_tokens=120_000_000_001),
        lambda value: value["phases"][0]["by_pool_sequences"].update(reasoning=1),
        lambda value: value["phases"][3]["by_pool_sequences"].update(foundation=0),
        lambda value: value["total_by_pool"]["code"].update(sequences=1),
        lambda value: value["source_inventories"][0].update(role="late_curriculum"),
        lambda value: value["canonical_source_pools"][4].update(
            minimum_phase="grounding"
        ),
        lambda value: value["admission_policy"].update(
            source_shortfall_policy="silently_reweight"
        ),
    ],
)
def test_rejects_resigned_scientific_drift(mutate) -> None:
    payload = deepcopy(_payload())
    mutate(payload)
    _resign(payload)
    with pytest.raises(FourBCurriculumError):
        validate_payload(payload)


def test_rejects_hash_tamper_and_unsafe_path(tmp_path: Path) -> None:
    payload = _payload()
    payload["target"]["total_sequences"] += 1
    with pytest.raises(FourBCurriculumError, match="receipt hash"):
        validate_payload(payload)

    target = tmp_path / "target.json"
    target.write_text(CANDIDATE.read_text())
    link = tmp_path / "candidate.json"
    link.symlink_to(target)
    with pytest.raises(FourBCurriculumError, match="unsafe"):
        validate_file(link)
