import copy

import pytest

from sai.data.source_reservoir import (
    SOURCE_SPECS,
    SourceReservoirError,
    select_reservoir,
)


def _inventories() -> dict[str, list[dict]]:
    result = {}
    for index, spec in enumerate(SOURCE_SPECS):
        result[spec.source_id] = [
            {
                "path": f"data/{index:02d}/part-00000{spec.suffix}",
                "bytes": 100 + index,
                "sha256": f"{index + 1:064x}",
            }
        ]
    filler = next(spec for spec in SOURCE_SPECS if spec.fill_source)
    result[filler.source_id].append(
        {
            "path": f"data/99/part-00001{filler.suffix}",
            "bytes": 1_000,
            "sha256": f"{99:064x}",
        }
    )
    return result


def test_reservoir_preserves_specialists_then_adds_minimum_fill() -> None:
    inventories = _inventories()
    specialist_bytes = sum(
        rows[0]["bytes"]
        for source, rows in inventories.items()
        if source != "fineweb_edu_fill"
    )
    rows = select_reservoir(inventories, target_bytes=specialist_bytes + 50)
    assert rows[-1]["source_id"] == "fineweb_edu_fill"
    assert rows[-1]["ordinal"] == len(rows) - 1
    assert sum(row["bytes"] for row in rows) >= specialist_bytes + 50
    assert all(row["raw_source_is_training_ready"] is False for row in rows)


def test_reservoir_rejects_missing_source_or_tampered_hash() -> None:
    inventories = _inventories()
    missing = copy.deepcopy(inventories)
    missing.pop("finemath")
    with pytest.raises(SourceReservoirError, match="inputs"):
        select_reservoir(missing, 100)
    tampered = copy.deepcopy(inventories)
    tampered["finemath"][0]["sha256"] = "not-a-hash"
    with pytest.raises(SourceReservoirError, match="identity"):
        select_reservoir(tampered, 100)


def test_reservoir_fails_when_fill_cannot_reach_target() -> None:
    with pytest.raises(SourceReservoirError, match="cannot reach"):
        select_reservoir(_inventories(), target_bytes=10_000_000)
