import pytest

from sai.data.frontier_source_reservoir import (
    SOURCE_SPECS,
    FrontierSourceReservoirError,
    select_frontier_sources,
)


def _inventories() -> dict[str, list[dict]]:
    inventories = {}
    for index, spec in enumerate(SOURCE_SPECS, start=1):
        prefix = spec.prefixes[0]
        inventories[spec.source_id] = [
            {
                "path": f"{prefix}part-{index:03d}{spec.suffix}",
                "bytes": 2 * 1024**4,
                "sha256": f"{index:064x}",
            }
        ]
    return inventories


def test_frontier_source_selection_is_exact_large_and_non_admitting() -> None:
    rows = select_frontier_sources(_inventories())
    assert len(rows) == len(SOURCE_SPECS)
    assert sum(row["physical_bytes"] for row in rows) == len(rows) * 2 * 1024**4
    assert all(row["physical_bytes_are_text_payload_bytes"] is False for row in rows)
    assert all(row["source_candidate_is_training_ready"] is False for row in rows)
    assert [row["ordinal"] for row in rows] == list(range(len(rows)))


def test_frontier_source_selection_rejects_unlisted_path() -> None:
    inventories = _inventories()
    first = SOURCE_SPECS[0]
    inventories[first.source_id][0]["path"] = f"unlisted/file{first.suffix}"
    with pytest.raises(FrontierSourceReservoirError, match="identity"):
        select_frontier_sources(inventories)
