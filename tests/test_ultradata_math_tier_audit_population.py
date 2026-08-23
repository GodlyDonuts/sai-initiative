import hashlib

import pytest

from sai.data.reservoir_audit_aggregate import load_population
from sai.data.ultradata_math_tier_audit_population import (
    BATCH_ROWS,
    EXPECTED_BATCHES,
    EXPECTED_ROWS,
    REVISION,
    TIER_SPECS,
    UltraDataMathTierAuditError,
    build_batch_plan,
    build_population,
)


def _fake_batch(plan: dict) -> dict:
    rows = []
    for index in range(plan["length"]):
        row_index = plan["offset"] + index
        content = f"A rigorous mathematical explanation for row {row_index}. " * 8
        row = {"content": content}
        if plan["config"].endswith("preview"):
            row["quality_label"] = 4
        else:
            row["uid"] = f"uid-{plan['config']}-{row_index}"
        rows.append({"row_idx": row_index, "row": row})
    response = repr(rows).encode()
    return {
        "x_revision": REVISION,
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "payload": {
            "rows": rows,
            "num_rows_total": plan["expected_rows"],
        },
        "request_url": "https://datasets-server.huggingface.co/rows?test=1",
    }


def test_batch_plan_is_exact_deterministic_and_non_overlapping() -> None:
    first = build_batch_plan()
    assert first == build_batch_plan()
    assert len(first) == EXPECTED_BATCHES == 20
    for spec in TIER_SPECS:
        selected = [row for row in first if row["config"] == spec.config]
        assert len(selected) == 4
        intervals = [(row["offset"], row["offset"] + BATCH_ROWS) for row in selected]
        assert all(0 <= start < end <= spec.expected_rows for start, end in intervals)
        assert all(
            left[1] <= right[0] or left[0] >= right[1]
            for index, left in enumerate(intervals)
            for right in intervals[index + 1 :]
        )


def test_population_is_compatible_with_shared_audit_pipeline(tmp_path) -> None:
    root = tmp_path / "population"
    receipt = build_population(
        root,
        revision_resolver=lambda: REVISION,
        batch_fetcher=_fake_batch,
    )
    candidates, lineage, replay = load_population(root)
    assert receipt == replay
    assert len(candidates) == len(lineage) == EXPECTED_ROWS == 160
    assert receipt["dataset_server_batches"] == 20
    assert set(receipt["by_stratum"].values()) == {32}
    assert receipt["training_ready"] is False


def test_population_rejects_dataset_server_revision_mismatch(tmp_path) -> None:
    def wrong_revision(plan: dict) -> dict:
        result = _fake_batch(plan)
        result["x_revision"] = "0" * 40
        return result

    with pytest.raises(UltraDataMathTierAuditError, match="response geometry"):
        build_population(
            tmp_path / "population",
            revision_resolver=lambda: REVISION,
            batch_fetcher=wrong_revision,
        )
