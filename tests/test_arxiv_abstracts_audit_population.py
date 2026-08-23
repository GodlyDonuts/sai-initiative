import hashlib

import pytest

from sai.data.arxiv_abstracts_audit_population import (
    EXPECTED_ROWS,
    FETCH_ROWS_PER_STRATUM,
    REVISION,
    SOURCE_ROWS,
    TEMPORAL_STRATA,
    ArxivAbstractsAuditError,
    build_batch_plan,
    build_population,
)
from sai.data.reservoir_audit_aggregate import load_population


def _fake_batch(plan: dict) -> dict:
    rows = []
    for index in range(plan["length"]):
        row_index = plan["offset"] + index
        rows.append(
            {
                "row_idx": row_index,
                "row": {
                    "id": f"arxiv-{row_index}",
                    "text": (
                        f"A rigorous scientific abstract for row {row_index} " * 12
                    ),
                    "metadata": {
                        "authors": "Ada Example and Lin Example",
                        "full_text_license": "upstream terms",
                        "license": "Creative Commons Zero - Public Domain - test",
                        "provenance": f"parent.json.gz:{row_index + 1}",
                        "url": f"https://arxiv.org/abs/{row_index}",
                    },
                },
            }
        )
    response = repr(rows).encode()
    return {
        "x_revision": REVISION,
        "response_sha256": hashlib.sha256(response).hexdigest(),
        "request_url": "https://datasets-server.huggingface.co/rows?test=1",
        "payload": {"rows": rows, "num_rows_total": SOURCE_ROWS},
    }


def _no_exclusions(roots):
    return frozenset(), [
        {
            "root_name": str(roots[0]),
            "receipt_sha256": "a" * 64,
            "population_sha256": "b" * 64,
            "lineage_sha256": "c" * 64,
        }
    ]


def test_temporal_plan_is_exact_and_non_overlapping() -> None:
    plan = build_batch_plan()
    assert plan == build_batch_plan()
    assert len(plan) == TEMPORAL_STRATA == 32
    for row in plan:
        assert row["stratum_start"] <= row["offset"]
        assert row["offset"] + FETCH_ROWS_PER_STRATUM <= row["stratum_end"]
    assert all(
        left["stratum_end"] <= right["stratum_start"]
        for left, right in zip(plan, plan[1:], strict=False)
    )


def test_population_is_source_disjoint_and_shared_pipeline_compatible(
    tmp_path,
) -> None:
    root = tmp_path / "population"
    receipt = build_population(
        root,
        [tmp_path / "audit"],
        revision_resolver=lambda: REVISION,
        batch_fetcher=_fake_batch,
        exclusion_loader=_no_exclusions,
    )
    candidates, lineage, replay = load_population(root)
    assert receipt == replay
    assert len(candidates) == len(lineage) == EXPECTED_ROWS == 1024
    assert receipt["dataset_server_batches"] == 32
    assert set(receipt["by_stratum"].values()) == {32}
    assert receipt["source_disjoint_from_audit_populations"] is True
    assert all(
        row["declared_license"].startswith("Creative Commons Zero")
        for row in lineage
    )
    assert receipt["training_ready"] is False


def test_population_rejects_revision_mismatch(tmp_path) -> None:
    with pytest.raises(ArxivAbstractsAuditError, match="output boundary"):
        build_population(
            tmp_path / "population",
            [tmp_path / "audit"],
            revision_resolver=lambda: "0" * 40,
            batch_fetcher=_fake_batch,
            exclusion_loader=_no_exclusions,
        )
