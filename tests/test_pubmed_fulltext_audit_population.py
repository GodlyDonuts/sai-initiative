import hashlib

from sai.data.pubmed_fulltext_audit_population import (
    CC0_DECLARATION,
    CC_BY_DECLARATION,
    FETCH_ROWS_PER_STRATUM,
    INDEX_STRATA,
    REVISION,
    SOURCE_ROWS,
    WINDOWS_PER_STRATUM,
    _select_stratum,
    _validate_batch,
    build_batch_plan,
)


def _result(plan: dict) -> dict:
    rows = []
    for local_index in range(FETCH_ROWS_PER_STRATUM):
        row_index = plan["offset"] + local_index
        rows.append(
            {
                "row_idx": row_index,
                "row": {
                    "id": f"pmc-{row_index}",
                    "text": f"A rigorous biomedical source row {row_index}. " * 12,
                    "metadata": {
                        "authors": [{"first": "Ada", "last": "Lovelace"}],
                        "created": "2010-01-01",
                        "journal": "Test Journal",
                        "license": (
                            CC_BY_DECLARATION
                            if local_index % 2 == 0
                            else CC0_DECLARATION
                        ),
                        "provenance": f"licensed_pubmed-0000.json.gz:{row_index + 1}",
                        "url": f"https://example.invalid/{row_index}",
                    },
                    "source": "pubmed",
                },
            }
        )
    return {
        "x_revision": REVISION,
        "response_sha256": hashlib.sha256(b"response").hexdigest(),
        "request_url": "https://datasets-server.huggingface.co/rows?test=1",
        "payload": {"rows": rows, "num_rows_total": SOURCE_ROWS},
    }


def test_index_stratified_plan_has_exact_nonoverlapping_geometry() -> None:
    plan = build_batch_plan()
    assert len(plan) == INDEX_STRATA * WINDOWS_PER_STRATUM
    assert plan[0]["stratum_start"] == 0
    assert plan[-1]["stratum_end"] == SOURCE_ROWS
    assert all(
        row["window_start"] <= row["offset"]
        and row["offset"] + FETCH_ROWS_PER_STRATUM <= row["window_end"]
        for row in plan
    )
    for stratum_index in range(INDEX_STRATA):
        windows = [row for row in plan if row["stratum_index"] == stratum_index]
        assert len(windows) == WINDOWS_PER_STRATUM
        assert windows[0]["window_start"] == windows[0]["stratum_start"]
        assert windows[-1]["window_end"] == windows[-1]["stratum_end"]
        assert all(
            left["window_end"] == right["window_start"]
            for left, right in zip(windows, windows[1:], strict=False)
        )


def test_batch_selection_accepts_only_exact_recognized_license_contract() -> None:
    plan = build_batch_plan()[0]
    eligible, receipt = _validate_batch(plan, _result(plan), frozenset())
    selected = _select_stratum(eligible)
    assert len(selected) == 32
    assert {row["canonical_license"] for row in selected} <= {
        "CC-BY-4.0",
        "CC0-1.0",
    }
    assert receipt["fetched_rows"] == 64
    assert receipt["eligible_rows"] == 64
    assert receipt["source_text_persisted"] is False


def test_fixed_multiwindow_pool_recovers_from_sparse_first_window() -> None:
    plans = build_batch_plan()[:WINDOWS_PER_STRATUM]
    eligible = []
    for window_index, plan in enumerate(plans):
        result = _result(plan)
        if window_index == 0:
            for item in result["payload"]["rows"]:
                item["row"]["text"] = ""
        batch_eligible, _receipt = _validate_batch(plan, result, frozenset())
        eligible.extend(batch_eligible)
    selected = _select_stratum(eligible)
    assert len(selected) == 32
    assert all(row["row_index"] >= plans[1]["window_start"] for row in selected)
