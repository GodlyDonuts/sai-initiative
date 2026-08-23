from __future__ import annotations

import pytest

from sai.data.opencoder_code_audit_population import (
    OpenCoderCodeAuditPopulationError,
    _host,
    select_final_rows,
    select_screen_rows,
)


def _row(index: int, host: str, content: int, key: int) -> dict:
    return {
        "row_index": index,
        "host": host,
        "content_sha256": f"{content:064x}",
        "selection_key": f"{key:064x}",
    }


def test_screen_selection_deduplicates_content_and_caps_hosts() -> None:
    rows = [
        _row(0, "a.example", 1, 9),
        _row(1, "b.example", 1, 1),
        _row(2, "b.example", 2, 2),
        _row(3, "b.example", 3, 3),
        _row(4, "c.example", 4, 4),
    ]
    selected = select_screen_rows(rows, screen_rows=3, maximum_rows_per_host=2)
    assert [row["row_index"] for row in selected] == [1, 2, 4]
    assert len({row["content_sha256"] for row in selected}) == 3


def test_final_selection_uses_tighter_host_ceiling() -> None:
    rows = [
        _row(0, "a.example", 1, 1),
        _row(1, "a.example", 2, 2),
        _row(2, "b.example", 3, 3),
        _row(3, "c.example", 4, 4),
    ]
    selected = select_final_rows(rows, target_rows=3, maximum_rows_per_host=1)
    assert [row["host"] for row in selected] == [
        "a.example",
        "b.example",
        "c.example",
    ]


def test_underfilled_host_diversity_fails_closed() -> None:
    rows = [_row(index, "one.example", index + 1, index + 1) for index in range(4)]
    with pytest.raises(OpenCoderCodeAuditPopulationError, match="underfilled"):
        select_final_rows(rows, target_rows=2, maximum_rows_per_host=1)


def test_host_normalization_rejects_non_host_urls() -> None:
    assert _host("HTTPS://Docs.Example.COM/path") == "docs.example.com"
    assert _host("not a url") is None
    assert _host("") is None
