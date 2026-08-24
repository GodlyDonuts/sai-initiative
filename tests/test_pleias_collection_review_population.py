import pytest

from sai.data.pleias_collection_review_population import (
    PleiasCollectionReviewError,
    select_collection_rows,
)


def _row(identity, collection):
    return (
        {"candidate_identity_sha256": identity},
        {"locator": {"collection": collection}},
    )


def test_selects_lowest_source_disjoint_identities_per_collection():
    rows = [
        _row("3" * 64, "A"),
        _row("1" * 64, "A"),
        _row("2" * 64, "A"),
        _row("5" * 64, "B"),
        _row("4" * 64, "B"),
    ]
    selected = select_collection_rows(
        [row[0] for row in rows],
        [row[1] for row in rows],
        ["A", "B"],
        2,
        frozenset({"1" * 64}),
    )
    assert [row[0]["candidate_identity_sha256"] for row in selected] == [
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
    ]


def test_rejects_underfilled_collection():
    candidate, lineage = _row("1" * 64, "A")
    with pytest.raises(PleiasCollectionReviewError, match="underfilled"):
        select_collection_rows(
            [candidate], [lineage], ["A"], 2, frozenset()
        )
