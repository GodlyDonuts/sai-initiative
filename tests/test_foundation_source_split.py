from sai.data.foundation_source_split import (
    POLICY_SHA256,
    assign_source_group,
)


def test_same_source_parent_cannot_cross_train_and_development():
    identifiers = {
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": "data/0001.parquet",
        "source_parent_sha256": "b" * 64,
    }
    first = assign_source_group("pleias_common_corpus", identifiers)
    second = assign_source_group("pleias_common_corpus", identifiers)
    assert first == second
    assert first[1] in {"train", "development"}
    assert 0 <= first[2] < 1_000
    assert len(POLICY_SHA256) == 64


def test_book_editions_group_by_connected_work_family():
    first = assign_source_group("institutional_books", {"work_family_sha256": "a" * 64})
    second = assign_source_group(
        "institutional_books", {"work_family_sha256": "a" * 64}
    )
    assert first == second
    assert len(first[0]) == 64
