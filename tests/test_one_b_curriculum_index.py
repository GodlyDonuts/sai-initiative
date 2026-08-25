from __future__ import annotations

from sai.data.one_b_curriculum_index import (
    _book_band,
    _descriptor,
    _pleias_band,
    _priority,
    _split,
)


def test_book_topics_and_length_move_only_one_band() -> None:
    assert _book_band("EDUCATION", 100_000)[0] == "foundation"
    assert _book_band("SCIENCE", 100_000)[0] == "advanced"
    assert _book_band("SCIENCE", 300_000)[0] == "expert"
    assert _book_band("SCIENCE", 10_000)[0] == "intermediate"


def test_pleias_collection_prior_and_surface_adjustment() -> None:
    assert _pleias_band("Wikipedia", 500, 650)[0] == "foundation"
    assert _pleias_band("arXiv", 2_000, 3_000)[0] == "expert"
    assert _pleias_band("StackExchange", 500, 700)[0] == "advanced"
    assert _pleias_band("Gutenberg", 2_000, 4_000)[0] == "intermediate"


def test_split_and_priority_are_stable_and_component_bound() -> None:
    identity = "0" * 64
    assert _split(identity) == "development"
    assert _split(identity, bulk=False) == "train"
    assert _priority("books", identity) == _priority("books", identity)
    assert _priority("books", identity) != _priority("pleias", identity)


def test_optional_descriptor_distinguishes_an_empty_shard() -> None:
    admission = {"outputs": {"descriptors": [{"shard_index": 1, "rows": 2}]}}
    assert _descriptor(admission, 1) == {"shard_index": 1, "rows": 2}
    assert _descriptor(admission, 7, allow_empty=True) is None
