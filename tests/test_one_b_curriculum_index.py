from __future__ import annotations

import pytest

from sai.data.one_b_curriculum_index import (
    _book_band,
    _descriptor,
    _pleias_band,
    _priority,
    _split,
    _write_index,
)


def test_book_topics_use_source_informed_tiers_not_length_as_difficulty() -> None:
    assert _book_band("EDUCATION", 100_000, "0" * 64)[0] == "foundation"
    expert_identity = f"{99:016x}" + "0" * 48
    assert _book_band("SCIENCE", 100_000, expert_identity)[0] == "expert"
    assert _book_band("SCIENCE", 300_000, expert_identity)[0] == "expert"


def test_pleias_collection_prior_and_stable_within_source_distribution() -> None:
    assert _pleias_band("Wikipedia", 500, 650, "0" * 64)[0] == "foundation"
    expert_identity = f"{99:016x}" + "0" * 48
    advanced_identity = f"{60:016x}" + "0" * 48
    assert _pleias_band("arXiv", 2_000, 3_000, expert_identity)[0] == "expert"
    assert _pleias_band("StackExchange", 500, 700, advanced_identity)[0] == "advanced"
    assert _pleias_band("Gutenberg", 2_000, 4_000, expert_identity)[0] == "intermediate"


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


def test_empty_index_has_explicit_zero_accounting(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    result = _write_index(iter(()), tmp_path / "empty")
    assert result["output"]["rows"] == 0
    assert result["counts"]["rows"] == 0
    assert result["counts"]["text_utf8_bytes"] == 0
    assert result["counts"]["band::expert::rows"] == 0
    assert result["counts"]["split::development::rows"] == 0
