from __future__ import annotations

from sai.data.one_b_parent_partition import PARTITION_COUNT, _bucket


def test_parent_bucket_is_stable_and_bounded() -> None:
    assert _bucket("data/example.parquet") == _bucket("data/example.parquet")
    assert 0 <= _bucket("data/example.parquet") < PARTITION_COUNT
    assert _bucket("data/example.parquet") != _bucket("data/other.parquet")
