import hashlib
from pathlib import Path

from sai.data.finemath_full_census import (
    candidate_profiles,
    summarize_digest_files,
)


def _write_hashes(path: Path, values: list[str]) -> None:
    path.write_bytes(
        b"".join(sorted(hashlib.sha256(value.encode()).digest() for value in values))
    )


def test_digest_merge_counts_cross_shard_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    _write_hashes(first, ["a", "b", "b"])
    _write_hashes(second, ["b", "c", "c"])
    result = summarize_digest_files([first, second])
    assert result == {
        "rows": 6,
        "unique_hashes": 3,
        "duplicate_rows_after_keep_first": 3,
        "duplicate_groups": 2,
        "maximum_multiplicity": 3,
    }


def test_profiles_are_nested_measurements() -> None:
    assert candidate_profiles(
        language="en",
        language_score=0.95,
        int_score=5,
        token_count=512,
        text_bytes=2048,
    ) == (
        "broad_mechanical_profile",
        "core_mechanical_profile",
        "elite_mechanical_profile",
    )


def test_profiles_reject_short_or_non_english_rows() -> None:
    assert not candidate_profiles(
        language="fr",
        language_score=0.99,
        int_score=5,
        token_count=512,
        text_bytes=2048,
    )
    assert not candidate_profiles(
        language="en",
        language_score=0.99,
        int_score=5,
        token_count=32,
        text_bytes=2048,
    )
