import copy

import pytest

from sai.data.finemath_full_census import (
    AGGREGATE_SCHEMA,
    EXPECTED_SHARDS,
    REPOSITORY,
    REVISION,
    SHARD_SCHEMA,
    SUBSET,
)
from sai.data.finemath_full_census_publication import (
    FineMathFullCensusPublicationError,
    summarize_publication,
)
from sai.data.token_stream import canonical_sha256


def _evidence() -> tuple[dict, list[dict]]:
    shards = []
    for index in range(EXPECTED_SHARDS):
        receipt = {
            "schema": SHARD_SCHEMA,
            "status": "complete_source_safe_mechanical_census_shard",
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "shard_index": index,
            },
            "source_file_sha256_verified": True,
            "full_shard_scanned": True,
            "source_text_persisted": False,
            "training_ready": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        shards.append(receipt)
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_source_safe_full_mechanical_census",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "subset": SUBSET,
        },
        "summary": {"rows": 100, "text_utf8_bytes": 1_000, "token_count": 250},
        "shards": {
            "ordered_receipts_sha256": canonical_sha256(
                [row["receipt_sha256"] for row in shards]
            )
        },
        "exact_content_multiplicity": {"rows": 100},
        "normalized_content_multiplicity": {"rows": 100},
        "all_source_file_sha256_values_verified": True,
        "all_source_rows_scanned": True,
        "global_exact_duplicate_census_complete": True,
        "source_text_persisted": False,
        "training_ready": False,
        "receipt_sha256": "a" * 64,
    }
    return aggregate, shards


def test_publication_covers_all_ordered_shards() -> None:
    result = summarize_publication(*_evidence())
    assert result["rows"] == 100
    assert result["ordered_shard_receipts_sha256"]


def test_publication_rejects_reordered_shard() -> None:
    aggregate, shards = _evidence()
    reordered = copy.deepcopy(shards)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(FineMathFullCensusPublicationError, match="shard"):
        summarize_publication(aggregate, reordered)


def test_publication_rejects_missing_shard() -> None:
    aggregate, shards = _evidence()
    with pytest.raises(FineMathFullCensusPublicationError, match="aggregate"):
        summarize_publication(aggregate, shards[:-1])
