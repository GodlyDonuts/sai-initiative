import copy

import pytest

from sai.data.quality_core_envelope import (
    QualityCoreEnvelopeError,
    build_envelope_payload,
)
from sai.data.token_stream import canonical_sha256


def _removal():
    payload = {
        "schema": "sai-hf-source-removal-receipt-v1",
        "status": "complete_verified_recoverable_source_prefix_removal",
        "prefix": "sources/remove",
        "removed_objects": 1,
        "removed_bytes": 100,
        "post_removal_source_tree": {"data_files": 3, "data_bytes": 900},
        "repository": "org/data",
        "verified_current_revision": "b" * 40,
        "recoverable_from_repository_history": True,
        "remaining_prefix_files": 0,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _rows():
    return [
        {
            "destination_path": "sources/remove/data/a",
            "source_id": "removed",
            "bytes": 100,
            "raw_source_is_training_ready": False,
        },
        {
            "destination_path": "sources/bulk/a",
            "source_id": "bulk",
            "bytes": 600,
            "raw_source_is_training_ready": False,
        },
        {
            "destination_path": "sources/special/a",
            "source_id": "special",
            "bytes": 200,
            "raw_source_is_training_ready": False,
        },
        {
            "destination_path": "sources/special/b",
            "source_id": "special",
            "bytes": 100,
            "raw_source_is_training_ready": False,
        },
    ]


def test_builds_exact_ceiling_without_admission():
    result = build_envelope_payload(_rows(), _removal(), 500, "bulk")
    assert result["post_removal_lake"]["candidate_bytes"] == 900
    assert result["nonbulk_sources"]["raw_candidate_bytes"] == 300
    assert (
        result["bulk_source"][
            "provisional_maximum_bytes_if_all_nonbulk_candidates_survive"
        ]
        == 200
    )
    assert (
        result["bulk_source"][
            "provisional_excess_bytes_if_all_nonbulk_candidates_survive"
        ]
        == 400
    )
    assert result["target"]["is_padding_floor"] is False
    assert result["automatic_training_admission"] is False
    assert result["training_ready"] is False


def test_rejects_accounting_drift():
    removal = copy.deepcopy(_removal())
    removal["removed_bytes"] = 99
    with pytest.raises(QualityCoreEnvelopeError, match="accounting"):
        build_envelope_payload(_rows(), removal, 500, "bulk")


def test_rejects_duplicate_destination_path():
    rows = _rows()
    rows[-1]["destination_path"] = rows[-2]["destination_path"]
    with pytest.raises(QualityCoreEnvelopeError, match="identity"):
        build_envelope_payload(rows, _removal(), 500, "bulk")
