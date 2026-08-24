import json

import pytest

from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as CROSS_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA as PLEIAS_SCHEMA,
)
from sai.data.token_stream import canonical_sha256
from sai.data.virtual_corpus_custody_manifest import (
    VirtualCorpusCustodyManifestError,
    build_manifest,
)
from sai.data.virtual_foundation_corpus_ledger import SCHEMA as LEDGER_SCHEMA


def _signed(path, payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return payload


def _workspace(tmp_path):
    source = tmp_path / "manifest.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema": "sai-hf-materialized-source-file-v1",
                "source_repository": "PleIAs/common_corpus",
                "destination_path": "sources/pleias/part-0.parquet",
                "bytes": 1_000,
                "sha256": "a" * 64,
                "raw_source_is_training_ready": False,
            }
        )
        + "\n"
    )
    cross = _signed(
        tmp_path / "cross.json",
        {
            "schema": CROSS_SCHEMA,
            "cross_source_subdocument_decision_complete": True,
            "decision_contains_source_text": False,
            "training_ready": False,
        },
    )
    books = _signed(
        tmp_path / "books.json",
        {
            "schema": BOOK_SCHEMA,
            "cross_source_subdocument_deduplication_complete": True,
            "training_ready": False,
        },
    )
    pleias = _signed(
        tmp_path / "pleias.json",
        {
            "schema": PLEIAS_SCHEMA,
            "source": {
                "cross_decision_aggregate_receipt_sha256": cross["receipt_sha256"]
            },
            "cross_source_subdocument_deduplication_complete": True,
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    _signed(
        tmp_path / "ledger.json",
        {
            "schema": LEDGER_SCHEMA,
            "components": [
                {
                    "component": "institutional_books",
                    "aggregate_receipt_sha256": books["receipt_sha256"],
                },
                {
                    "component": "pleias_common_corpus",
                    "aggregate_receipt_sha256": pleias["receipt_sha256"],
                },
            ],
            "totals": {"post_rewrite_text_utf8_bytes": 900},
            "byte_ceiling_respected": True,
            "pleias_virtual_reconstruction_complete": True,
            "pleias_payload_materialization_complete": False,
            "training_ready": False,
        },
    )
    return (
        source,
        tmp_path / "books.json",
        tmp_path / "pleias.json",
        tmp_path / "cross.json",
        tmp_path / "ledger.json",
    )


def test_custody_manifest_binds_sources_components_and_two_copies(tmp_path):
    source, books, pleias, cross, ledger = _workspace(tmp_path)
    output = tmp_path / "custody" / "receipt.json"
    durable = tmp_path / "evidence" / "receipt.json"
    result = build_manifest(
        source,
        books,
        pleias,
        cross,
        ledger,
        output,
        durable,
        runtime_commit="f" * 40,
    )
    assert result["all_irreplaceable_receipts_hash_manifested"] is True
    assert result["durable_evidence_copy_complete"] is True
    assert output.read_bytes() == durable.read_bytes()
    assert result["reconstruction_contract"]["final_content_sha256_bound_per_row"]
    assert (
        result["reconstruction_contract"]["source_text_persisted_in_manifest"] is False
    )
    assert result["training_ready"] is False


def test_custody_manifest_rejects_component_receipt_substitution(tmp_path):
    source, books, pleias, cross, ledger = _workspace(tmp_path)
    payload = json.loads(ledger.read_text())
    payload.pop("receipt_sha256")
    payload["components"][1]["aggregate_receipt_sha256"] = "0" * 64
    _signed(ledger, payload)
    with pytest.raises(VirtualCorpusCustodyManifestError, match="component"):
        build_manifest(
            source,
            books,
            pleias,
            cross,
            ledger,
            tmp_path / "receipt.json",
            tmp_path / "durable.json",
            runtime_commit="f" * 40,
        )
