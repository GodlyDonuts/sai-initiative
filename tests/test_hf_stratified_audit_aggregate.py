import json
from pathlib import Path

import pytest

from sai.data.hf_shard_audit import SCHEMA as SHARD_SCHEMA
from sai.data.hf_stratified_audit_aggregate import (
    HFStratifiedAuditAggregateError,
    aggregate_audits,
)
from sai.data.hf_stratified_audit_plan import PLAN_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file


def _seal(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _plan(path: Path) -> dict:
    selection = {
        "stratum": "science",
        "group": {"topic": "math"},
        "component": "component",
        "path": "data/component/shard.jsonl.zst",
        "compressed_bytes": 10,
        "compressed_sha256": "1" * 64,
        "selection_rank_sha256": "2" * 64,
    }
    payload = _seal(
        {
            "schema": PLAN_SCHEMA,
            "status": "prospective_stratified_members_selected_no_download",
            "training_authorized": False,
            "source_admitted": False,
            "content_downloaded": False,
            "dataset_inventory": {
                "file_sha256": "3" * 64,
                "receipt_sha256": "4" * 64,
                "dataset": "owner/data",
                "revision": "5" * 40,
            },
            "specification": {
                "file_sha256": "6" * 64,
                "spec_sha256": "7" * 64,
                "selection_seed": "seed",
            },
            "selected_shards": 1,
            "selected_compressed_bytes": 10,
            "stratum_summaries": [],
            "selections_sha256": canonical_sha256([selection]),
            "selections": [selection],
            "checks": {},
        }
    )
    path.write_text(json.dumps(payload))
    return payload


def _audit(path: Path, plan: dict) -> dict:
    payload = _seal(
        {
            "schema": SHARD_SCHEMA,
            "status": "diagnostic_complete_source_not_admitted",
            "training_authorized": False,
            "source_admitted": False,
            "rows_selected_for_training": 0,
            "dataset_inventory": plan["dataset_inventory"],
            "member": {
                "path": "data/component/shard.jsonl.zst",
                "compressed_bytes": 10,
                "compressed_sha256": "1" * 64,
            },
            "population": {
                "rows": 5,
                "unique_document_ids": 3,
                "unique_texts": 2,
                "duplicate_document_id_rows": 2,
                "duplicate_text_rows": 3,
                "duplicate_document_id_fraction": 0.4,
                "duplicate_text_fraction": 0.6,
                "distinct_ids_sharing_text": 1,
                "empty_text_rows": 1,
                "max_document_id_multiplicity": 2,
                "document_id_multiplicity_histogram": {"1": 1, "2": 2},
                "text_length_characters": {
                    "minimum": 0,
                    "median": 2,
                    "p95": 4,
                    "maximum": 4,
                },
                "ordered_identity_sha256": "8" * 64,
            },
            "metadata": {
                "distinct_sources": 1,
                "source_counts_sha256": "9" * 64,
                "top_source_counts": [{"source": "x", "rows": 5}],
                "license_type_counts": {"permissive": 2, "no_license": 3},
                "integer_score_counts": {"3": 5},
                "metadata_key_row_counts": {"license_type": 5},
            },
            "checks": {},
        }
    )
    path.write_text(json.dumps(payload))
    return payload


def test_aggregates_complete_population(tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    plan = _plan(plan_path)
    audit_path = tmp_path / "audit.json"
    audit = _audit(audit_path, plan)
    result = aggregate_audits(plan_path, [audit_path])
    assert result["status"] == "complete_source_not_admitted"
    assert result["totals"]["physical_rows"] == 5
    assert result["totals"]["within_shard_duplicate_document_id_rows"] == 2
    assert result["totals"]["within_shard_duplicate_document_id_fraction"] == 0.4
    assert result["totals"]["license_type_counts"] == {
        "no_license": 3,
        "permissive": 2,
    }
    assert result["audit_files"][0]["file_sha256"] == sha256_file(audit_path)
    assert result["audit_files"][0]["receipt_sha256"] == audit["receipt_sha256"]
    assert result["checks"]["cross_shard_duplicate_identity_not_measured"] is True
    assert result["training_authorized"] is False


@pytest.mark.parametrize("mutation", ["missing", "tamper", "duplicate_inode"])
def test_rejects_incomplete_or_tampered_population(tmp_path: Path, mutation: str):
    plan_path = tmp_path / "plan.json"
    plan = _plan(plan_path)
    audit_path = tmp_path / "audit.json"
    payload = _audit(audit_path, plan)
    paths = [audit_path]
    if mutation == "missing":
        paths = []
    elif mutation == "tamper":
        payload["population"]["rows"] = 6
        audit_path.write_text(json.dumps(payload))
    else:
        plan["selections"].append(
            {**plan["selections"][0], "path": "data/component/other.zst"}
        )
        plan["selected_shards"] = 2
        plan["selected_compressed_bytes"] = 20
        plan["selections_sha256"] = canonical_sha256(plan["selections"])
        plan["receipt_sha256"] = canonical_sha256(
            {key: value for key, value in plan.items() if key != "receipt_sha256"}
        )
        plan_path.write_text(json.dumps(plan))
        paths = [audit_path, audit_path]
    with pytest.raises(HFStratifiedAuditAggregateError):
        aggregate_audits(plan_path, paths)
