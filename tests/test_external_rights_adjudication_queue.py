from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.external_rights_adjudication_queue import (
    ROUTES,
    ExternalRightsAdjudicationQueueError,
    build_queue,
)
from sai.data.external_rights_page_probe import LICENSE_PATTERNS, SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file


def _record(label: str, source_id: str, license_name: str, metadata: dict) -> dict:
    row = {
        "identity_sha256": canonical_sha256(label),
        "source_id": source_id,
        "declared_license": license_name,
        "source_metadata": metadata,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _target(
    source_id: str,
    scope: str,
    url: str,
    expected_license: str,
    records: list[dict],
) -> dict:
    target = {
        "scope": scope,
        "source_id": source_id,
        "url": url,
        "expected_license": expected_license,
        "record_count": len(records),
        "ordered_identity_sha256": canonical_sha256(
            [row["identity_sha256"] for row in records]
        ),
    }
    target["target_sha256"] = canonical_sha256(target)
    return target


def _result(target: dict, *, status: int, observed: bool) -> dict:
    pattern = (
        LICENSE_PATTERNS[target["expected_license"]][0].decode() if observed else None
    )
    response_bytes = 100 if status == 200 else 0
    result = {
        **target,
        "attempts": [
            {
                "attempt": 1,
                "outcome": "response" if status == 200 else "http_error",
                "status": status,
            }
        ],
        "http_status": status,
        "final_url": target["url"] if status == 200 else None,
        "content_type": "text/html" if status == 200 else None,
        "response_bytes_inspected": response_bytes,
        "response_sha256": "a" * 64 if response_bytes else None,
        "response_truncated": False,
        "expected_license_evidence_observed": observed,
        "observed_pattern": pattern,
        "error_type": None if status == 200 else "http_error",
        "source_page_text_persisted": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _fixture(tmp_path: Path) -> tuple[Path, list[dict], list[dict], list[dict]]:
    by = (
        "Creative Commons - Attribution - "
        "https://creativecommons.org/licenses/by/4.0/"
    )
    by_sa = (
        "Creative Commons - Attribution Share-Alike - "
        "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    pressbooks = [
        _record(
            "pressbooks-1",
            "common_pile_pressbooks",
            by,
            {"metadata.book_url": "https://books.example/work"},
        ),
        _record(
            "pressbooks-2",
            "common_pile_pressbooks",
            by,
            {"metadata.book_url": "https://books.example/work"},
        ),
    ]
    pdr = [
        _record(
            "pdr-essay",
            "common_pile_public_domain_review",
            by_sa,
            {
                "metadata.url": "https://publicdomainreview.org/essay/work/",
                "type": "essay",
            },
        )
    ]
    targets = [
        _target(
            "common_pile_pressbooks",
            "pressbooks_work_page",
            "https://books.example/work",
            "CC-BY-4.0",
            pressbooks,
        ),
        _target(
            "common_pile_public_domain_review",
            "public_domain_review_essay_page",
            "https://publicdomainreview.org/essay/work/",
            "CC-BY-SA-4.0",
            pdr,
        ),
    ]
    provenance = [
        {"source_id": "common_pile_pressbooks", "records": 2},
        {"source_id": "common_pile_public_domain_review", "records": 1},
    ]
    results = [_result(targets[0], status=403, observed=False)]
    results.append(_result(targets[1], status=200, observed=True))
    probe = tmp_path / "probe"
    probe.mkdir()
    result_path = probe / "results.jsonl"
    result_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in results
        )
    )
    receipt = {
        "schema": SCHEMA,
        "status": "complete_text_free_page_evidence_probe",
        "provenance_bindings": provenance,
        "targets": {"count": 2},
        "population_records": 3,
        "records_covered_by_targets": 3,
        "results": {
            "path": result_path.name,
            "rows": 2,
            "bytes": result_path.stat().st_size,
            "sha256": sha256_file(result_path),
        },
        "source_page_text_persisted": False,
        "rights_provenance_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (probe / "receipt.json").write_text(json.dumps(receipt))
    return probe, targets, provenance, pressbooks + pdr


def test_builds_exact_fail_closed_identity_routes(tmp_path: Path) -> None:
    probe, targets, provenance, records = _fixture(tmp_path)
    result = build_queue(
        [tmp_path / "pressbooks", tmp_path / "pdr"],
        probe,
        tmp_path / "queue",
        target_builder=lambda _roots: (targets, provenance, records),
    )
    assert result["population_records"] == 3
    assert result["records_by_adjudication_route"] == {
        ROUTES["access_blocked"]: 2,
        ROUTES["pdr_essay_observed"]: 1,
    }
    assert result["records_with_observed_license_evidence"] == 1
    assert result["exact_identity_coverage"] is True
    assert result["access_control_bypassed"] is False
    assert result["rights_provenance_verified"] is False
    assert result["training_ready"] is False


def test_rejects_grouped_identity_hash_tamper(tmp_path: Path) -> None:
    probe, targets, provenance, records = _fixture(tmp_path)
    records[:2] = reversed(records[:2])
    with pytest.raises(ExternalRightsAdjudicationQueueError, match="grouped identity"):
        build_queue(
            [tmp_path / "pressbooks", tmp_path / "pdr"],
            probe,
            tmp_path / "queue",
            target_builder=lambda _roots: (targets, provenance, records),
        )
