from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.common_pile_external_provenance import RECORD_SCHEMA, SCHEMA
from sai.data.external_rights_page_probe import (
    LICENSE_PATTERNS,
    ExternalRightsPageProbeError,
    build_probe,
    build_targets,
)
from sai.data.token_stream import canonical_sha256, sha256_file

BY = "Creative Commons - Attribution - " "https://creativecommons.org/licenses/by/4.0/"
BY_SA = (
    "Creative Commons - Attribution Share-Alike - "
    "https://creativecommons.org/licenses/by-sa/4.0/"
)
CC0 = (
    "Creative Commons Zero - Public Domain - "
    "https://creativecommons.org/publicdomain/zero/1.0/"
)


def _record(
    label: str,
    source_id: str,
    declared_license: str,
    source_metadata: dict[str, str],
) -> dict:
    row = {
        "schema": RECORD_SCHEMA,
        "source_id": source_id,
        "identity_sha256": canonical_sha256(label),
        "declared_license": declared_license,
        "source_metadata": source_metadata,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _root(tmp_path: Path, source_id: str, records: list[dict]) -> Path:
    root = tmp_path / source_id
    root.mkdir(parents=True)
    manifest = root / "external_provenance_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in records
        )
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete_text_free_source_metadata_replay",
        "source_id": source_id,
        "output": {
            "path": manifest.name,
            "rows": len(records),
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
        "source_text_persisted": False,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    (root / "receipt.json").write_text(json.dumps(payload))
    return root


def _population(tmp_path: Path) -> list[Path]:
    pressbooks = _root(
        tmp_path,
        "common_pile_pressbooks",
        [
            _record(
                "pressbooks-1",
                "common_pile_pressbooks",
                BY,
                {"metadata.book_url": "https://books.example/work"},
            ),
            _record(
                "pressbooks-2",
                "common_pile_pressbooks",
                BY,
                {"metadata.book_url": "https://books.example/work"},
            ),
        ],
    )
    pdr = _root(
        tmp_path,
        "common_pile_public_domain_review",
        [
            _record(
                "pdr-collection",
                "common_pile_public_domain_review",
                BY_SA,
                {
                    "metadata.url": "https://publicdomainreview.org/collection/a/",
                    "type": "collection",
                },
            ),
            _record(
                "pdr-conjecture",
                "common_pile_public_domain_review",
                BY_SA,
                {
                    "metadata.url": "https://publicdomainreview.org/conjectures/b/",
                    "type": "conjecture",
                },
            ),
            _record(
                "pdr-essay",
                "common_pile_public_domain_review",
                BY_SA,
                {
                    "metadata.url": "https://publicdomainreview.org/essay/c/",
                    "type": "essay",
                },
            ),
        ],
    )
    return [pressbooks, pdr]


def _successful_result(target: dict, **_kwargs: object) -> dict:
    observed = LICENSE_PATTERNS[target["expected_license"]][0].decode()
    result = {
        **target,
        "attempts": [{"attempt": 1, "outcome": "response", "status": 200}],
        "http_status": 200,
        "final_url": target["url"],
        "content_type": "text/html",
        "response_bytes_inspected": 100,
        "response_sha256": "a" * 64,
        "response_truncated": False,
        "expected_license_evidence_observed": True,
        "observed_pattern": observed,
        "error_type": None,
        "source_page_text_persisted": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def test_groups_targets_and_seals_text_free_coverage(tmp_path: Path) -> None:
    roots = _population(tmp_path)
    result = build_probe(
        roots,
        tmp_path / "probe",
        concurrency=3,
        timeout_seconds=5,
        maximum_attempts=1,
        fetch_function=_successful_result,
    )
    assert result["targets"]["count"] == 3
    assert result["population_records"] == 5
    assert result["records_covered_by_targets"] == 5
    assert result["records_with_observed_license_evidence"] == 5
    assert result["source_page_text_persisted"] is False
    assert result["rights_provenance_verified"] is False
    assert result["legal_clearance_established"] is False
    assert result["training_ready"] is False
    assert "page text" not in (tmp_path / "probe" / "results.jsonl").read_text()


def test_rejects_conflicting_pressbooks_work_declarations(tmp_path: Path) -> None:
    roots = _population(tmp_path)
    pressbooks_manifest = roots[0] / "external_provenance_manifest.jsonl"
    rows = [json.loads(line) for line in pressbooks_manifest.read_text().splitlines()]
    rows[1]["declared_license"] = CC0
    rows[1]["record_sha256"] = canonical_sha256(
        {key: value for key, value in rows[1].items() if key != "record_sha256"}
    )
    roots[0] = _root(tmp_path / "alternate", "common_pile_pressbooks", rows)
    with pytest.raises(ExternalRightsPageProbeError, match="conflicting"):
        build_targets(roots)


def test_rejects_tampered_fetch_result(tmp_path: Path) -> None:
    def tampered(target: dict, **kwargs: object) -> dict:
        result = _successful_result(target, **kwargs)
        result["response_bytes_inspected"] = 101
        return result

    with pytest.raises(ExternalRightsPageProbeError, match="result differs"):
        build_probe(
            _population(tmp_path),
            tmp_path / "probe",
            concurrency=1,
            timeout_seconds=5,
            maximum_attempts=1,
            fetch_function=tampered,
        )
    assert not (tmp_path / "probe").exists()
