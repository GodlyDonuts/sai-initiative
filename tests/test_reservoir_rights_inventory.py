from __future__ import annotations

import json
from pathlib import Path

from sai.data.reservoir_rights_inventory import build_inventory
from sai.data.token_stream import canonical_sha256, sha256_file


def _reservoir(tmp_path: Path) -> Path:
    root = tmp_path / "reservoir"
    root.mkdir()
    rows = [
        {
            "source_id": "licensed_source",
            "repository": "owner/licensed",
            "revision": "a" * 40,
            "license": "odc-by-1.0",
            "access": "public",
            "physical_bytes": 100,
        },
        {
            "source_id": "row_specific_source",
            "repository": "owner/row-specific",
            "revision": "b" * 40,
            "license": "source_specific_public_domain_or_open_license",
            "access": "public",
            "physical_bytes": 200,
        },
        {
            "source_id": "missing_card_source",
            "repository": "owner/missing-card",
            "revision": "c" * 40,
            "license": "source_specific_public_domain_or_open_license",
            "access": "public",
            "physical_bytes": 50,
        },
        {
            "source_id": "composite_terms_source",
            "repository": "owner/composite",
            "revision": "d" * 40,
            "license": "apache-2.0_project_upstream_source_terms_apply",
            "access": "public",
            "physical_bytes": 25,
        },
    ]
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {
        "schema": "sai-source-reservoir-receipt-v1",
        "manifest": {
            "path": manifest.name,
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def _card(repository: str, revision: str, _token: str, _root: Path) -> dict:
    if repository == "owner/missing-card":
        return {
            "repository": repository,
            "revision": revision,
            "readme_present": False,
            "readme_bytes": 0,
            "readme_sha256": None,
            "top_level_card_license": None,
            "card_text_persisted": False,
        }
    return {
        "repository": repository,
        "revision": revision,
        "readme_bytes": 100,
        "readme_sha256": "f" * 64,
        "top_level_card_license": {
            "owner/licensed": "odc-by-1.0",
            "owner/composite": "apache-2.0",
        }.get(repository),
        "card_text_persisted": False,
    }


def test_inventory_routes_exact_and_row_specific_rights(tmp_path: Path) -> None:
    result = build_inventory(
        [_reservoir(tmp_path)],
        tmp_path / "inventory.json",
        token="test-token",
        acquire_card_function=_card,
    )
    rows = {row["source_id"]: row for row in result["source_rows"]}
    assert rows["licensed_source"]["rights_work_route"] == (
        "recognized_declaration_obligations_required"
    )
    assert rows["licensed_source"]["manifest_declaration_classification"][
        "canonical_license"
    ] == "ODC-By-1.0"
    assert rows["row_specific_source"]["rights_work_route"] == (
        "per_row_license_evidence_required"
    )
    assert rows["missing_card_source"]["rights_work_route"] == (
        "source_terms_resolution_required"
    )
    assert rows["composite_terms_source"]["rights_work_route"] == (
        "source_terms_resolution_required"
    )
    assert result["summary"]["physical_candidate_bytes"] == 375
    assert result["legal_clearance_established"] is False
    assert result["training_ready"] is False
