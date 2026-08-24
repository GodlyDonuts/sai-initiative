from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.common_pile_full_source_candidates import (
    CommonPileFullSourceCandidatesError,
    load_promotion,
)
from sai.data.common_pile_full_source_promotion import SCHEMA as PROMOTION_SCHEMA
from sai.data.token_stream import canonical_sha256


def _promotion(path: Path, *, authorized: bool = True) -> Path:
    source_id = "common_pile_pressbooks"
    source = {
        "source_id": source_id,
        "parent": {
            "source_id": source_id,
            "repository": "common-pile/pressbooks_filtered",
            "revision": "a" * 40,
            "path": "pressbooks-0000.json.gz",
            "bytes": 1_000,
            "sha256": "b" * 64,
            "manifest_license": "CC-BY-4.0",
            "domain": "english",
        },
        "checks": {"minimum_retain_ppm": authorized},
        "full_source_candidate_materialization_authorized": authorized,
        "bulk_training_admission": False,
        "training_ready": False,
    }
    payload = {
        "schema": PROMOTION_SCHEMA,
        "status": "complete_candidate_only_source_decision",
        "sources": [source],
        "authorized_source_ids": [source_id] if authorized else [],
        "full_source_materialization_is_training_admission": False,
        "rights_provenance_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_loads_exact_candidate_only_promotion(tmp_path: Path) -> None:
    payload, source = load_promotion(
        _promotion(tmp_path / "promotion.json"), "common_pile_pressbooks"
    )
    assert payload["training_ready"] is False
    assert source["parent"]["path"] == "pressbooks-0000.json.gz"


def test_rejects_source_without_per_source_authorization(tmp_path: Path) -> None:
    with pytest.raises(CommonPileFullSourceCandidatesError, match="not promoted"):
        load_promotion(
            _promotion(tmp_path / "promotion.json", authorized=False),
            "common_pile_pressbooks",
        )
