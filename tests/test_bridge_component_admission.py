from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sai.data.bridge_component_admission import (
    BridgeComponentAdmissionError,
    admit,
)
from sai.data.bridge_transfer_confirmation import SCHEMA as CONFIRMATION_SCHEMA
from sai.data.practical_bridge_reconcile import SCHEMA as RECONCILIATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload: dict) -> dict:
    value = dict(payload)
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _stream(root: Path, name: str, rows: list[dict]) -> dict:
    path = root / name
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "path": name,
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fixture(root: Path, confirmation_pass: bool = True) -> tuple[Path, Path]:
    reconciliation = root / "reconciliation"
    reconciliation.mkdir(parents=True)
    train = [
        {
            "pair_identity_sha256": "1" * 64,
            "document_identity_sha256": "2" * 64,
            "corpus_split": "train",
            "text": "A verified cross-domain training lesson.",
            "transfer_ablation_complete": False,
            "training_ready": False,
        }
    ]
    development = [
        {
            "pair_identity_sha256": "3" * 64,
            "document_identity_sha256": "4" * 64,
            "corpus_split": "development",
            "text": "A source-disjoint development lesson.",
            "transfer_ablation_complete": False,
            "training_ready": False,
        }
    ]
    receipt = _signed(
        {
            "schema": RECONCILIATION_SCHEMA,
            "status": "complete_practical_bridge_foundation_reconciliation",
            "outputs": {
                "train": _stream(reconciliation, "train.jsonl", train),
                "development": _stream(
                    reconciliation, "development.jsonl", development
                ),
            },
            "global_exact_content_deduplication_complete": True,
            "development_source_disjoint_against_foundation_complete": True,
            "transfer_ablation_complete": False,
            "training_ready": False,
        }
    )
    (reconciliation / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )
    confirmation = root / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            _signed(
                {
                    "schema": CONFIRMATION_SCHEMA,
                    "status": "complete_bridge_transfer_proxy_confirmation",
                    "lineage": {
                        "reconciliation_receipt_sha256": receipt["receipt_sha256"]
                    },
                    "confirmation_pass": confirmation_pass,
                    "connection_component_admission_authorized": confirmation_pass,
                    "transfer_ablation_complete": True,
                    "training_ready": False,
                    "four_b_training_authorized": False,
                }
            ),
            sort_keys=True,
        )
        + "\n"
    )
    return reconciliation, confirmation


def test_admits_train_only_after_positive_confirmation(tmp_path: Path) -> None:
    reconciliation, confirmation = _fixture(tmp_path)
    result = admit(
        reconciliation,
        confirmation,
        tmp_path / "output",
        tmp_path / "evidence" / "receipt.json",
    )
    assert result["training_ready"] is True
    assert result["four_b_training_authorized"] is False
    assert result["counts"]["train_documents"] == 1
    assert result["counts"]["development_documents_excluded"] == 1
    with gzip.open(tmp_path / "output" / "train.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 1
    assert rows[0]["corpus_split"] == "train"
    assert rows[0]["training_ready"] is True
    assert rows[0]["transfer_ablation_complete"] is True
    assert "development" not in (tmp_path / "output" / "train.jsonl.gz").name


def test_deterministic_gzip_replays_exactly(tmp_path: Path) -> None:
    first_reconciliation, first_confirmation = _fixture(tmp_path / "first")
    first = admit(
        first_reconciliation,
        first_confirmation,
        tmp_path / "first-output",
        tmp_path / "first-evidence.json",
    )
    second_reconciliation, second_confirmation = _fixture(tmp_path / "second")
    second = admit(
        second_reconciliation,
        second_confirmation,
        tmp_path / "second-output",
        tmp_path / "second-evidence.json",
    )
    assert first["train"] == second["train"]


def test_failed_confirmation_cannot_admit(tmp_path: Path) -> None:
    reconciliation, confirmation = _fixture(tmp_path, confirmation_pass=False)
    with pytest.raises(BridgeComponentAdmissionError, match="not admissible"):
        admit(
            reconciliation,
            confirmation,
            tmp_path / "output",
            tmp_path / "evidence.json",
        )
