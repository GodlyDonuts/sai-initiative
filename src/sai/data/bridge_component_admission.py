"""Admit only positively confirmed, reconciled connection lessons for training."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_transfer_confirmation import SCHEMA as CONFIRMATION_SCHEMA
from sai.data.practical_bridge_reconcile import SCHEMA as RECONCILIATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bridge-training-component-admission-v1"
STATUS = "complete_bridge_training_component_admission"


class BridgeComponentAdmissionError(RuntimeError):
    """A confirmation, split, row, output, or admission invariant differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BridgeComponentAdmissionError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeComponentAdmissionError("signed input differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise BridgeComponentAdmissionError("signed input differs")
    return payload


def _load_rows(root: Path, descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / str(descriptor.get("path"))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise BridgeComponentAdmissionError("reconciled stream differs")
    try:
        rows = [json.loads(line) for line in path.open()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeComponentAdmissionError("reconciled rows differ") from error
    if len(rows) != descriptor.get("rows"):
        raise BridgeComponentAdmissionError("reconciled row coverage differs")
    return rows


def admit(
    reconciliation_root: Path,
    confirmation_path: Path,
    output_root: Path,
    durable_receipt: Path,
) -> dict[str, Any]:
    """Write a train-only deterministic gzip after all transfer evidence passes."""

    if output_root.exists() or output_root.is_symlink() or durable_receipt.exists():
        raise BridgeComponentAdmissionError("admission output exists")
    reconciliation = _load_signed(
        reconciliation_root / "receipt.json", RECONCILIATION_SCHEMA
    )
    confirmation = _load_signed(confirmation_path, CONFIRMATION_SCHEMA)
    if (
        reconciliation.get("status")
        != "complete_practical_bridge_foundation_reconciliation"
        or reconciliation.get("global_exact_content_deduplication_complete") is not True
        or reconciliation.get("development_source_disjoint_against_foundation_complete")
        is not True
        or reconciliation.get("transfer_ablation_complete") is not False
        or reconciliation.get("training_ready") is not False
    ):
        raise BridgeComponentAdmissionError("reconciliation is not admissible")
    if (
        confirmation.get("status") != "complete_bridge_transfer_proxy_confirmation"
        or confirmation.get("confirmation_pass") is not True
        or confirmation.get("connection_component_admission_authorized") is not True
        or confirmation.get("transfer_ablation_complete") is not True
        or confirmation.get("training_ready") is not False
        or confirmation.get("four_b_training_authorized") is not False
        or confirmation.get("lineage", {}).get("reconciliation_receipt_sha256")
        != reconciliation["receipt_sha256"]
    ):
        raise BridgeComponentAdmissionError("confirmation is not admissible")
    outputs = reconciliation.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("train"), dict):
        raise BridgeComponentAdmissionError("reconciliation outputs differ")
    if not isinstance(outputs.get("development"), dict):
        raise BridgeComponentAdmissionError("development descriptor differs")
    train_rows = _load_rows(reconciliation_root, outputs["train"])
    development_rows = _load_rows(reconciliation_root, outputs["development"])
    train_pairs = {row.get("pair_identity_sha256") for row in train_rows}
    development_pairs = {row.get("pair_identity_sha256") for row in development_rows}
    if (
        not train_rows
        or not development_rows
        or None in train_pairs
        or None in development_pairs
        or train_pairs & development_pairs
    ):
        raise BridgeComponentAdmissionError("reconciled pair custody differs")

    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        gzip_path = stage / "train.jsonl.gz"
        ordered_documents = []
        uncompressed_digest = hashlib.sha256()
        uncompressed_bytes = 0
        text_bytes = 0
        with (
            gzip_path.open("xb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            for row in train_rows:
                if (
                    row.get("corpus_split") != "train"
                    or row.get("training_ready") is not False
                    or row.get("transfer_ablation_complete") is not False
                    or not isinstance(row.get("document_identity_sha256"), str)
                    or not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                ):
                    raise BridgeComponentAdmissionError("training row differs")
                value = dict(row)
                value["connection_component_admission_authorized"] = True
                value["transfer_ablation_complete"] = True
                value["training_ready"] = True
                encoded = (
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                compressed.write(encoded)
                uncompressed_digest.update(encoded)
                uncompressed_bytes += len(encoded)
                text_bytes += len(value["text"].encode())
                ordered_documents.append(value["document_identity_sha256"])
        descriptor = {
            "path": gzip_path.name,
            "rows": len(train_rows),
            "bytes": gzip_path.stat().st_size,
            "sha256": sha256_file(gzip_path),
            "decompressed_bytes": uncompressed_bytes,
            "decompressed_sha256": uncompressed_digest.hexdigest(),
            "text_utf8_bytes": text_bytes,
            "ordered_document_identities_sha256": canonical_sha256(ordered_documents),
            "compression": "gzip-mtime-0-no-filename",
        }
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "inputs": {
                "reconciliation_receipt_sha256": reconciliation["receipt_sha256"],
                "confirmation_receipt_sha256": confirmation["receipt_sha256"],
            },
            "train": descriptor,
            "counts": {
                "train_documents": len(train_rows),
                "train_pairs": len(train_pairs),
                "development_documents_excluded": len(development_rows),
                "development_pairs_excluded": len(development_pairs),
            },
            "development_rows_physically_excluded": True,
            "global_exact_content_deduplication_complete": True,
            "development_source_disjoint_against_foundation_complete": True,
            "transfer_ablation_complete": True,
            "connection_component_admission_authorized": True,
            "training_ready": True,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        durable_receipt.parent.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_create(durable_receipt, payload)
        except Exception:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        return payload
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation-root", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = admit(
        args.reconciliation_root,
        args.confirmation,
        args.output_root,
        args.durable_receipt,
    )
    print(
        json.dumps(
            {
                "train_documents": result["counts"]["train_documents"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
