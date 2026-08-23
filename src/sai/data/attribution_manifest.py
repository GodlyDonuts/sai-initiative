"""Build a text-free attribution and license-obligation manifest for documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import RAW_SCHEMA
from sai.data.license_policy import POLICY, classify_declared_license
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-document-attribution-manifest-v1"


class AttributionManifestError(RuntimeError):
    """Raw lineage, retained-document coverage, or license evidence differs."""


def _raw_source(row: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(row, dict) or row.get("schema") != RAW_SCHEMA:
        raise AttributionManifestError("attribution raw schema differs")
    source = row.get("source")
    if (
        not isinstance(row.get("text"), str)
        or not row["text"]
        or not isinstance(source, dict)
        or not isinstance(source.get("dataset"), str)
        or not source["dataset"]
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
        or not isinstance(source.get("source_file"), str)
        or not source["source_file"]
        or not isinstance(source.get("row_index"), int)
        or isinstance(source.get("row_index"), bool)
        or source["row_index"] < 0
        or not isinstance(source.get("license"), str)
        or not source["license"]
        or not isinstance(source.get("declared_license"), str)
        or not source["declared_license"]
        or not isinstance(source.get("domain"), str)
        or not source["domain"]
    ):
        raise AttributionManifestError("attribution raw provenance differs")
    row_id = canonical_sha256(
        {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "source_file": source["source_file"],
            "row_index": source["row_index"],
        }
    )
    return source, row_id


def _retained_documents(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise AttributionManifestError("attribution retained input is unsafe")
    retained = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                document = normalize_document(json.loads(line))
            except Exception as error:
                raise AttributionManifestError(
                    f"attribution retained row {line_number} differs"
                ) from error
            row_id = document["source"]["row_id"]
            if row_id in retained:
                raise AttributionManifestError(
                    "attribution retained source row is duplicated"
                )
            retained[row_id] = document
    if not retained:
        raise AttributionManifestError("attribution retained input is empty")
    return retained


def build_manifest(
    raw_path: Path,
    retained_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Replay raw lineage and seal exact obligations for retained documents."""

    if (
        output_path.exists()
        or receipt_path.exists()
        or not raw_path.is_file()
        or raw_path.is_symlink()
        or raw_path.stat().st_nlink != 1
    ):
        raise AttributionManifestError("attribution input or output boundary differs")
    raw_sha256 = sha256_file(raw_path)
    retained_sha256 = sha256_file(retained_path)
    retained = _retained_documents(retained_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    found: set[str] = set()
    ordered_identity = hashlib.sha256()
    obligation_counts = {
        "attribution_required": 0,
        "share_alike_required": 0,
    }
    try:
        with raw_path.open() as source_handle, temporary.open("w") as target:
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    source, row_id = _raw_source(raw)
                except (json.JSONDecodeError, AttributionManifestError) as error:
                    raise AttributionManifestError(
                        f"attribution raw row {line_number} differs"
                    ) from error
                document = retained.get(row_id)
                if document is None:
                    continue
                if row_id in found:
                    raise AttributionManifestError(
                        "attribution raw source row is duplicated"
                    )
                classification = classify_declared_license(
                    source["declared_license"]
                )
                if (
                    classification["rights_hold"]
                    or classification["canonical_license"] != source["license"]
                    or document["source"]["dataset"] != source["dataset"]
                    or document["source"]["license"] != source["license"]
                    or document["source"]["domain"] != source["domain"]
                ):
                    raise AttributionManifestError(
                        "attribution retained provenance or license differs"
                    )
                record = {
                    "schema": SCHEMA,
                    "identity_sha256": document["identity_sha256"],
                    "row_id": row_id,
                    "source": {
                        "dataset": source["dataset"],
                        "revision": source["revision"],
                        "source_file": source["source_file"],
                        "row_index": source["row_index"],
                        "domain": source["domain"],
                    },
                    "rights_declaration": {
                        "declared_license": classification["declared_license"],
                        "canonical_license": classification["canonical_license"],
                        "classification_sha256": classification[
                            "classification_sha256"
                        ],
                        "attribution_required": classification[
                            "attribution_required"
                        ],
                        "share_alike_required": classification[
                            "share_alike_required"
                        ],
                        "rights_hold": False,
                    },
                }
                record["record_sha256"] = canonical_sha256(record)
                target.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                ordered_identity.update(bytes.fromhex(document["identity_sha256"]))
                obligation_counts["attribution_required"] += bool(
                    classification["attribution_required"]
                )
                obligation_counts["share_alike_required"] += bool(
                    classification["share_alike_required"]
                )
                found.add(row_id)
        if found != set(retained):
            raise AttributionManifestError(
                "attribution retained-document coverage differs"
            )
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if (
        sha256_file(raw_path) != raw_sha256
        or sha256_file(retained_path) != retained_sha256
    ):
        output_path.unlink(missing_ok=True)
        raise AttributionManifestError("attribution input changed during replay")
    payload = {
        "schema": SCHEMA,
        "status": "complete_text_free_attribution_manifest",
        "raw_input": {
            "path": str(raw_path.resolve()),
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha256,
        },
        "retained_input": {
            "path": str(retained_path.resolve()),
            "bytes": retained_path.stat().st_size,
            "sha256": retained_sha256,
            "documents": len(retained),
        },
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "records": len(found),
            "ordered_identity_sha256": ordered_identity.hexdigest(),
        },
        "obligation_counts": obligation_counts,
        "license_policy": POLICY,
        "license_policy_sha256": canonical_sha256(POLICY),
        "exact_retained_document_coverage": True,
        "rights_declarations_recognized": True,
        "source_revision_file_and_row_lineage_replayed": True,
        "source_text_persisted_in_manifest": False,
        "external_source_provenance_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    try:
        _atomic_create(receipt_path, payload)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.raw, args.retained, args.output, args.receipt)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
