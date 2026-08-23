"""Replay text-free external provenance metadata for a bounded Common Pile pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.agent_labeling import _atomic_create
from sai.data.attribution_manifest import _raw_source
from sai.data.bounded_pilot_compiler_population import _load_attribution
from sai.data.common_pile_audit_population import _declared_license, _native_id
from sai.data.common_pile_streaming_pilot import (
    SCHEMA as PILOT_SCHEMA,
)
from sai.data.common_pile_streaming_pilot import (
    download_parent,
)
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-external-provenance-manifest-v1"
RECORD_SCHEMA = "sai-common-pile-external-provenance-record-v1"
TOP_LEVEL_FIELDS = ("id", "source", "added", "created", "author", "date", "type")
METADATA_FIELDS = (
    "url",
    "book_url",
    "provenance",
    "author",
    "institution",
    "subject",
    "title",
    "license",
)
URL_FIELDS = frozenset({"id", "url", "book_url"})


class CommonPileExternalProvenanceError(RuntimeError):
    """A pilot, parent row, or source metadata identity differs."""


def _descriptor(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        "path": row.get(f"{prefix}path"),
        "bytes": row.get(f"{prefix}bytes"),
        "sha256": row.get(f"{prefix}sha256"),
    }


def _safe_string(value: Any, label: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value.encode()) > 16_384:
        raise CommonPileExternalProvenanceError(f"{label} differs")
    return value


def _safe_url(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise CommonPileExternalProvenanceError(f"{label} URL differs")
    return value


def _source_metadata(row: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise CommonPileExternalProvenanceError("parent metadata differs")
    extracted: dict[str, str] = {}
    urls = set()
    for field in TOP_LEVEL_FIELDS:
        value = _safe_string(row.get(field), f"parent {field}")
        if value is None:
            continue
        extracted[field] = value
        if field in URL_FIELDS and urlsplit(value).scheme:
            urls.add(_safe_url(value, f"parent {field}"))
    for field in METADATA_FIELDS:
        value = _safe_string(metadata.get(field), f"parent metadata {field}")
        if value is None:
            continue
        extracted[f"metadata.{field}"] = value
        if field in URL_FIELDS:
            urls.add(_safe_url(value, f"parent metadata {field}"))
    return dict(sorted(extracted.items())), sorted(urls)


def _retained_raw_rows(
    raw_path: Path, attribution: dict[str, dict[str, Any]], parent: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    by_row_id = {row["row_id"]: row for row in attribution.values()}
    retained: dict[int, dict[str, Any]] = {}
    with raw_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
                source, row_id = _raw_source(raw)
            except Exception as error:
                raise CommonPileExternalProvenanceError(
                    f"raw row {line_number} differs"
                ) from error
            record = by_row_id.get(row_id)
            if record is None:
                continue
            if (
                source["dataset"] != parent["repository"]
                or source["revision"] != parent["revision"]
                or source["source_file"] != parent["path"]
                or source["declared_license"]
                != record["rights_declaration"]["declared_license"]
                or source["license"]
                != record["rights_declaration"]["canonical_license"]
                or source["row_index"] in retained
            ):
                raise CommonPileExternalProvenanceError(
                    "raw and attribution provenance differ"
                )
            retained[source["row_index"]] = {
                "row_id": row_id,
                "identity_sha256": record["identity_sha256"],
                "text_sha256": hashlib.sha256(raw["text"].strip().encode()).hexdigest(),
                "declared_license": source["declared_license"],
                "attribution_record_sha256": record["record_sha256"],
            }
    if len(retained) != len(attribution):
        raise CommonPileExternalProvenanceError(
            "retained raw and attribution coverage differ"
        )
    return retained


def _replay_parent(
    compressed_path: Path,
    parent: dict[str, Any],
    retained: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not compressed_path.is_file()
        or compressed_path.is_symlink()
        or compressed_path.stat().st_size != parent.get("bytes")
        or sha256_file(compressed_path) != parent.get("sha256")
    ):
        raise CommonPileExternalProvenanceError("external provenance parent differs")
    records = []
    found = set()
    try:
        with gzip.open(compressed_path, "rt", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                expected = retained.get(row_index)
                if expected is None:
                    continue
                row = json.loads(line)
                text = row.get("text")
                if (
                    not isinstance(row, dict)
                    or not isinstance(text, str)
                    or hashlib.sha256(text.strip().encode()).hexdigest()
                    != expected["text_sha256"]
                    or _declared_license(row, parent["manifest_license"])
                    != expected["declared_license"]
                ):
                    raise CommonPileExternalProvenanceError(
                        "external provenance parent row differs"
                    )
                metadata, urls = _source_metadata(row)
                native_id = _native_id(row)
                if native_id is not None:
                    native_id = _safe_string(native_id, "parent native ID")
                record = {
                    "schema": RECORD_SCHEMA,
                    "source_id": parent["source_id"],
                    "identity_sha256": expected["identity_sha256"],
                    "row_id": expected["row_id"],
                    "parent_row_index": row_index,
                    "parent_line_number": row_index + 1,
                    "native_id": native_id,
                    "declared_license": expected["declared_license"],
                    "attribution_record_sha256": expected["attribution_record_sha256"],
                    "source_metadata": metadata,
                    "source_urls": urls,
                    "source_metadata_sha256": canonical_sha256(metadata),
                }
                record["record_sha256"] = canonical_sha256(record)
                records.append(record)
                found.add(row_index)
                if len(found) == len(retained):
                    break
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CommonPileExternalProvenanceError(
            "external provenance parent cannot be replayed"
        ) from error
    if found != set(retained):
        raise CommonPileExternalProvenanceError(
            "external provenance parent coverage differs"
        )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_manifest(
    pilot_root: Path,
    output_root: Path,
    *,
    token: str,
    download_function: Callable[[dict[str, Any], str, Path], Path] = download_parent,
) -> dict[str, Any]:
    """Replay one exact compressed parent and retain no source text."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise CommonPileExternalProvenanceError(
            "external provenance credential or output differs"
        )
    pilot_receipt_path = pilot_root / "receipt.json"
    pilot = _load_receipt(pilot_receipt_path)
    raw = pilot.get("raw_population")
    attribution = pilot.get("attribution_manifest")
    parent = pilot.get("parent")
    if (
        pilot.get("schema") != PILOT_SCHEMA
        or pilot.get("status") != "complete_nontraining_pilot"
        or pilot.get("training_ready") is not False
        or not isinstance(raw, dict)
        or not isinstance(attribution, dict)
        or not isinstance(parent, dict)
    ):
        raise CommonPileExternalProvenanceError("external provenance pilot differs")
    raw_path = _bound_file(pilot_root, _descriptor(raw))
    attribution_path = _bound_file(pilot_root, _descriptor(attribution, "output_"))
    attribution_records = _load_attribution(attribution_path)
    retained = _retained_raw_rows(raw_path, attribution_records, parent)
    output_root.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="sai-external-provenance-") as temp:
            compressed = download_function(parent, token, Path(temp))
            records = _replay_parent(compressed, parent, retained)
        output_path = output_root / "external_provenance_manifest.jsonl"
        _write_jsonl(output_path, records)
        field_counts = Counter(
            field for record in records for field in record["source_metadata"]
        )
        unique_urls = {url for record in records for url in record["source_urls"]}
        payload = {
            "schema": SCHEMA,
            "status": "complete_text_free_source_metadata_replay",
            "source_id": pilot["source_id"],
            "pilot": {
                "root_name": pilot_root.name,
                "receipt_file_sha256": sha256_file(pilot_receipt_path),
                "receipt_sha256": pilot["receipt_sha256"],
            },
            "parent": parent,
            "output": {
                "path": output_path.name,
                "rows": len(records),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in records]
                ),
            },
            "source_metadata_field_counts": dict(sorted(field_counts.items())),
            "unique_source_urls": len(unique_urls),
            "exact_retained_document_coverage": True,
            "full_compressed_parent_size_and_sha256_replayed": True,
            "source_metadata_replay_complete": True,
            "source_text_persisted": False,
            "parent_removed_after_replay": True,
            "maximum_simultaneous_parent_files": 1,
            "external_source_pages_verified": False,
            "rights_provenance_verified": False,
            "legal_clearance_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = build_manifest(
        args.pilot_root,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
