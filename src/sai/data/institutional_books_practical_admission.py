"""Admit the private English/non-slop Institutional Books pretraining core."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_materializer import (
    AGGREGATE_SCHEMA as MATERIALIZER_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA,
    _load_json,
    _valid_receipt,
)
from sai.data.institutional_books_mechanical_filter import (
    AGGREGATE_SCHEMA as FILTER_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_mechanical_filter import (
    SHARD_SCHEMA as FILTER_SHARD_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-practical-admission-receipt-v1"
RECORD_SCHEMA = "sai-institutional-books-practical-admission-row-v1"
SELECTION_SCHEMA = "sai-institutional-books-strict-english-selection-v1"
SELECTION_ROW_SCHEMA = "sai-institutional-books-strict-english-selection-row-v1"
ALLOWED_RIGHTS = ("cc-zero", "pd", "pdus")
POLICY = {
    "language_gen": "eng",
    "minimum_ocr_score_gen": 95,
    "allowed_rights_codes": list(ALLOWED_RIGHTS),
    "quality_requirement": "pass_mechanical_gate",
    "duplicate_policy": "one_smallest_barcode_per_exact_content_sha256",
    "semantic_model_review_required": False,
    "benchmark_decontamination_blocks_pretraining": False,
    "benchmark_decontamination_blocks_evaluation_claims": True,
    "redistribution": "private_only",
}
POLICY_SHA256 = canonical_sha256(POLICY)


class InstitutionalBooksPracticalAdmissionError(RuntimeError):
    """Selection, filter, or practical-admission custody differs."""


def _selection(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("selection")
    if (
        receipt.get("schema") != SELECTION_SCHEMA
        or receipt.get("status") != "complete_nontraining_strict_english_book_selection"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise InstitutionalBooksPracticalAdmissionError("selection receipt differs")
    path = root / descriptor.get("path", "")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise InstitutionalBooksPracticalAdmissionError("selection bytes differ")
    rows: dict[str, dict[str, Any]] = {}
    ordered = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            unsigned_row = {
                key: value for key, value in row.items() if key != "row_sha256"
            }
            barcode = row.get("barcode_src")
            if (
                row.get("schema") != SELECTION_ROW_SCHEMA
                or not isinstance(barcode, str)
                or not barcode
                or barcode in rows
                or row.get("row_sha256") != canonical_sha256(unsigned_row)
                or row.get("language_gen") != "eng"
                or row.get("rights_code") not in ALLOWED_RIGHTS
                or not isinstance(row.get("ocr_score_gen"), int)
                or row["ocr_score_gen"] < 95
                or row.get("training_ready") is not False
            ):
                raise InstitutionalBooksPracticalAdmissionError("selection row differs")
            rows[barcode] = row
            ordered.append(row["row_sha256"])
    if len(rows) != descriptor.get("rows") or canonical_sha256(
        ordered
    ) != descriptor.get("ordered_rows_sha256"):
        raise InstitutionalBooksPracticalAdmissionError("selection coverage differs")
    return rows, receipt


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksPracticalAdmissionError("admission manifest exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_admission(
    selection_root: Path,
    materialized_root: Path,
    filtered_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Seal a private practical-pretraining manifest over mechanical survivors."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksPracticalAdmissionError("output root exists")
    selection, selection_receipt = _selection(selection_root)
    materialized = _load_json(materialized_root / "aggregate.json")
    filtered = _load_json(filtered_root / "aggregate.json")
    if (
        not _valid_receipt(materialized, MATERIALIZER_AGGREGATE_SCHEMA)
        or not _valid_receipt(filtered, FILTER_AGGREGATE_SCHEMA)
        or materialized.get("selection", {}).get("receipt_sha256")
        != selection_receipt["receipt_sha256"]
        or filtered.get("materializer_receipt_sha256") != materialized["receipt_sha256"]
        or filtered.get("training_ready") is not False
    ):
        raise InstitutionalBooksPracticalAdmissionError("admission inputs differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksPracticalAdmissionError(
            "pyarrow is required"
        ) from error
    candidates = []
    seen_barcodes = set()
    logical_shards = filtered.get("shards", {}).get("logical_shards")
    if not isinstance(logical_shards, int) or logical_shards < 1:
        raise InstitutionalBooksPracticalAdmissionError("filter geometry differs")
    for shard_index in range(logical_shards):
        shard_root = filtered_root / "shards" / f"shard_{shard_index:05d}"
        receipt = _load_json(shard_root / "receipt.json")
        if (
            not _valid_receipt(receipt, FILTER_SHARD_SCHEMA)
            or receipt.get("shard_index") != shard_index
            or receipt.get("logical_shards") != logical_shards
        ):
            raise InstitutionalBooksPracticalAdmissionError("filter shard differs")
        descriptor = receipt.get("output")
        if descriptor is None:
            continue
        path = shard_root / descriptor.get("path", "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise InstitutionalBooksPracticalAdmissionError("filter shard bytes differ")
        parquet = pq.ParquetFile(path)
        rows = 0
        for batch in parquet.iter_batches(
            batch_size=4096,
            columns=[
                "schema",
                "barcode_src",
                "source_content_sha256",
                "enriched_token_count_gen",
                "text",
                "training_ready",
            ],
            use_threads=False,
        ):
            for value in batch.to_pylist():
                barcode = value.get("barcode_src")
                selected = selection.get(barcode)
                content_sha256 = value.get("source_content_sha256")
                tokens = value.get("enriched_token_count_gen")
                text = value.get("text")
                if (
                    value.get("schema") != OUTPUT_SCHEMA
                    or not isinstance(barcode, str)
                    or barcode in seen_barcodes
                    or selected is None
                    or not isinstance(content_sha256, str)
                    or len(content_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in content_sha256)
                    or not isinstance(tokens, int)
                    or tokens < 1
                    or not isinstance(text, str)
                    or not text
                    or value.get("training_ready") is not False
                ):
                    raise InstitutionalBooksPracticalAdmissionError(
                        "filtered practical row differs"
                    )
                seen_barcodes.add(barcode)
                rows += 1
                candidates.append(
                    {
                        "barcode_src": barcode,
                        "shard_index": shard_index,
                        "source_content_sha256": content_sha256,
                        "enriched_token_count_gen": tokens,
                        "source_text_utf8_bytes": len(text.encode()),
                        "selection_row_sha256": selected["row_sha256"],
                        "rights_code": selected["rights_code"],
                        "ocr_score_gen": selected["ocr_score_gen"],
                    }
                )
        if rows != descriptor.get("rows"):
            raise InstitutionalBooksPracticalAdmissionError(
                "filter shard row count differs"
            )
    expected = filtered.get("counts", {}).get("retained_rows")
    if len(candidates) != expected or len(seen_barcodes) != expected:
        raise InstitutionalBooksPracticalAdmissionError(
            "filtered practical coverage differs"
        )
    winners: dict[str, dict[str, Any]] = {}
    for row in candidates:
        current = winners.get(row["source_content_sha256"])
        if current is None or row["barcode_src"] < current["barcode_src"]:
            winners[row["source_content_sha256"]] = row
    accepted = []
    for row in sorted(
        winners.values(), key=lambda item: (item["shard_index"], item["barcode_src"])
    ):
        record = {
            "schema": RECORD_SCHEMA,
            **row,
            "language": "english",
            "quality_route": "pass_mechanical_gate",
            "private_filtered_relative_path": (
                f"shards/shard_{row['shard_index']:05d}/data.parquet"
            ),
            "official_benchmark_decontamination_complete": False,
            "evaluation_claims_allowed": False,
            "practical_pretraining_ready": True,
            "training_ready": True,
        }
        record["record_sha256"] = canonical_sha256(record)
        accepted.append(record)
    output_root.mkdir(parents=True)
    manifest = output_root / "manifest.jsonl"
    _atomic_jsonl(manifest, accepted)
    rights = Counter(row["rights_code"] for row in accepted)
    receipt = {
        "schema": SCHEMA,
        "status": "complete_practical_private_pretraining_admission",
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        "selection_receipt_sha256": selection_receipt["receipt_sha256"],
        "materializer_receipt_sha256": materialized["receipt_sha256"],
        "mechanical_filter_receipt_sha256": filtered["receipt_sha256"],
        "manifest": {
            "path": manifest.name,
            "rows": len(accepted),
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
            "ordered_records_sha256": canonical_sha256(
                [row["record_sha256"] for row in accepted]
            ),
        },
        "counts": {
            "mechanically_retained_rows": len(candidates),
            "exact_duplicate_rows_excluded": len(candidates) - len(accepted),
            "admitted_rows": len(accepted),
            "admitted_enriched_tokens": sum(
                row["enriched_token_count_gen"] for row in accepted
            ),
            "admitted_text_utf8_bytes": sum(
                row["source_text_utf8_bytes"] for row in accepted
            ),
            "private_filtered_compressed_bytes": filtered["counts"]["output_bytes"],
            "rights_codes": dict(sorted(rights.items())),
        },
        "source_text_location": str(filtered_root.resolve()),
        "source_text_rewritten": False,
        "source_text_uploaded_to_huggingface": False,
        "huggingface_redistribution_authorized": False,
        "semantic_model_review_required": False,
        "official_benchmark_decontamination_complete": False,
        "evaluation_claims_allowed": False,
        "practical_pretraining_ready": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_create(output_root / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_admission(
        args.selection_root,
        args.materialized_root,
        args.filtered_root,
        args.output_root,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
