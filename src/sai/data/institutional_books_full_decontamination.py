"""Screen full consensus book texts against official benchmark boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Container
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_overlap_count,
    _normalize,
    _overlap_count,
    binary_boundary_index,
)
from sai.data.institutional_books_independent_agreement import (
    RECORD_SCHEMA as AGREEMENT_RECORD_SCHEMA,
)
from sai.data.institutional_books_independent_agreement import (
    SCHEMA as AGREEMENT_SCHEMA,
)
from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA,
    _load_json,
    _valid_receipt,
)
from sai.data.institutional_books_mechanical_filter import (
    SHARD_SCHEMA as FILTER_SHARD_SCHEMA,
)
from sai.data.institutional_books_semantic_population import _filtered_index
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-full-benchmark-decontamination-receipt-v1"
DECISION_SCHEMA = "sai-institutional-books-full-benchmark-decontamination-v1"
CLEAN_SCHEMA = "sai-institutional-books-benchmark-disjoint-consensus-v1"


class InstitutionalBooksFullDecontaminationError(RuntimeError):
    """Agreement, full book, boundary, or decontamination custody differs."""


class _Union:
    def __init__(self, members: list[Container[bytes]]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def screen_book(
    agreement: dict[str, Any],
    text: str,
    full_source_content_sha256: str,
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> dict[str, Any]:
    """Create one text-free decision from the complete selected book text."""

    if (
        agreement.get("schema") != AGREEMENT_RECORD_SCHEMA
        or agreement.get("disposition") != "consensus_candidate"
        or agreement.get("training_ready") is not False
        or not isinstance(text, str)
        or hashlib.sha256(text.encode()).hexdigest() != full_source_content_sha256
    ):
        raise InstitutionalBooksFullDecontaminationError(
            "book decontamination candidate differs"
        )
    normalized = _normalize(text)
    word_overlaps = _overlap_count(
        _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
    )
    code_overlaps = _code_overlap_count(_CODE.findall(normalized), code_boundary)
    decision = {
        "schema": DECISION_SCHEMA,
        "agreement_record_sha256": agreement["record_sha256"],
        "candidate_identity_sha256": agreement["candidate_identity_sha256"],
        "source_book_id": agreement["source_book_id"],
        "full_source_content_sha256": full_source_content_sha256,
        "full_source_utf8_bytes": len(text.encode()),
        "word_overlap_count": word_overlaps,
        "code_overlap_count": code_overlaps,
        "contaminated": bool(word_overlaps or code_overlaps),
        "full_source_text_persisted": False,
        "training_ready": False,
    }
    decision["record_sha256"] = canonical_sha256(decision)
    return decision


def _agreement(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("manifest")
    if (
        receipt.get("schema") != AGREEMENT_SCHEMA
        or receipt.get("status") != "complete_nontraining_independent_book_agreement"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("consensus_is_training_admission") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise InstitutionalBooksFullDecontaminationError(
            "book agreement receipt differs"
        )
    path = root / descriptor.get("path", "")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise InstitutionalBooksFullDecontaminationError(
            "book agreement manifest differs"
        )
    rows = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            unsigned_row = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            identity = row.get("candidate_identity_sha256")
            if (
                row.get("schema") != AGREEMENT_RECORD_SCHEMA
                or not isinstance(identity, str)
                or identity in seen
                or row.get("record_sha256") != canonical_sha256(unsigned_row)
                or row.get("training_ready") is not False
            ):
                raise InstitutionalBooksFullDecontaminationError(
                    "book agreement record differs"
                )
            seen.add(identity)
            rows.append(row)
    if len(rows) != descriptor.get("rows") or canonical_sha256(
        [row["record_sha256"] for row in rows]
    ) != descriptor.get("ordered_records_sha256"):
        raise InstitutionalBooksFullDecontaminationError(
            "book agreement coverage differs"
        )
    return rows, receipt


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksFullDecontaminationError("output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_screen(
    agreement_root: Path,
    filtered_root: Path,
    boundary_roots: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    """Replay full selected books and emit benchmark-disjoint identities."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not boundary_roots
        or len(boundary_roots) != len(set(boundary_roots))
    ):
        raise InstitutionalBooksFullDecontaminationError(
            "book decontamination output differs"
        )
    agreements, agreement_receipt = _agreement(agreement_root)
    consensus = {
        row["source_book_id"]: row
        for row in agreements
        if row.get("disposition") == "consensus_candidate"
    }
    if len(consensus) != sum(
        row.get("disposition") == "consensus_candidate" for row in agreements
    ):
        raise InstitutionalBooksFullDecontaminationError(
            "consensus book identity differs"
        )
    filtered, filter_aggregate = _filtered_index(filtered_root)
    if not set(consensus).issubset(filtered):
        raise InstitutionalBooksFullDecontaminationError(
            "consensus-to-filter coverage differs"
        )
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    decisions = []
    found = set()
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        logical_shards = filter_aggregate["shards"]["logical_shards"]
        shard_indexes = range(logical_shards) if consensus else ()
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise InstitutionalBooksFullDecontaminationError(
                "pyarrow is required"
            ) from error
        for index in shard_indexes:
            shard_root = filtered_root / "shards" / f"shard_{index:05d}"
            receipt = _load_json(shard_root / "receipt.json")
            if not _valid_receipt(receipt, FILTER_SHARD_SCHEMA):
                raise InstitutionalBooksFullDecontaminationError(
                    "filtered shard differs"
                )
            descriptor = receipt.get("output")
            if descriptor is None:
                continue
            parquet = pq.ParquetFile(shard_root / descriptor["path"])
            for batch in parquet.iter_batches(
                batch_size=16,
                columns=[
                    "schema",
                    "barcode_src",
                    "text",
                    "source_content_sha256",
                    "training_ready",
                ],
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    barcode = row.get("barcode_src")
                    if barcode not in consensus:
                        continue
                    if (
                        row.get("schema") != OUTPUT_SCHEMA
                        or barcode in found
                        or row.get("training_ready") is not False
                        or row.get("source_content_sha256")
                        != filtered[barcode]["source_content_sha256"]
                    ):
                        raise InstitutionalBooksFullDecontaminationError(
                            "consensus filtered book differs"
                        )
                    decisions.append(
                        screen_book(
                            consensus[barcode],
                            row["text"],
                            row["source_content_sha256"],
                            word_boundary,
                            code_boundary,
                        )
                    )
                    found.add(barcode)
    finally:
        for member in [*words, *code]:
            member.close()
    if found != set(consensus):
        raise InstitutionalBooksFullDecontaminationError(
            "consensus full-text coverage differs"
        )
    decisions.sort(key=lambda row: row["candidate_identity_sha256"])
    clean = []
    totals: Counter[str] = Counter()
    for decision in decisions:
        totals["contaminated_rows"] += decision["contaminated"]
        totals["word_overlap_shingles"] += decision["word_overlap_count"]
        totals["code_overlap_shingles"] += decision["code_overlap_count"]
        if decision["contaminated"]:
            continue
        agreement = consensus[decision["source_book_id"]]
        row = {
            "schema": CLEAN_SCHEMA,
            "agreement_record_sha256": agreement["record_sha256"],
            "decontamination_record_sha256": decision["record_sha256"],
            "candidate_identity_sha256": agreement["candidate_identity_sha256"],
            "source_book_id": agreement["source_book_id"],
            "full_source_content_sha256": decision["full_source_content_sha256"],
            "full_source_utf8_bytes": decision["full_source_utf8_bytes"],
            "token_count_o200k_base_gen": agreement["token_count_o200k_base_gen"],
            "agreed_genre": agreement["agreed_genre"],
            "shared_domains": agreement["shared_domains"],
            "benchmark_decontamination_complete": True,
            "global_semantic_deduplication_complete": False,
            "source_text_persisted": False,
            "training_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        clean.append(row)
    output_root.mkdir(parents=True)
    try:
        decision_path = output_root / "decisions.jsonl"
        clean_path = output_root / "benchmark_disjoint_books.jsonl"
        _atomic_jsonl(decision_path, decisions)
        _atomic_jsonl(clean_path, clean)
        payload = {
            "schema": SCHEMA,
            "status": "complete_full_consensus_book_benchmark_decontamination",
            "agreement": {
                "receipt_sha256": agreement_receipt["receipt_sha256"],
                "manifest_sha256": agreement_receipt["manifest"]["sha256"],
            },
            "filter_aggregate_receipt_sha256": filter_aggregate["receipt_sha256"],
            "boundary_indexes": boundary_receipts,
            "boundary_indexes_sha256": canonical_sha256(boundary_receipts),
            "policy": POLICY,
            "policy_sha256": canonical_sha256(POLICY),
            "input_rows": len(consensus),
            "clean_rows": len(clean),
            "contaminated_rows": totals["contaminated_rows"],
            "word_overlap_shingles": totals["word_overlap_shingles"],
            "code_overlap_shingles": totals["code_overlap_shingles"],
            "decisions": {
                "path": decision_path.name,
                "rows": len(decisions),
                "bytes": decision_path.stat().st_size,
                "sha256": sha256_file(decision_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in decisions]
                ),
            },
            "benchmark_disjoint_books": {
                "path": clean_path.name,
                "rows": len(clean),
                "bytes": clean_path.stat().st_size,
                "sha256": sha256_file(clean_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in clean]
                ),
                "full_source_utf8_bytes": sum(
                    row["full_source_utf8_bytes"] for row in clean
                ),
                "tokens": sum(row["token_count_o200k_base_gen"] for row in clean),
            },
            "full_selected_source_population_decontaminated": True,
            "global_semantic_deduplication_complete": False,
            "source_text_persisted": False,
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
    parser.add_argument("--agreement-root", type=Path, required=True)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_screen(
        args.agreement_root,
        args.filtered_root,
        args.boundary_index,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
