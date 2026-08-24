"""Stream rights-bound final Institutional Books into bounded tokenizer sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books import ENRICHED_REPOSITORY, ENRICHED_REVISION
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    OUTPUT_SCHEMA,
    SHARD_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_quality_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from sai.data.institutional_books_quality_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from sai.data.institutional_books_selection import ALLOWED_RIGHTS_CODES
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import (
    ROW_SCHEMA,
    canonical_sha256,
    normalize_document,
    sha256_file,
)
from sai.data.transient_tokenizer_sample import (
    GENERIC_SOURCE_RECEIPT_SCHEMA,
    GENERIC_SOURCE_STATUS,
    INSTITUTIONAL_BOOK_ENVELOPE_SCHEMA,
)

RIGHTS_LABELS = {
    "pd": "Public Domain (HathiTrust rights code: pd)",
    "pdus": "Public Domain in the United States (HathiTrust rights code: pdus)",
    "cc-zero": "CC0 (HathiTrust rights code: cc-zero)",
}


class InstitutionalBooksTransientTokenizerStreamError(RuntimeError):
    """Book rights, final shard custody, row identity, or stream coverage differs."""


def _selection(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_signed(root / "receipt.json", SELECTION_SCHEMA)
    descriptor = receipt.get("selection")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        receipt.get("status") != "complete_nontraining_strict_english_book_selection"
        or receipt.get("selection_contains_source_text") is not False
        or not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise InstitutionalBooksTransientTokenizerStreamError("book selection differs")
    selected = {}
    ordered = []
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise InstitutionalBooksTransientTokenizerStreamError(
                    "book selection row differs"
                ) from error
            unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
            barcode = row.get("barcode_src")
            if (
                row.get("schema") != SELECTION_ROW_SCHEMA
                or row.get("row_sha256") != canonical_sha256(unsigned)
                or not isinstance(barcode, str)
                or not barcode
                or barcode in selected
                or row.get("language_gen") != "eng"
                or row.get("rights_code") not in ALLOWED_RIGHTS_CODES
                or row.get("training_ready") is not False
            ):
                raise InstitutionalBooksTransientTokenizerStreamError(
                    "book selection row differs"
                )
            selected[barcode] = row
            ordered.append(row["row_sha256"])
    if len(selected) != descriptor.get("rows") or canonical_sha256(
        ordered
    ) != descriptor.get("ordered_rows_sha256"):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "book selection coverage differs"
        )
    return selected, receipt


def _book_domain(domains: list[str]) -> str:
    key = " ".join(domains).casefold()
    if any(marker in key for marker in ("math", "algebra", "geometry", "statistics")):
        return "math"
    if any(
        marker in key
        for marker in (
            "physics",
            "chemistry",
            "biology",
            "astronomy",
            "medicine",
            "health",
            "science",
        )
    ):
        return "science"
    if any(
        marker in key
        for marker in (
            "computer",
            "software",
            "technical",
            "technology",
            "engineering",
        )
    ):
        return "technical"
    return "english"


def _envelope(
    row: dict[str, Any],
    selection: dict[str, Any],
    shard_receipt_sha256: str,
) -> dict[str, Any]:
    text = row.get("text")
    curriculum_json = row.get("curriculum_metadata_json")
    domains = row.get("semantic_domains")
    if (
        row.get("schema") != OUTPUT_SCHEMA
        or row.get("training_ready") is not False
        or row.get("corpus_split") != "train"
        or not isinstance(text, str)
        or not text
        or hashlib.sha256(text.encode()).hexdigest() != row.get("content_sha256")
        or not isinstance(curriculum_json, str)
        or not isinstance(domains, list)
        or not domains
        or any(not isinstance(domain, str) or not domain for domain in domains)
        or row.get("barcode_src") != selection.get("barcode_src")
        or row.get("selection_row_sha256") != selection.get("row_sha256")
    ):
        raise InstitutionalBooksTransientTokenizerStreamError("final book row differs")
    try:
        curriculum = json.loads(curriculum_json)
    except json.JSONDecodeError as error:
        raise InstitutionalBooksTransientTokenizerStreamError(
            "book curriculum metadata differs"
        ) from error
    unsigned_curriculum = (
        {key: value for key, value in curriculum.items() if key != "metadata_sha256"}
        if isinstance(curriculum, dict)
        else {}
    )
    if (
        not isinstance(curriculum, dict)
        or curriculum.get("metadata_sha256") != row.get("curriculum_metadata_sha256")
        or curriculum.get("metadata_sha256") != canonical_sha256(unsigned_curriculum)
    ):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "book curriculum metadata differs"
        )
    quality = curriculum.get("quality_floor") if isinstance(curriculum, dict) else None
    complexity = (
        curriculum.get("complexity_range") if isinstance(curriculum, dict) else None
    )
    phases = (
        curriculum.get("curriculum_band_votes")
        if isinstance(curriculum, dict)
        else None
    )
    concepts = (
        curriculum.get("shared_concepts") if isinstance(curriculum, dict) else None
    )
    prerequisites = (
        curriculum.get("shared_prerequisites") if isinstance(curriculum, dict) else None
    )
    if (
        not isinstance(quality, dict)
        or not quality
        or any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5
            for value in quality.values()
        )
        or not isinstance(complexity, dict)
        or not complexity
        or not isinstance(phases, list)
        or not phases
        or any(not isinstance(value, str) or not value for value in phases)
        or not isinstance(concepts, list)
        or not isinstance(prerequisites, list)
    ):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "book curriculum metadata differs"
        )
    maximum_complexity = max(value["maximum"] for value in complexity.values())
    rights_code = selection["rights_code"]
    evidence = canonical_sha256(
        {
            "final_shard_receipt_sha256": shard_receipt_sha256,
            "selection_row_sha256": selection["row_sha256"],
            "quality_agreement_record_sha256": row["quality_agreement_record_sha256"],
            "benchmark_decontamination_record_sha256": row[
                "benchmark_decontamination_record_sha256"
            ],
            "cross_source_subdocument_transform_sha256": row[
                "cross_source_subdocument_transform_sha256"
            ],
            "content_sha256": row["content_sha256"],
        }
    )
    document = normalize_document(
        {
            "schema": ROW_SCHEMA,
            "text": text,
            "source": {
                "dataset": f"{ENRICHED_REPOSITORY}@{ENRICHED_REVISION}",
                "row_id": row["barcode_src"],
                "license": RIGHTS_LABELS[rights_code],
                "domain": _book_domain(domains),
            },
            "verification": {
                "benchmark_disjoint": True,
                "evidence_sha256": evidence,
            },
        }
    )
    result = {
        "schema": INSTITUTIONAL_BOOK_ENVELOPE_SCHEMA,
        "document": document,
        "corpus_split": "train",
        "semantic_curriculum_phase": "+".join(sorted(set(phases))),
        "semantic_difficulty_mean_milli": (maximum_complexity - 1) * 1_000,
        "semantic_prerequisite_burden_mean_milli": min(4_000, len(prerequisites) * 500),
        "semantic_quality_floor_milli": min(quality.values()) * 2_000,
        "semantic_domains": sorted(set(domains)),
        "semantic_recurring_concepts": sorted(set(concepts)),
        "semantic_recurring_prerequisites": sorted(set(prerequisites)),
        "final_locator_sha256": evidence,
        "tokenization_ready": True,
        "training_ready": False,
    }
    result["envelope_sha256"] = canonical_sha256(result)
    return result


def stream_shard(
    final_root: Path,
    selection_root: Path,
    output: TextIO,
    receipt_path: Path,
    *,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Stream one final train partition and persist only a source-text-free receipt."""

    if (
        receipt_path.exists()
        or receipt_path.is_symlink()
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "book transient stream arguments differ"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksTransientTokenizerStreamError(
            "pyarrow is required"
        ) from error
    aggregate = _load_signed(final_root / "aggregate.json", AGGREGATE_SCHEMA)
    if (
        aggregate.get("status")
        != "complete_nontraining_institutional_books_cross_source_rewritten"
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("complete_benchmark_disjoint_book_coverage") is not True
        or aggregate.get("private_storage_only") is not True
        or aggregate.get("benchmark_decontamination_complete") is not True
        or aggregate.get("cross_source_subdocument_deduplication_complete") is not True
    ):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "final book aggregate differs"
        )
    receipts = []
    shard_receipt = None
    for index in range(logical_shards):
        value = _load_signed(
            final_root / "shards" / f"shard_{index:05d}" / "receipt.json",
            SHARD_SCHEMA,
        )
        receipts.append(value["receipt_sha256"])
        if index == shard_index:
            shard_receipt = value
    if shard_receipt is None or canonical_sha256(receipts) != aggregate.get(
        "shards", {}
    ).get("ordered_receipts_sha256"):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "final book shard custody differs"
        )
    selected, selection_receipt = _selection(selection_root)
    descriptor = shard_receipt.get("outputs", {}).get("train")
    root = final_root / "shards" / f"shard_{shard_index:05d}"
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("path") != "train.parquet"
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
        or descriptor.get("rows")
        != shard_receipt.get("counts", {}).get("split::train::documents")
    ):
        raise InstitutionalBooksTransientTokenizerStreamError(
            "final book train partition differs"
        )
    digest = hashlib.sha256()
    envelope_digests = hashlib.sha256()
    counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, use_threads=False):
        for row in batch.to_pylist():
            selection = selected.get(row.get("barcode_src"))
            if selection is None:
                raise InstitutionalBooksTransientTokenizerStreamError(
                    "final book selection binding differs"
                )
            envelope = _envelope(row, selection, shard_receipt["receipt_sha256"])
            encoded = (
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.write(encoded)
            digest.update(encoded.encode())
            envelope_digests.update(bytes.fromhex(envelope["envelope_sha256"]))
            counts["documents"] += 1
            counts["text_utf8_bytes"] += len(row["text"].encode())
            counts[
                f"domain::{envelope['document']['source']['domain']}::documents"
            ] += 1
            counts[f"rights_code::{selection['rights_code']}::documents"] += 1
    if counts["documents"] != descriptor.get("rows") or not counts["documents"]:
        raise InstitutionalBooksTransientTokenizerStreamError(
            "final book stream coverage differs"
        )
    output.flush()
    payload = {
        "schema": GENERIC_SOURCE_RECEIPT_SCHEMA,
        "status": GENERIC_SOURCE_STATUS,
        "source": {
            "final_book_aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "final_book_shard_receipt_sha256": shard_receipt["receipt_sha256"],
            "selection_receipt_sha256": selection_receipt["receipt_sha256"],
        },
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "counts": dict(sorted(counts.items())),
        "ordered_jsonl_sha256": digest.hexdigest(),
        "ordered_envelope_digests_sha256": envelope_digests.hexdigest(),
        "source_text_persisted_by_compiler": False,
        "rights_evidence_bound": True,
        "development_partition_excluded": True,
        "tokenization_ready": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(receipt_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    result = stream_shard(
        args.final_root,
        args.selection_root,
        sys.stdout,
        args.receipt,
        logical_shards=args.logical_shards,
        shard_index=args.shard_index,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
