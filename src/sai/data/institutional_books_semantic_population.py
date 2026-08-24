"""Build a private, diverse Institutional Books semantic-review population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books import (
    METADATA_PARQUET_BYTES,
    METADATA_PARQUET_SHA256,
    TEXT_FIELD,
    build_book_candidate,
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
from sai.data.institutional_books_quality_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from sai.data.institutional_books_quality_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-semantic-candidate-population-v1"
DEFAULT_MAXIMUM_CANDIDATES = 8_192
DEFAULT_SEED = "sai-institutional-books-diverse-semantic-population-20260826-r1"
POLICY_SCHEMA = "sai-institutional-books-diverse-selection-policy-v1"
CANDIDATE_METADATA_COLUMNS = {
    "barcode_src",
    "title_src",
    "author_src",
    "date1_src",
    "date2_src",
    "page_count_src",
    "token_count_o200k_base_gen",
    "language_src",
    "language_gen",
    "topic_or_subject_src",
    "topic_or_subject_gen",
    "genre_or_form_src",
    "general_note_src",
    "ocr_score_src",
    "ocr_score_gen",
    "likely_duplicates_barcodes_gen",
    "identifiers_src",
    "hathitrust_data_ext",
}


class InstitutionalBooksSemanticPopulationError(RuntimeError):
    """Filtered text, metadata, selection, or population custody differs."""


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksSemanticPopulationError("population output exists")
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


def _label(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.split(r"(?:--|[;|/])", normalized, maxsplit=1)[0]
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return " ".join(normalized.split()[:8])[:96] or "unknown"


def _token_band(tokens: int) -> str:
    if tokens < 20_000:
        return "short_lt_20k"
    if tokens < 80_000:
        return "medium_20k_80k"
    if tokens < 250_000:
        return "long_80k_250k"
    return "very_long_ge_250k"


def _rank(seed: str, value: Any) -> str:
    return hashlib.sha256((seed + "\0" + canonical_sha256(value)).encode()).hexdigest()


def select_diverse_barcodes(
    rows: list[dict[str, Any]], maximum_candidates: int, seed: str
) -> tuple[list[str], dict[str, Any]]:
    """Select stable round-robin coverage across subject, genre, and length."""

    if (
        not rows
        or isinstance(maximum_candidates, bool)
        or maximum_candidates <= 0
        or not isinstance(seed, str)
        or not seed
    ):
        raise InstitutionalBooksSemanticPopulationError("selection geometry differs")
    groups: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        barcode = row.get("barcode_src")
        tokens = row.get("token_count_o200k_base_gen")
        if (
            not isinstance(barcode, str)
            or not barcode
            or barcode in seen
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            raise InstitutionalBooksSemanticPopulationError(
                "selection candidate differs"
            )
        seen.add(barcode)
        stratum = (
            _label(row.get("topic_or_subject_gen")),
            _label(row.get("genre_or_form_src")),
            _token_band(tokens),
        )
        groups[stratum].append((_rank(seed, barcode), barcode))
    queues = {
        key: deque(barcode for _, barcode in sorted(values))
        for key, values in groups.items()
    }
    ordered_strata = sorted(groups, key=lambda key: _rank(seed, list(key)))
    selected: list[str] = []
    target = min(maximum_candidates, len(rows))
    while len(selected) < target:
        advanced = False
        for key in ordered_strata:
            if queues[key] and len(selected) < target:
                selected.append(queues[key].popleft())
                advanced = True
        if not advanced:
            raise InstitutionalBooksSemanticPopulationError(
                "diverse selection terminated early"
            )
    barcode_to_stratum = {
        barcode: key for key, values in groups.items() for _, barcode in values
    }
    selected_counts = Counter(barcode_to_stratum[barcode] for barcode in selected)
    statistics = {
        "eligible_rows": len(rows),
        "eligible_strata": len(groups),
        "selected_rows": len(selected),
        "selected_strata": len(selected_counts),
        "selected_token_bands": dict(
            sorted(Counter(key[2] for key in selected_counts.elements()).items())
        ),
        "ordered_selected_barcodes_sha256": canonical_sha256(selected),
    }
    return selected, statistics


def _selection_rows(
    root: Path, retained_barcodes: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(root / "receipt.json")
    path = root / "selection.jsonl"
    if not _valid_receipt(receipt, SELECTION_SCHEMA) or receipt.get(
        "selection", {}
    ).get("sha256") != sha256_file(path):
        raise InstitutionalBooksSemanticPopulationError("strict selection differs")
    found = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
            barcode = row.get("barcode_src")
            if (
                row.get("schema") != SELECTION_ROW_SCHEMA
                or not isinstance(barcode, str)
                or barcode in seen
                or row.get("row_sha256") != canonical_sha256(unsigned)
                or row.get("training_ready") is not False
            ):
                raise InstitutionalBooksSemanticPopulationError(
                    "strict selection row differs"
                )
            seen.add(barcode)
            if barcode in retained_barcodes:
                found.append(row)
    if (
        len(seen) != receipt.get("selection", {}).get("rows")
        or {row["barcode_src"] for row in found} != retained_barcodes
    ):
        raise InstitutionalBooksSemanticPopulationError(
            "filtered-to-selection coverage differs"
        )
    return found, receipt


def _filtered_index(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    aggregate = _load_json(root / "aggregate.json")
    if not _valid_receipt(aggregate, FILTER_AGGREGATE_SCHEMA):
        raise InstitutionalBooksSemanticPopulationError("filter aggregate differs")
    logical_shards = aggregate.get("shards", {}).get("logical_shards")
    if isinstance(logical_shards, bool) or not isinstance(logical_shards, int):
        raise InstitutionalBooksSemanticPopulationError("filter shard count differs")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksSemanticPopulationError(
            "pyarrow is required"
        ) from error
    rows: dict[str, dict[str, Any]] = {}
    ordered_receipts = []
    for index in range(logical_shards):
        shard_root = root / "shards" / f"shard_{index:05d}"
        receipt = _load_json(shard_root / "receipt.json")
        if (
            not _valid_receipt(receipt, FILTER_SHARD_SCHEMA)
            or receipt.get("shard_index") != index
            or receipt.get("logical_shards") != logical_shards
        ):
            raise InstitutionalBooksSemanticPopulationError(
                "filter shard receipt differs"
            )
        ordered_receipts.append(receipt["receipt_sha256"])
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
            raise InstitutionalBooksSemanticPopulationError(
                "filtered book bytes differ"
            )
        parquet = pq.ParquetFile(path)
        required = {
            "schema",
            "barcode_src",
            "source_content_sha256",
            "metadata_token_count_o200k_base_gen",
            "training_ready",
        }
        if not required.issubset(parquet.schema_arrow.names):
            raise InstitutionalBooksSemanticPopulationError(
                "filtered book columns differ"
            )
        shard_rows = 0
        for batch in parquet.iter_batches(
            batch_size=4_096, columns=sorted(required), use_threads=False
        ):
            for row in batch.to_pylist():
                barcode = row.get("barcode_src")
                if (
                    row.get("schema") != OUTPUT_SCHEMA
                    or not isinstance(barcode, str)
                    or barcode in rows
                    or not isinstance(row.get("source_content_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", row["source_content_sha256"])
                    is None
                    or row.get("training_ready") is not False
                ):
                    raise InstitutionalBooksSemanticPopulationError(
                        "filtered book row differs"
                    )
                rows[barcode] = row
                shard_rows += 1
        if shard_rows != descriptor.get("rows"):
            raise InstitutionalBooksSemanticPopulationError(
                "filtered book row coverage differs"
            )
    if (
        canonical_sha256(ordered_receipts)
        != aggregate.get("shards", {}).get("ordered_receipts_sha256")
        or len(rows) != aggregate.get("counts", {}).get("retained_rows")
        or canonical_sha256(sorted(rows))
        != aggregate.get("ordered_retained_barcodes_sha256")
    ):
        raise InstitutionalBooksSemanticPopulationError(
            "filtered population accounting differs"
        )
    return rows, aggregate


def _selected_candidates(
    root: Path,
    selected_order: list[str],
    metadata: dict[str, dict[str, Any]],
    filter_receipt_sha256: str,
) -> list[dict[str, Any]]:
    """Replay private Parquet and retain only bounded representative excerpts."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksSemanticPopulationError(
            "pyarrow is required"
        ) from error
    selected = set(selected_order)
    candidates: dict[str, dict[str, Any]] = {}
    logical_shards = _load_json(root / "aggregate.json")["shards"]["logical_shards"]
    for index in range(logical_shards):
        shard_root = root / "shards" / f"shard_{index:05d}"
        receipt = _load_json(shard_root / "receipt.json")
        descriptor = receipt.get("output")
        if descriptor is None:
            continue
        parquet = pq.ParquetFile(shard_root / descriptor["path"])
        for batch in parquet.iter_batches(batch_size=16, use_threads=False):
            for row in batch.to_pylist():
                barcode = row.get("barcode_src")
                if barcode not in selected:
                    continue
                text = row.get("text")
                if (
                    barcode in candidates
                    or not isinstance(text, str)
                    or hashlib.sha256(text.encode()).hexdigest()
                    != row.get("source_content_sha256")
                    or row.get("training_ready") is not False
                ):
                    raise InstitutionalBooksSemanticPopulationError(
                        "selected filtered book row differs"
                    )
                enriched = {key: value for key, value in row.items() if key != "text"}
                enriched[TEXT_FIELD] = text
                enriched["semantic_population_filter_receipt_sha256"] = (
                    filter_receipt_sha256
                )
                candidates[barcode] = build_book_candidate(metadata[barcode], enriched)
    if set(candidates) != selected:
        raise InstitutionalBooksSemanticPopulationError(
            "selected filtered text coverage differs"
        )
    return [candidates[barcode] for barcode in selected_order]


def _metadata_rows(
    path: Path, selected: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != METADATA_PARQUET_BYTES
        or sha256_file(path) != METADATA_PARQUET_SHA256
    ):
        raise InstitutionalBooksSemanticPopulationError("book metadata differs")
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksSemanticPopulationError(
            "pyarrow is required"
        ) from error
    table = pq.read_table(path)
    table = table.filter(
        pc.is_in(table["barcode_src"], value_set=pa.array(sorted(selected)))
    )
    rows = {}
    for row in table.to_pylist():
        barcode = row.get("barcode_src")
        selection = selected.get(barcode)
        if (
            not isinstance(barcode, str)
            or barcode in rows
            or selection is None
            or canonical_sha256(row) != selection.get("metadata_row_sha256")
            or not CANDIDATE_METADATA_COLUMNS.issubset(row)
        ):
            raise InstitutionalBooksSemanticPopulationError(
                "book metadata identity differs"
            )
        rows[barcode] = {key: row[key] for key in sorted(CANDIDATE_METADATA_COLUMNS)}
    if set(rows) != set(selected):
        raise InstitutionalBooksSemanticPopulationError(
            "selected book metadata coverage differs"
        )
    return rows


def build_population(
    filtered_root: Path,
    selection_root: Path,
    metadata_path: Path,
    output_root: Path,
    maximum_candidates: int = DEFAULT_MAXIMUM_CANDIDATES,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build an exact private candidate population; admit no training text."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksSemanticPopulationError("population root exists")
    filtered, aggregate = _filtered_index(filtered_root)
    selection, selection_receipt = _selection_rows(selection_root, set(filtered))
    selected_order, statistics = select_diverse_barcodes(
        selection, maximum_candidates, seed
    )
    selected = set(selected_order)
    selection_by_barcode = {
        row["barcode_src"]: row for row in selection if row["barcode_src"] in selected
    }
    metadata = _metadata_rows(metadata_path, selection_by_barcode)
    candidates = _selected_candidates(
        filtered_root, selected_order, metadata, aggregate["receipt_sha256"]
    )
    selected_tokens = sum(
        metadata[barcode]["token_count_o200k_base_gen"] for barcode in selected_order
    )
    policy = {
        "schema": POLICY_SCHEMA,
        "seed": seed,
        "maximum_candidates": maximum_candidates,
        "method": "stable_round_robin_subject_genre_token_band",
        "subject_normalization": "nfkc_casefold_primary_segment_first_8_tokens",
        "token_bands": [20_000, 80_000, 250_000],
        "mechanical_pass_required": True,
        "semantic_judgment_not_used_for_selection": True,
    }
    output_root.mkdir(parents=True)
    try:
        candidate_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidate_path, candidates)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_private_semantic_candidate_population",
            "source": {
                "filtered_aggregate_receipt_sha256": aggregate["receipt_sha256"],
                "filtered_rows": len(filtered),
                "selection_receipt_sha256": selection_receipt["receipt_sha256"],
                "metadata_sha256": METADATA_PARQUET_SHA256,
            },
            "policy": policy,
            "policy_sha256": canonical_sha256(policy),
            "statistics": {
                **statistics,
                "selected_tokens": selected_tokens,
            },
            "output": {
                "path": candidate_path.name,
                "rows": len(candidates),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_candidate_identities_sha256": canonical_sha256(
                    [row["candidate_identity_sha256"] for row in candidates]
                ),
            },
            "source_text_private": True,
            "source_text_publishable": False,
            "semantic_admission_complete": False,
            "rights_review_complete": False,
            "benchmark_decontamination_complete": False,
            "global_semantic_deduplication_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        for child in output_root.iterdir():
            child.unlink(missing_ok=True)
        output_root.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--maximum-candidates", type=int, default=DEFAULT_MAXIMUM_CANDIDATES
    )
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    result = build_population(
        args.filtered_root,
        args.selection_root,
        args.metadata,
        args.output_root,
        args.maximum_candidates,
        args.seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
