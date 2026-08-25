"""Build the source-text-free production curriculum index for Sai 1B."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_component_admission import SCHEMA as BRIDGE_ADMISSION_SCHEMA
from sai.data.common_pile_stack_edu_practical_admission import (
    SCHEMA as CODE_ADMISSION_SCHEMA,
)
from sai.data.institutional_books_practical_admission import (
    RECORD_SCHEMA as BOOK_RECORD_SCHEMA,
)
from sai.data.institutional_books_practical_admission import (
    SCHEMA as BOOK_ADMISSION_SCHEMA,
)
from sai.data.institutional_books_practical_admission import (
    SELECTION_ROW_SCHEMA,
    SELECTION_SCHEMA,
)
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_ADMISSION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-1b-curriculum-index-row-v1"
SHARD_SCHEMA = "sai-1b-curriculum-index-shard-v1"
AGGREGATE_SCHEMA = "sai-1b-curriculum-index-aggregate-v1"
BANDS = ("foundation", "intermediate", "advanced", "expert")
COMPONENTS = ("books", "pleias", "code", "connections")
DEVELOPMENT_FRACTION_PPM = 1_000
_WORLD_HISTORY = (
    "WORLD HISTORY AND HISTORY OF EUROPE, ASIA, AFRICA, AUSTRALIA, "
    "NEW ZEALAND, ETC."
)

_BOOK_BASE = {
    "EDUCATION": 0,
    "GENERAL WORKS": 0,
    "LANGUAGE AND LITERATURE": 1,
    "FINE ARTS": 1,
    "GEOGRAPHY. ANTHROPOLOGY. RECREATION": 1,
    "HISTORY OF THE AMERICAS": 1,
    _WORLD_HISTORY: 1,
    "MUSIC AND BOOKS ON MUSIC": 1,
    "AGRICULTURE": 1,
    "AUXILIARY SCIENCES OF HISTORY": 1,
    "BIBLIOGRAPHY. LIBRARY SCIENCE. INFORMATION RESOURCES (GENERAL)": 1,
    "SOCIAL SCIENCES": 1,
    "PHILOSOPHY. PSYCHOLOGY. RELIGION": 2,
    "POLITICAL SCIENCE": 2,
    "LAW": 2,
    "SCIENCE": 2,
    "MEDICINE": 2,
    "TECHNOLOGY": 2,
    "NAVAL SCIENCE": 2,
    "MILITARY SCIENCE": 2,
}
_PLEIAS_EXPERT = ("arxiv", "uspto")
_PLEIAS_ADVANCED = (
    "openalex",
    "stackexchange",
    "case law",
    "court",
    "law",
    "legal",
    "sec",
    "dockets",
    "eurlex",
    "ecfr",
    "reg_docs",
    "usc",
)
_PLEIAS_FOUNDATION = (
    "wikipedia",
    "gutenberg",
    "librilight",
    "youtube-commons",
    "books",
)


class OneBCurriculumIndexError(RuntimeError):
    """A source receipt, locator, band, split, or index identity differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise OneBCurriculumIndexError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBCurriculumIndexError("signed input differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != schema or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise OneBCurriculumIndexError("signed input differs")
    return payload


def _hex(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OneBCurriculumIndexError(f"{field} differs")
    return value


def _split(identity: str, *, bulk: bool = True) -> str:
    if not bulk:
        return "train"
    return (
        "development"
        if int(identity[:16], 16) % 1_000_000 < DEVELOPMENT_FRACTION_PPM
        else "train"
    )


def _book_band(topic: str, tokens: int) -> tuple[str, int]:
    base = _BOOK_BASE.get(topic, 1)
    adjustment = -1 if tokens < 25_000 else (1 if tokens >= 200_000 else 0)
    value = min(3, max(0, base + adjustment))
    return BANDS[value], value * 1_000 + 500


def _pleias_band(collection: str, word_count: int, token_count: int) -> tuple[str, int]:
    folded = collection.casefold()
    if any(pattern in folded for pattern in _PLEIAS_EXPERT):
        base = 3
    elif any(pattern in folded for pattern in _PLEIAS_ADVANCED):
        base = 2
    elif any(pattern in folded for pattern in _PLEIAS_FOUNDATION):
        base = 0
    else:
        base = 1
    ratio_milli = token_count * 1_000 // max(1, word_count)
    adjustment = 1 if ratio_milli >= 1_650 or word_count >= 1_500 else 0
    if ratio_milli <= 1_100 or word_count < 100:
        adjustment = -1
    value = min(3, max(0, base + adjustment))
    difficulty = value * 1_000 + min(999, max(0, ratio_milli - 900))
    return BANDS[value], difficulty


def _priority(component: str, identity: str) -> str:
    return hashlib.sha256(
        f"sai-1b-curriculum-v1\0{component}\0{identity}".encode()
    ).hexdigest()


def _index_schema() -> Any:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise OneBCurriculumIndexError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("component", pa.string()),
            ("source_shard", pa.int32()),
            ("document_identity_sha256", pa.string()),
            ("content_sha256", pa.string()),
            ("source_path", pa.string()),
            ("source_row_index", pa.int64()),
            ("text_utf8_bytes", pa.int64()),
            ("source_token_estimate", pa.int64()),
            ("curriculum_band", pa.string()),
            ("difficulty_milli", pa.int32()),
            ("split", pa.string()),
            ("domain", pa.string()),
            ("source_bucket", pa.string()),
            ("curriculum_priority_sha256", pa.string()),
        ]
    )


def _write_index(rows: Iterable[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBCurriculumIndexError("pyarrow is required") from error
    if output_root.exists() or output_root.is_symlink():
        raise OneBCurriculumIndexError("index output exists")
    output_root.mkdir(parents=True)
    temporary = output_root / f".index.partial.{os.getpid()}.parquet"
    output = output_root / "index.parquet"
    writer = pq.ParquetWriter(temporary, _index_schema(), compression="zstd")
    pending: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    ordered = hashlib.sha256()
    identities = set()
    try:
        for row in rows:
            identity = _hex(row.get("document_identity_sha256"), "document identity")
            if identity in identities:
                raise OneBCurriculumIndexError("index identity overlaps")
            identities.add(identity)
            if row.get("curriculum_band") not in BANDS or row.get("split") not in {
                "train",
                "development",
            }:
                raise OneBCurriculumIndexError("index band or split differs")
            ordered.update(canonical_sha256(row).encode())
            counts["rows"] += 1
            counts["text_utf8_bytes"] += row["text_utf8_bytes"]
            counts["source_token_estimate"] += row["source_token_estimate"]
            counts[f"band::{row['curriculum_band']}::rows"] += 1
            counts[f"band::{row['curriculum_band']}::bytes"] += row["text_utf8_bytes"]
            counts[f"split::{row['split']}::rows"] += 1
            counts[f"split::{row['split']}::bytes"] += row["text_utf8_bytes"]
            pending.append(row)
            if len(pending) >= 8_192:
                writer.write_table(
                    pa.Table.from_pylist(pending, schema=_index_schema())
                )
                pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=_index_schema()))
        writer.close()
        os.replace(temporary, output)
    except BaseException:
        writer.close()
        raise
    # Empty identities are part of a fixed shard namespace.  Preserve their
    # custody with explicit zero counters instead of making downstream replay
    # infer that a missing key means zero.
    for key in ("rows", "text_utf8_bytes", "source_token_estimate"):
        counts.setdefault(key, 0)
    for band in BANDS:
        counts.setdefault(f"band::{band}::rows", 0)
        counts.setdefault(f"band::{band}::bytes", 0)
    for split in ("train", "development"):
        counts.setdefault(f"split::{split}::rows", 0)
        counts.setdefault(f"split::{split}::bytes", 0)
    return {
        "counts": dict(sorted(counts.items())),
        "ordered_rows_sha256": ordered.hexdigest(),
        "output": {
            "path": output.name,
            "rows": counts["rows"],
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def _seal_shard(
    output_root: Path,
    component: str,
    source_shard: int,
    source_receipt_sha256: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_1b_curriculum_index_shard",
        "component": component,
        "source_shard": source_shard,
        "source_receipt_sha256": source_receipt_sha256,
        **result,
        "source_text_persisted": False,
        "bulk_internal_development_fraction_ppm": (
            DEVELOPMENT_FRACTION_PPM if component != "connections" else 0
        ),
        "curriculum_index_ready": True,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def index_books(
    admission_root: Path, selection_root: Path, output_root: Path
) -> dict[str, Any]:
    """Join admitted Books to its exact topic metadata and assign bands."""

    admission = _load_signed(admission_root / "receipt.json", BOOK_ADMISSION_SCHEMA)
    selection = _load_signed(selection_root / "receipt.json", SELECTION_SCHEMA)
    if (
        admission.get("training_ready") is not True
        or admission.get("selection_receipt_sha256") != selection["receipt_sha256"]
    ):
        raise OneBCurriculumIndexError("book admission differs")
    selection_path = selection_root / selection.get("selection", {}).get("path", "")
    topics: dict[str, tuple[str, int]] = {}
    with selection_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema") != SELECTION_ROW_SCHEMA:
                raise OneBCurriculumIndexError("book selection row differs")
            topics[row["row_sha256"]] = (
                row.get("topic_or_subject_gen") or "GENERAL WORKS",
                row["token_count_o200k_base_gen"],
            )
    manifest = admission_root / admission.get("manifest", {}).get("path", "")

    def rows() -> Iterable[dict[str, Any]]:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                metadata = topics.get(row.get("selection_row_sha256"))
                if row.get("schema") != BOOK_RECORD_SCHEMA or metadata is None:
                    raise OneBCurriculumIndexError("book admission row differs")
                topic, source_tokens = metadata
                band, difficulty = _book_band(topic, source_tokens)
                identity = _hex(row["record_sha256"], "book record identity")
                yield {
                    "schema": ROW_SCHEMA,
                    "component": "books",
                    "source_shard": row["shard_index"],
                    "document_identity_sha256": identity,
                    "content_sha256": _hex(
                        row["source_content_sha256"], "book content"
                    ),
                    "source_path": row["private_filtered_relative_path"],
                    "source_row_index": -1,
                    "text_utf8_bytes": row["source_text_utf8_bytes"],
                    "source_token_estimate": row["enriched_token_count_gen"],
                    "curriculum_band": band,
                    "difficulty_milli": difficulty,
                    "split": _split(identity),
                    "domain": topic.casefold().replace(" ", "_"),
                    "source_bucket": topic,
                    "curriculum_priority_sha256": _priority("books", identity),
                }

    result = _write_index(rows(), output_root)
    if result["counts"]["rows"] != admission["counts"]["admitted_rows"]:
        raise OneBCurriculumIndexError("book index coverage differs")
    return _seal_shard(output_root, "books", -1, admission["receipt_sha256"], result)


def _descriptor(
    admission: dict[str, Any], shard_index: int, *, allow_empty: bool = False
) -> dict[str, Any] | None:
    values = [
        row
        for row in admission.get("outputs", {}).get("descriptors", [])
        if row.get("shard_index") == shard_index
    ]
    if allow_empty and not values:
        return None
    if len(values) != 1:
        raise OneBCurriculumIndexError("source descriptor differs")
    return values[0]


def index_pleias_shard(
    admission_root: Path, shard_index: int, output_root: Path
) -> dict[str, Any]:
    """Assign practical PleIAs locators to deterministic spiral bands."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBCurriculumIndexError("pyarrow is required") from error
    admission = _load_signed(admission_root / "receipt.json", PLEIAS_ADMISSION_SCHEMA)
    descriptor = _descriptor(admission, shard_index)
    if descriptor is None:  # pragma: no cover - impossible without allow_empty
        raise OneBCurriculumIndexError("PleIAs descriptor differs")
    path = admission_root / descriptor["path"]
    if (
        path.stat().st_size != descriptor["bytes"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise OneBCurriculumIndexError("PleIAs descriptor bytes differ")

    def rows() -> Iterable[dict[str, Any]]:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8_192, use_threads=False
        ):
            for row in batch.to_pylist():
                identity = _hex(row["source_row_identity_sha256"], "PleIAs identity")
                band, difficulty = _pleias_band(
                    row["collection"], row["word_count"], row["source_token_count"]
                )
                yield {
                    "schema": ROW_SCHEMA,
                    "component": "pleias",
                    "source_shard": shard_index,
                    "document_identity_sha256": identity,
                    "content_sha256": _hex(row["content_sha256"], "PleIAs content"),
                    "source_path": row["source_path"],
                    "source_row_index": row["source_row_index"],
                    "text_utf8_bytes": row["text_utf8_bytes"],
                    "source_token_estimate": row["source_token_count"],
                    "curriculum_band": band,
                    "difficulty_milli": difficulty,
                    "split": _split(identity),
                    "domain": row["open_type"].casefold().replace(" ", "_"),
                    "source_bucket": row["collection"],
                    "curriculum_priority_sha256": _priority("pleias", identity),
                }

    result = _write_index(rows(), output_root)
    if result["counts"]["rows"] != descriptor["rows"]:
        raise OneBCurriculumIndexError("PleIAs index coverage differs")
    return _seal_shard(
        output_root, "pleias", shard_index, admission["receipt_sha256"], result
    )


def index_code_shard(
    admission_root: Path, shard_index: int, output_root: Path
) -> dict[str, Any]:
    """Assign admitted Stack-Edu locators to advanced or expert bands."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBCurriculumIndexError("pyarrow is required") from error
    admission = _load_signed(admission_root / "receipt.json", CODE_ADMISSION_SCHEMA)
    descriptor = _descriptor(admission, shard_index, allow_empty=True)
    path = admission_root / descriptor["path"] if descriptor is not None else None
    if descriptor is not None and path is not None and (
        path.stat().st_size != descriptor["bytes"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise OneBCurriculumIndexError("code descriptor bytes differ")

    def rows() -> Iterable[dict[str, Any]]:
        if descriptor is None or path is None:
            return
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=8_192, use_threads=False
        ):
            for row in batch.to_pylist():
                identity = _hex(row["source_row_identity_sha256"], "code identity")
                value = 2 if row["integer_score"] == 3 else 3
                yield {
                    "schema": ROW_SCHEMA,
                    "component": "code",
                    "source_shard": shard_index,
                    "document_identity_sha256": identity,
                    "content_sha256": _hex(row["content_sha256"], "code content"),
                    "source_path": row["source_path"],
                    "source_row_index": row["source_row_index"],
                    "text_utf8_bytes": row["text_utf8_bytes"],
                    "source_token_estimate": max(1, (row["text_utf8_bytes"] + 3) // 4),
                    "curriculum_band": BANDS[value],
                    "difficulty_milli": value * 1_000 + row["integer_score"] * 100,
                    "split": _split(identity),
                    "domain": f"code::{row['language'].casefold()}",
                    "source_bucket": row["language"],
                    "curriculum_priority_sha256": _priority("code", identity),
                }

    result = _write_index(rows(), output_root)
    expected_rows = descriptor["rows"] if descriptor is not None else 0
    if result["counts"].get("rows", 0) != expected_rows:
        raise OneBCurriculumIndexError("code index coverage differs")
    return _seal_shard(
        output_root, "code", shard_index, admission["receipt_sha256"], result
    )


def index_connections(admission_root: Path, output_root: Path) -> dict[str, Any]:
    """Index only the already admitted train connection component."""

    admission = _load_signed(admission_root / "receipt.json", BRIDGE_ADMISSION_SCHEMA)
    descriptor = admission.get("train", {})
    path = admission_root / descriptor.get("path", "")
    if path.stat().st_size != descriptor.get("bytes") or sha256_file(
        path
    ) != descriptor.get("sha256"):
        raise OneBCurriculumIndexError("connection train bytes differ")

    def rows() -> Iterable[dict[str, Any]]:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    row.get("corpus_split") != "train"
                    or row.get("training_ready") is not True
                ):
                    raise OneBCurriculumIndexError("connection row differs")
                identity = _hex(row["document_identity_sha256"], "connection identity")
                difficulty = row["difficulty_milli"]
                value = min(3, max(0, (difficulty - 1) // 1_000))
                domains = sorted(row.get("semantic_domains", []))
                yield {
                    "schema": ROW_SCHEMA,
                    "component": "connections",
                    "source_shard": 0,
                    "document_identity_sha256": identity,
                    "content_sha256": _hex(row["content_sha256"], "connection content"),
                    "source_path": descriptor["path"],
                    "source_row_index": row["document_index"],
                    "text_utf8_bytes": row["text_utf8_bytes"],
                    "source_token_estimate": max(1, (row["text_utf8_bytes"] + 3) // 4),
                    "curriculum_band": BANDS[value],
                    "difficulty_milli": difficulty,
                    "split": _split(identity, bulk=False),
                    "domain": "×".join(domains),
                    "source_bucket": row["document_type"],
                    "curriculum_priority_sha256": _priority("connections", identity),
                }

    result = _write_index(rows(), output_root)
    if result["counts"]["rows"] != admission["counts"]["train_documents"]:
        raise OneBCurriculumIndexError("connection index coverage differs")
    return _seal_shard(
        output_root, "connections", 0, admission["receipt_sha256"], result
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    books = commands.add_parser("books")
    books.add_argument("--admission-root", type=Path, required=True)
    books.add_argument("--selection-root", type=Path, required=True)
    books.add_argument("--output-root", type=Path, required=True)
    for name in ("pleias", "code"):
        command = commands.add_parser(name)
        command.add_argument("--admission-root", type=Path, required=True)
        command.add_argument("--shard-index", type=int, required=True)
        command.add_argument("--output-root", type=Path, required=True)
    connections = commands.add_parser("connections")
    connections.add_argument("--admission-root", type=Path, required=True)
    connections.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "books":
        result = index_books(args.admission_root, args.selection_root, args.output_root)
    elif args.command == "pleias":
        result = index_pleias_shard(
            args.admission_root, args.shard_index, args.output_root
        )
    elif args.command == "code":
        result = index_code_shard(
            args.admission_root, args.shard_index, args.output_root
        )
    else:
        result = index_connections(args.admission_root, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
