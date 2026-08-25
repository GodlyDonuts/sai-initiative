"""Replay and aggregate every Sai 1B source-text-free curriculum shard."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_component_admission import SCHEMA as BRIDGE_ADMISSION_SCHEMA
from sai.data.common_pile_stack_edu_practical_admission import (
    SCHEMA as CODE_ADMISSION_SCHEMA,
)
from sai.data.final_training_release import SCHEMA as RELEASE_SCHEMA
from sai.data.institutional_books_practical_admission import (
    SCHEMA as BOOK_ADMISSION_SCHEMA,
)
from sai.data.one_b_curriculum_index import (
    AGGREGATE_SCHEMA,
    BANDS,
    COMPONENTS,
    SHARD_SCHEMA,
)
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_ADMISSION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file


class OneBCurriculumAggregateError(RuntimeError):
    """A curriculum shard, component total, or release edge differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise OneBCurriculumAggregateError("signed aggregate input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBCurriculumAggregateError("signed aggregate input differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBCurriculumAggregateError("signed aggregate input differs")
    return payload


def _shard_paths(index_root: Path) -> list[Path]:
    values = [index_root / "books" / "receipt.json"]
    values.extend(
        index_root / "pleias" / f"shard_{index:05d}" / "receipt.json"
        for index in range(128)
    )
    values.extend(
        index_root / "code" / f"shard_{index:05d}" / "receipt.json"
        for index in range(32)
    )
    values.append(index_root / "connections" / "receipt.json")
    return values


def aggregate(
    index_root: Path,
    release_path: Path,
    book_admission_path: Path,
    pleias_admission_path: Path,
    code_admission_path: Path,
    bridge_admission_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind all 162 index shards to the exact signed three-component release."""

    if output.exists() or output.is_symlink():
        raise OneBCurriculumAggregateError("aggregate output exists")
    release = _load_signed(release_path, RELEASE_SCHEMA)
    books = _load_signed(book_admission_path, BOOK_ADMISSION_SCHEMA)
    pleias = _load_signed(pleias_admission_path, PLEIAS_ADMISSION_SCHEMA)
    code = _load_signed(code_admission_path, CODE_ADMISSION_SCHEMA)
    bridge = _load_signed(bridge_admission_path, BRIDGE_ADMISSION_SCHEMA)
    if (
        release.get("training_data_ready") is not True
        or release.get("model_training_started") is not False
        or books.get("training_ready") is not True
        or pleias.get("training_ready") is not True
        or code.get("training_ready") is not True
        or bridge.get("training_ready") is not True
    ):
        raise OneBCurriculumAggregateError("released source is not index eligible")

    expected = {
        "books": (
            books["counts"]["admitted_rows"],
            books["counts"]["admitted_text_utf8_bytes"],
            books["receipt_sha256"],
            1,
        ),
        "pleias": (
            pleias["counts"]["admitted_rows"],
            pleias["counts"]["admitted_text_utf8_bytes"],
            pleias["receipt_sha256"],
            128,
        ),
        "code": (
            code["counts"]["admitted_rows"],
            code["counts"]["admitted_text_utf8_bytes"],
            code["receipt_sha256"],
            32,
        ),
        "connections": (
            bridge["counts"]["train_documents"],
            bridge["train"]["text_utf8_bytes"],
            bridge["receipt_sha256"],
            1,
        ),
    }
    counts: Counter[str] = Counter()
    receipts = []
    component_shards: Counter[str] = Counter()
    component_rows: Counter[str] = Counter()
    component_bytes: Counter[str] = Counter()
    for receipt_path in _shard_paths(index_root):
        receipt = _load_signed(receipt_path, SHARD_SCHEMA)
        component = receipt.get("component")
        descriptor = receipt.get("output", {})
        path = receipt_path.parent / descriptor.get("path", "")
        if (
            component not in COMPONENTS
            or receipt.get("status") != "complete_1b_curriculum_index_shard"
            or receipt.get("source_receipt_sha256") != expected[component][2]
            or receipt.get("source_text_persisted") is not False
            or receipt.get("curriculum_index_ready") is not True
            or receipt.get("model_training_started") is not False
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise OneBCurriculumAggregateError("curriculum shard differs")
        shard_counts = receipt.get("counts", {})
        if (
            descriptor.get("rows") != shard_counts.get("rows")
            or sum(
                shard_counts.get(f"band::{band}::rows", 0) for band in BANDS
            )
            != shard_counts.get("rows")
            or sum(
                shard_counts.get(f"split::{split}::rows", 0)
                for split in ("train", "development")
            )
            != shard_counts.get("rows")
        ):
            raise OneBCurriculumAggregateError("curriculum shard accounting differs")
        component_shards[component] += 1
        component_rows[component] += shard_counts["rows"]
        component_bytes[component] += shard_counts["text_utf8_bytes"]
        for key, value in shard_counts.items():
            counts[key] += value
            counts[f"component::{component}::{key}"] += value
        receipts.append(receipt["receipt_sha256"])

    for component in COMPONENTS:
        expected_rows, expected_bytes, _, expected_shards = expected[component]
        if (
            component_shards[component] != expected_shards
            or component_rows[component] != expected_rows
            or component_bytes[component] != expected_bytes
        ):
            raise OneBCurriculumAggregateError("curriculum component coverage differs")
    if (
        counts["rows"] != release["totals"]["rows"]
        or counts["text_utf8_bytes"] != release["totals"]["logical_text_utf8_bytes"]
        or any(counts[f"band::{band}::rows"] <= 0 for band in BANDS)
        or counts["split::development::rows"] <= 0
        or counts["component::connections::split::development::rows"] != 0
    ):
        raise OneBCurriculumAggregateError("curriculum release accounting differs")

    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_1b_curriculum_index_aggregate",
        "release_receipt_sha256": release["receipt_sha256"],
        "ordered_shard_receipts_sha256": canonical_sha256(receipts),
        "shards": len(receipts),
        "counts": dict(sorted(counts.items())),
        "component_shards": dict(sorted(component_shards.items())),
        "all_release_rows_accounted": True,
        "all_release_bytes_accounted": True,
        "all_four_spiral_bands_present": True,
        "bulk_internal_development_partition_complete": True,
        "connection_development_rows_physically_excluded": True,
        "source_text_persisted": False,
        "curriculum_index_complete": True,
        "production_tokenizer_selected": False,
        "packed_stream_smoke_complete": False,
        "training_ready": False,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--book-admission", type=Path, required=True)
    parser.add_argument("--pleias-admission", type=Path, required=True)
    parser.add_argument("--code-admission", type=Path, required=True)
    parser.add_argument("--bridge-admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.index_root,
        args.release,
        args.book_admission,
        args.pleias_admission,
        args.code_admission,
        args.bridge_admission,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
