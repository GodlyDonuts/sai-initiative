"""Compile source-text-free final documents into a spiral curriculum index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    OUTPUT_SCHEMA as BOOK_ROW_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    SHARD_SCHEMA as BOOK_SHARD_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    _valid_consensus_curriculum,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_transient_tokenizer_stream import (
    RIGHTS_LABELS,
    _selection,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA as PLEIAS_AGGREGATE_SCHEMA,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_STATUS as PLEIAS_AGGREGATE_STATUS,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    LOCATOR_SCHEMA as PLEIAS_LOCATOR_SCHEMA,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    SHARD_SCHEMA as PLEIAS_SHARD_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-virtual-spiral-curriculum-index-row-v1"
SHARD_SCHEMA = "sai-virtual-spiral-curriculum-index-shard-v1"
AGGREGATE_SCHEMA = "sai-virtual-spiral-curriculum-index-aggregate-v1"
SHARD_STATUS = "complete_nontraining_virtual_spiral_curriculum_index_shard"
AGGREGATE_STATUS = "complete_nontraining_virtual_spiral_curriculum_index"
BANDS = ("foundation", "intermediate", "advanced", "expert")
STAGE_POLICY = {
    "foundation": {
        "token_fraction_ppm": 250_000,
        "band_fraction_ppm": [650_000, 250_000, 80_000, 20_000],
    },
    "expansion": {
        "token_fraction_ppm": 350_000,
        "band_fraction_ppm": [400_000, 400_000, 150_000, 50_000],
    },
    "depth": {
        "token_fraction_ppm": 250_000,
        "band_fraction_ppm": [200_000, 400_000, 300_000, 100_000],
    },
    "synthesis": {
        "token_fraction_ppm": 100_000,
        "band_fraction_ppm": [100_000, 250_000, 400_000, 250_000],
    },
    "annealing": {
        "token_fraction_ppm": 50_000,
        "band_fraction_ppm": [100_000, 200_000, 350_000, 350_000],
    },
}
POLICY = {
    "name": "sai-moving-center-of-gravity-spiral-v1",
    "bands": list(BANDS),
    "difficulty_score": "max(semantic_difficulty_milli, prerequisite_burden_milli)",
    "band_thresholds_milli": [1_000, 2_000, 3_000],
    "stage_policy": STAGE_POLICY,
    "fundamentals_never_zero": True,
    "expert_material_present_from_first_stage": True,
    "exact_token_allocation_deferred_until_selected_tokenizer_retokenization": True,
}
POLICY_SHA256 = canonical_sha256(POLICY)
ROW_FIELD_NAMES = frozenset(
    {
        "schema",
        "component",
        "component_shard",
        "component_row_index",
        "document_identity_sha256",
        "content_sha256",
        "output_text_utf8_bytes",
        "corpus_split",
        "source_group_sha256",
        "rights_label",
        "quality_floor_milli",
        "difficulty_milli",
        "prerequisite_burden_milli",
        "spiral_difficulty_score_milli",
        "spiral_band",
        "semantic_phase_hint",
        "semantic_domains",
        "concepts",
        "prerequisites",
        "concept_prerequisite_signature_sha256",
        "source_custody_sha256",
        "curriculum_priority_sha256",
        "curriculum_policy_sha256",
        "token_count_requires_recomputation",
        "training_ready",
    }
)


class VirtualSpiralCurriculumIndexError(RuntimeError):
    """Final source custody, semantic metadata, split, or index differs."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and bool(item) for item in value
    )


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise VirtualSpiralCurriculumIndexError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("component", pa.string()),
            ("component_shard", pa.int32()),
            ("component_row_index", pa.int64()),
            ("document_identity_sha256", pa.string()),
            ("content_sha256", pa.string()),
            ("output_text_utf8_bytes", pa.int64()),
            ("corpus_split", pa.string()),
            ("source_group_sha256", pa.string()),
            ("rights_label", pa.string()),
            ("quality_floor_milli", pa.int32()),
            ("difficulty_milli", pa.int32()),
            ("prerequisite_burden_milli", pa.int32()),
            ("spiral_difficulty_score_milli", pa.int32()),
            ("spiral_band", pa.string()),
            ("semantic_phase_hint", pa.string()),
            ("semantic_domains", pa.list_(pa.string())),
            ("concepts", pa.list_(pa.string())),
            ("prerequisites", pa.list_(pa.string())),
            ("concept_prerequisite_signature_sha256", pa.string()),
            ("source_custody_sha256", pa.string()),
            ("curriculum_priority_sha256", pa.string()),
            ("curriculum_policy_sha256", pa.string()),
            ("token_count_requires_recomputation", pa.bool_()),
            ("training_ready", pa.bool_()),
        ]
    )


def spiral_band(difficulty_milli: int, burden_milli: int) -> tuple[int, str]:
    if (
        isinstance(difficulty_milli, bool)
        or not isinstance(difficulty_milli, int)
        or isinstance(burden_milli, bool)
        or not isinstance(burden_milli, int)
        or not 0 <= difficulty_milli <= 4_000
        or not 0 <= burden_milli <= 4_000
    ):
        raise VirtualSpiralCurriculumIndexError("spiral difficulty differs")
    score = max(difficulty_milli, burden_milli)
    if score < 1_000:
        band = "foundation"
    elif score < 2_000:
        band = "intermediate"
    elif score < 3_000:
        band = "advanced"
    else:
        band = "expert"
    return score, band


def _index_row(
    *,
    component: str,
    component_shard: int,
    component_row_index: int,
    document_identity_sha256: str,
    content_sha256: str,
    output_text_utf8_bytes: int,
    corpus_split: str,
    source_group_sha256: str,
    rights_label: str,
    quality_floor_milli: int,
    difficulty_milli: int,
    prerequisite_burden_milli: int,
    semantic_phase_hint: str,
    semantic_domains: list[str],
    concepts: list[str],
    prerequisites: list[str],
    source_custody_sha256: str,
) -> dict[str, Any]:
    score, band = spiral_band(difficulty_milli, prerequisite_burden_milli)
    if not all(
        _string_list(value) for value in (semantic_domains, concepts, prerequisites)
    ):
        raise VirtualSpiralCurriculumIndexError("curriculum semantics differ")
    domains = sorted(set(semantic_domains))
    normalized_concepts = sorted(set(concepts))
    normalized_prerequisites = sorted(set(prerequisites))
    if (
        component not in {"institutional_books", "pleias_common_corpus"}
        or isinstance(component_shard, bool)
        or not isinstance(component_shard, int)
        or component_shard < 0
        or isinstance(component_row_index, bool)
        or not isinstance(component_row_index, int)
        or component_row_index < 0
        or not all(
            _is_sha256(value)
            for value in (
                document_identity_sha256,
                content_sha256,
                source_group_sha256,
                source_custody_sha256,
            )
        )
        or corpus_split not in {"train", "development"}
        or not isinstance(output_text_utf8_bytes, int)
        or isinstance(output_text_utf8_bytes, bool)
        or output_text_utf8_bytes <= 0
        or not domains
        or not rights_label
        or not semantic_phase_hint
        or not isinstance(quality_floor_milli, int)
        or isinstance(quality_floor_milli, bool)
        or not 0 <= quality_floor_milli <= 10_000
    ):
        raise VirtualSpiralCurriculumIndexError("curriculum index source differs")
    concept_signature = canonical_sha256(
        {"concepts": normalized_concepts, "prerequisites": normalized_prerequisites}
    )
    priority = canonical_sha256(
        {
            "policy_sha256": POLICY_SHA256,
            "component": component,
            "document_identity_sha256": document_identity_sha256,
            "content_sha256": content_sha256,
        }
    )
    return {
        "schema": ROW_SCHEMA,
        "component": component,
        "component_shard": component_shard,
        "component_row_index": component_row_index,
        "document_identity_sha256": document_identity_sha256,
        "content_sha256": content_sha256,
        "output_text_utf8_bytes": output_text_utf8_bytes,
        "corpus_split": corpus_split,
        "source_group_sha256": source_group_sha256,
        "rights_label": rights_label,
        "quality_floor_milli": quality_floor_milli,
        "difficulty_milli": difficulty_milli,
        "prerequisite_burden_milli": prerequisite_burden_milli,
        "spiral_difficulty_score_milli": score,
        "spiral_band": band,
        "semantic_phase_hint": semantic_phase_hint,
        "semantic_domains": domains,
        "concepts": normalized_concepts,
        "prerequisites": normalized_prerequisites,
        "concept_prerequisite_signature_sha256": concept_signature,
        "source_custody_sha256": source_custody_sha256,
        "curriculum_priority_sha256": priority,
        "curriculum_policy_sha256": POLICY_SHA256,
        "token_count_requires_recomputation": True,
        "training_ready": False,
    }


def _validated_index_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute every derived curriculum field from persisted primitives."""

    if not isinstance(row, dict) or set(row) != ROW_FIELD_NAMES:
        raise VirtualSpiralCurriculumIndexError("index row shape differs")
    replay = _index_row(
        component=row["component"],
        component_shard=row["component_shard"],
        component_row_index=row["component_row_index"],
        document_identity_sha256=row["document_identity_sha256"],
        content_sha256=row["content_sha256"],
        output_text_utf8_bytes=row["output_text_utf8_bytes"],
        corpus_split=row["corpus_split"],
        source_group_sha256=row["source_group_sha256"],
        rights_label=row["rights_label"],
        quality_floor_milli=row["quality_floor_milli"],
        difficulty_milli=row["difficulty_milli"],
        prerequisite_burden_milli=row["prerequisite_burden_milli"],
        semantic_phase_hint=row["semantic_phase_hint"],
        semantic_domains=row["semantic_domains"],
        concepts=row["concepts"],
        prerequisites=row["prerequisites"],
        source_custody_sha256=row["source_custody_sha256"],
    )
    if replay != row:
        raise VirtualSpiralCurriculumIndexError("index row replay differs")
    return replay


def _update_counts(counts: Counter[str], row: dict[str, Any]) -> None:
    counts["documents"] += 1
    counts["output_text_utf8_bytes"] += row["output_text_utf8_bytes"]
    counts[f"split::{row['corpus_split']}::documents"] += 1
    counts[f"split::{row['corpus_split']}::output_text_utf8_bytes"] += row[
        "output_text_utf8_bytes"
    ]
    counts[f"band::{row['spiral_band']}::documents"] += 1
    counts[f"band::{row['spiral_band']}::output_text_utf8_bytes"] += row[
        "output_text_utf8_bytes"
    ]
    for domain in row["semantic_domains"]:
        counts[f"domain::{domain}::documents"] += 1


def _row_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        _validated_index_row(row)
        _update_counts(counts, row)
    return counts


def _bound_receipts(
    root: Path,
    aggregate_schema: str,
    shard_schema: str,
    logical_shards: int,
    shard_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or logical_shards <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < logical_shards
    ):
        raise VirtualSpiralCurriculumIndexError("final shard geometry differs")
    aggregate = _load_signed(root / "aggregate.json", aggregate_schema)
    receipts = []
    selected = None
    for index in range(logical_shards):
        receipt = _load_signed(
            root / "shards" / f"shard_{index:05d}" / "receipt.json", shard_schema
        )
        receipts.append(receipt["receipt_sha256"])
        if index == shard_index:
            selected = receipt
    if (
        selected is None
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("shards", {}).get("ordered_receipts_sha256")
        != canonical_sha256(receipts)
    ):
        raise VirtualSpiralCurriculumIndexError("final shard custody differs")
    return aggregate, selected


def _write_shard(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    component: str,
    logical_shards: int,
    shard_index: int,
    source_receipt_sha256: str,
) -> dict[str, Any]:
    if (
        output_root.exists()
        or output_root.is_symlink()
        or not rows
        or component not in {"institutional_books", "pleias_common_corpus"}
        or isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or logical_shards <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < logical_shards
        or not _is_sha256(source_receipt_sha256)
    ):
        raise VirtualSpiralCurriculumIndexError("index shard output differs")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise VirtualSpiralCurriculumIndexError("pyarrow is required") from error
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        index_path = stage / "curriculum-index.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=_schema()), index_path, compression="zstd"
        )
        counts = _row_counts(rows)
        ordered = hashlib.sha256()
        for row in rows:
            ordered.update(bytes.fromhex(row["curriculum_priority_sha256"]))
        payload = {
            "schema": SHARD_SCHEMA,
            "status": SHARD_STATUS,
            "component": component,
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "source_receipt_sha256": source_receipt_sha256,
            "curriculum_policy": POLICY,
            "curriculum_policy_sha256": POLICY_SHA256,
            "counts": dict(sorted(counts.items())),
            "index": {
                "path": index_path.name,
                "rows": len(rows),
                "bytes": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
                "ordered_priority_digests_sha256": ordered.hexdigest(),
            },
            "source_text_persisted": False,
            "token_count_requires_recomputation": True,
            "exact_token_allocation_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def index_pleias_shard(
    final_root: Path,
    output_root: Path,
    *,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Index one final PleIAs locator shard without reconstructing source text."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise VirtualSpiralCurriculumIndexError("pyarrow is required") from error
    aggregate, receipt = _bound_receipts(
        final_root,
        PLEIAS_AGGREGATE_SCHEMA,
        PLEIAS_SHARD_SCHEMA,
        logical_shards,
        shard_index,
    )
    if aggregate.get("status") != PLEIAS_AGGREGATE_STATUS:
        raise VirtualSpiralCurriculumIndexError("final PleIAs aggregate differs")
    descriptor = receipt.get("final_locators")
    root = final_root / "shards" / f"shard_{shard_index:05d}"
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise VirtualSpiralCurriculumIndexError("final PleIAs locators differ")
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, use_threads=False):
        for locator in batch.to_pylist():
            unsigned = {
                key: value for key, value in locator.items() if key != "locator_sha256"
            }
            if (
                locator.get("schema") != PLEIAS_LOCATOR_SCHEMA
                or locator.get("locator_sha256") != canonical_sha256(unsigned)
                or locator.get("training_ready") is not False
            ):
                raise VirtualSpiralCurriculumIndexError("final PleIAs locator differs")
            rows.append(
                _index_row(
                    component="pleias_common_corpus",
                    component_shard=shard_index,
                    component_row_index=locator["virtual_row_index"],
                    document_identity_sha256=locator["source_row_identity_sha256"],
                    content_sha256=locator["content_sha256"],
                    output_text_utf8_bytes=locator["output_text_utf8_bytes"],
                    corpus_split=locator["corpus_split"],
                    source_group_sha256=locator["source_group_sha256"],
                    rights_label=locator["license"],
                    quality_floor_milli=locator["semantic_quality_floor_milli"],
                    difficulty_milli=locator["semantic_difficulty_mean_milli"],
                    prerequisite_burden_milli=locator[
                        "semantic_prerequisite_burden_mean_milli"
                    ],
                    semantic_phase_hint=locator["semantic_curriculum_phase"],
                    semantic_domains=locator["semantic_domains"],
                    concepts=locator["semantic_recurring_concepts"],
                    prerequisites=locator["semantic_recurring_prerequisites"],
                    source_custody_sha256=locator["locator_sha256"],
                )
            )
    if len(rows) != descriptor.get("rows"):
        raise VirtualSpiralCurriculumIndexError("PleIAs index coverage differs")
    return _write_shard(
        output_root,
        rows,
        component="pleias_common_corpus",
        logical_shards=logical_shards,
        shard_index=shard_index,
        source_receipt_sha256=receipt["receipt_sha256"],
    )


def index_book_shard(
    final_root: Path,
    selection_root: Path,
    output_root: Path,
    *,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Index one final private-book shard with exact rights and transient text hash."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise VirtualSpiralCurriculumIndexError("pyarrow is required") from error
    aggregate, receipt = _bound_receipts(
        final_root,
        BOOK_AGGREGATE_SCHEMA,
        BOOK_SHARD_SCHEMA,
        logical_shards,
        shard_index,
    )
    if (
        aggregate.get("status")
        != "complete_nontraining_institutional_books_cross_source_rewritten"
    ):
        raise VirtualSpiralCurriculumIndexError("final book aggregate differs")
    selected, selection_receipt = _selection(selection_root)
    rows = []
    component_row = 0
    root = final_root / "shards" / f"shard_{shard_index:05d}"
    for split in ("train", "development"):
        descriptor = receipt.get("outputs", {}).get(split)
        if descriptor is None:
            continue
        path = root / descriptor.get("path", "")
        if (
            descriptor.get("path") != f"{split}.parquet"
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise VirtualSpiralCurriculumIndexError("final book partition differs")
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=16, use_threads=False
        ):
            for row in batch.to_pylist():
                text = row.get("text")
                selection = selected.get(row.get("barcode_src"))
                try:
                    curriculum = json.loads(row.get("curriculum_metadata_json", ""))
                except json.JSONDecodeError as error:
                    raise VirtualSpiralCurriculumIndexError(
                        "book curriculum metadata differs"
                    ) from error
                if not isinstance(curriculum, dict):
                    raise VirtualSpiralCurriculumIndexError(
                        "book curriculum metadata differs"
                    )
                unsigned_curriculum = {
                    key: value
                    for key, value in curriculum.items()
                    if key != "metadata_sha256"
                }
                if (
                    row.get("schema") != BOOK_ROW_SCHEMA
                    or row.get("training_ready") is not False
                    or row.get("corpus_split") != split
                    or not isinstance(text, str)
                    or not text
                    or hashlib.sha256(text.encode()).hexdigest()
                    != row.get("content_sha256")
                    or selection is None
                    or row.get("selection_row_sha256") != selection.get("row_sha256")
                    or curriculum.get("metadata_sha256")
                    != row.get("curriculum_metadata_sha256")
                    or curriculum.get("metadata_sha256")
                    != canonical_sha256(unsigned_curriculum)
                    or not _valid_consensus_curriculum(curriculum)
                ):
                    raise VirtualSpiralCurriculumIndexError("final book row differs")
                complexity = curriculum["complexity_range"]
                difficulty = (
                    max(value["maximum"] for value in complexity.values()) - 1
                ) * 1_000
                prerequisites = curriculum["shared_prerequisites"]
                identity = canonical_sha256(
                    {
                        "component": "institutional_books",
                        "barcode_src": row["barcode_src"],
                        "content_sha256": row["content_sha256"],
                    }
                )
                custody = canonical_sha256(
                    {
                        "final_shard_receipt_sha256": receipt["receipt_sha256"],
                        "selection_receipt_sha256": selection_receipt["receipt_sha256"],
                        "selection_row_sha256": selection["row_sha256"],
                        "quality_agreement_record_sha256": row[
                            "quality_agreement_record_sha256"
                        ],
                        "benchmark_decontamination_record_sha256": row[
                            "benchmark_decontamination_record_sha256"
                        ],
                        "cross_source_subdocument_transform_sha256": row[
                            "cross_source_subdocument_transform_sha256"
                        ],
                    }
                )
                rows.append(
                    _index_row(
                        component="institutional_books",
                        component_shard=shard_index,
                        component_row_index=component_row,
                        document_identity_sha256=identity,
                        content_sha256=row["content_sha256"],
                        output_text_utf8_bytes=len(text.encode()),
                        corpus_split=split,
                        source_group_sha256=row["source_group_sha256"],
                        rights_label=RIGHTS_LABELS[selection["rights_code"]],
                        quality_floor_milli=min(curriculum["quality_floor"].values())
                        * 2_000,
                        difficulty_milli=difficulty,
                        prerequisite_burden_milli=min(4_000, len(prerequisites) * 500),
                        semantic_phase_hint="+".join(
                            sorted(set(curriculum["curriculum_band_votes"]))
                        ),
                        semantic_domains=row["semantic_domains"],
                        concepts=curriculum["shared_concepts"],
                        prerequisites=prerequisites,
                        source_custody_sha256=custody,
                    )
                )
                component_row += 1
    if component_row != receipt.get("counts", {}).get("documents"):
        raise VirtualSpiralCurriculumIndexError("book index coverage differs")
    return _write_shard(
        output_root,
        rows,
        component="institutional_books",
        logical_shards=logical_shards,
        shard_index=shard_index,
        source_receipt_sha256=receipt["receipt_sha256"],
    )


def aggregate_indexes(
    pleias_index_root: Path,
    book_index_root: Path,
    pleias_final_root: Path,
    book_final_root: Path,
    output: Path,
    *,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Verify all 192 source-text-free indexes and reject global duplicates."""

    if output.exists() or output.is_symlink():
        raise VirtualSpiralCurriculumIndexError("index aggregate output exists")
    pleias_final = _load_signed(
        pleias_final_root / "aggregate.json", PLEIAS_AGGREGATE_SCHEMA
    )
    book_final = _load_signed(book_final_root / "aggregate.json", BOOK_AGGREGATE_SCHEMA)
    if (
        pleias_final.get("status") != PLEIAS_AGGREGATE_STATUS
        or book_final.get("status")
        != "complete_nontraining_institutional_books_cross_source_rewritten"
    ):
        raise VirtualSpiralCurriculumIndexError("final aggregate status differs")
    totals: Counter[str] = Counter()
    receipts = []
    ordered_priorities = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        prefix="sai-virtual-curriculum-index-", dir=scratch_root
    ) as directory:
        database = sqlite3.connect(Path(directory) / "identities.sqlite3")
        database.execute(
            "CREATE TABLE documents (identity TEXT PRIMARY KEY, "
            "content TEXT UNIQUE) WITHOUT ROWID"
        )
        try:
            for component, root, final_root, final_schema, shards in (
                (
                    "institutional_books",
                    book_index_root,
                    book_final_root,
                    BOOK_SHARD_SCHEMA,
                    64,
                ),
                (
                    "pleias_common_corpus",
                    pleias_index_root,
                    pleias_final_root,
                    PLEIAS_SHARD_SCHEMA,
                    128,
                ),
            ):
                for shard_index in range(shards):
                    shard = root / "shards" / f"shard_{shard_index:05d}"
                    receipt = _load_signed(shard / "receipt.json", SHARD_SCHEMA)
                    final_receipt = _load_signed(
                        final_root
                        / "shards"
                        / f"shard_{shard_index:05d}"
                        / "receipt.json",
                        final_schema,
                    )
                    descriptor = receipt.get("index")
                    path = (
                        shard / descriptor.get("path", "")
                        if isinstance(descriptor, dict)
                        else shard
                    )
                    if (
                        receipt.get("status") != SHARD_STATUS
                        or receipt.get("component") != component
                        or receipt.get("logical_shards") != shards
                        or receipt.get("shard_index") != shard_index
                        or receipt.get("source_receipt_sha256")
                        != final_receipt["receipt_sha256"]
                        or receipt.get("curriculum_policy_sha256") != POLICY_SHA256
                        or receipt.get("source_text_persisted") is not False
                        or not isinstance(descriptor, dict)
                        or not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_nlink != 1
                        or path.stat().st_size != descriptor.get("bytes")
                        or sha256_file(path) != descriptor.get("sha256")
                    ):
                        raise VirtualSpiralCurriculumIndexError("index shard differs")
                    try:
                        import pyarrow.parquet as pq
                    except ImportError as error:
                        raise VirtualSpiralCurriculumIndexError(
                            "pyarrow is required"
                        ) from error
                    rows = 0
                    shard_counts: Counter[str] = Counter()
                    shard_ordered = hashlib.sha256()
                    for batch in pq.ParquetFile(path).iter_batches(
                        batch_size=1024, use_threads=False
                    ):
                        for row in batch.to_pylist():
                            if (
                                row.get("component") != component
                                or row.get("component_shard") != shard_index
                                or row.get("component_row_index") != rows
                            ):
                                raise VirtualSpiralCurriculumIndexError(
                                    "index row differs"
                                )
                            _validated_index_row(row)
                            _update_counts(shard_counts, row)
                            try:
                                database.execute(
                                    "INSERT INTO documents VALUES (?, ?)",
                                    (
                                        row["document_identity_sha256"],
                                        row["content_sha256"],
                                    ),
                                )
                            except sqlite3.IntegrityError as error:
                                raise VirtualSpiralCurriculumIndexError(
                                    "curriculum index contains a global duplicate"
                                ) from error
                            totals["documents"] += 1
                            totals["output_text_utf8_bytes"] += row[
                                "output_text_utf8_bytes"
                            ]
                            totals[f"component::{component}::documents"] += 1
                            totals[
                                f"component::{component}::output_text_utf8_bytes"
                            ] += row["output_text_utf8_bytes"]
                            totals[f"split::{row['corpus_split']}::documents"] += 1
                            totals[
                                f"split::{row['corpus_split']}::output_text_utf8_bytes"
                            ] += row["output_text_utf8_bytes"]
                            totals[f"band::{row['spiral_band']}::documents"] += 1
                            totals[
                                f"band::{row['spiral_band']}::output_text_utf8_bytes"
                            ] += row["output_text_utf8_bytes"]
                            for domain in row["semantic_domains"]:
                                totals[f"domain::{domain}::documents"] += 1
                            ordered_priorities.update(
                                bytes.fromhex(row["curriculum_priority_sha256"])
                            )
                            shard_ordered.update(
                                bytes.fromhex(row["curriculum_priority_sha256"])
                            )
                            rows += 1
                    if (
                        rows != descriptor.get("rows")
                        or dict(sorted(shard_counts.items())) != receipt.get("counts")
                        or shard_ordered.hexdigest()
                        != descriptor.get("ordered_priority_digests_sha256")
                    ):
                        raise VirtualSpiralCurriculumIndexError(
                            "index row coverage differs"
                        )
                    receipts.append(receipt["receipt_sha256"])
                database.commit()
        finally:
            database.close()
    if (
        totals["component::pleias_common_corpus::documents"]
        != pleias_final.get("totals", {}).get("documents")
        or totals["component::pleias_common_corpus::output_text_utf8_bytes"]
        != pleias_final.get("totals", {}).get("output_text_utf8_bytes")
        or totals["component::institutional_books::documents"]
        != book_final.get("totals", {}).get("documents")
        or totals["component::institutional_books::output_text_utf8_bytes"]
        != book_final.get("totals", {}).get("output_text_utf8_bytes")
        or totals["split::train::documents"] + totals["split::development::documents"]
        != totals["documents"]
        or totals["split::train::output_text_utf8_bytes"]
        + totals["split::development::output_text_utf8_bytes"]
        != totals["output_text_utf8_bytes"]
    ):
        raise VirtualSpiralCurriculumIndexError("index aggregate coverage differs")
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "source": {
            "pleias_final_aggregate_receipt_sha256": pleias_final["receipt_sha256"],
            "book_final_aggregate_receipt_sha256": book_final["receipt_sha256"],
        },
        "curriculum_policy": POLICY,
        "curriculum_policy_sha256": POLICY_SHA256,
        "index_shards": {
            "institutional_books": 64,
            "pleias_common_corpus": 128,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "ordered_curriculum_priority_digests_sha256": ordered_priorities.hexdigest(),
        "exact_document_identity_unique": True,
        "exact_content_identity_unique": True,
        "source_disjoint_split_preserved": True,
        "concept_prerequisite_metadata_complete": True,
        "spiral_band_assignment_complete": True,
        "source_text_persisted": False,
        "token_count_requires_recomputation": True,
        "exact_token_allocation_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pleias = commands.add_parser("pleias-shard")
    pleias.add_argument("--final-root", type=Path, required=True)
    pleias.add_argument("--output-root", type=Path, required=True)
    pleias.add_argument("--logical-shards", type=int, required=True)
    pleias.add_argument("--shard-index", type=int, required=True)
    books = commands.add_parser("book-shard")
    books.add_argument("--final-root", type=Path, required=True)
    books.add_argument("--selection-root", type=Path, required=True)
    books.add_argument("--output-root", type=Path, required=True)
    books.add_argument("--logical-shards", type=int, required=True)
    books.add_argument("--shard-index", type=int, required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--pleias-index-root", type=Path, required=True)
    combine.add_argument("--book-index-root", type=Path, required=True)
    combine.add_argument("--pleias-final-root", type=Path, required=True)
    combine.add_argument("--book-final-root", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    if args.command == "pleias-shard":
        result = index_pleias_shard(
            args.final_root,
            args.output_root,
            logical_shards=args.logical_shards,
            shard_index=args.shard_index,
        )
    elif args.command == "book-shard":
        result = index_book_shard(
            args.final_root,
            args.selection_root,
            args.output_root,
            logical_shards=args.logical_shards,
            shard_index=args.shard_index,
        )
    else:
        result = aggregate_indexes(
            args.pleias_index_root,
            args.book_index_root,
            args.pleias_final_root,
            args.book_final_root,
            args.output,
            scratch_root=args.scratch_root,
        )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
