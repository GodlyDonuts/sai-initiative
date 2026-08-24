"""Measure final virtual-corpus breadth without mistaking volume for quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_compiler_labeling import DOMAINS
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file
from sai.data.virtual_spiral_curriculum_index import (
    AGGREGATE_SCHEMA as INDEX_AGGREGATE_SCHEMA,
)
from sai.data.virtual_spiral_curriculum_index import (
    AGGREGATE_STATUS as INDEX_AGGREGATE_STATUS,
)
from sai.data.virtual_spiral_curriculum_index import BANDS, _validated_index_row
from sai.data.virtual_spiral_curriculum_index import SHARD_SCHEMA as INDEX_SHARD_SCHEMA
from sai.data.virtual_spiral_curriculum_index import SHARD_STATUS as INDEX_SHARD_STATUS

SCHEMA = "sai-virtual-curriculum-coverage-v1"
STATUS = "complete_nontraining_virtual_curriculum_coverage"
POLICY = {
    "name": "sai-two-terabyte-polymath-coverage-v1",
    "minimum_post_rewrite_utf8_bytes": 1_900_000_000_000,
    "maximum_post_rewrite_utf8_bytes": 2_000_000_000_000,
    "required_domains": list(DOMAINS),
    "required_bands": list(BANDS),
    "minimum_concept_document_coverage_ppm": 900_000,
    "minimum_prerequisite_document_coverage_ppm": 250_000,
    "minimum_cross_domain_metadata_document_coverage_ppm": 50_000,
    "high_quality_floor_milli": 6_000,
    "minimum_high_quality_byte_coverage_ppm": 800_000,
    "development_required_in_every_component": True,
    "cross_domain_labels_are_not_verified_bridges": True,
    "verified_bridge_component_required_for_final_corpus": True,
    "selected_tokenizer_required_for_exact_allocation": True,
}
POLICY_SHA256 = canonical_sha256(POLICY)


class VirtualCurriculumCoverageError(RuntimeError):
    """Index custody, accounting, semantics, or coverage evidence differs."""


def _ppm(numerator: int, denominator: int) -> int:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise VirtualCurriculumCoverageError("coverage denominator differs")
    return numerator * 1_000_000 // denominator


def _descriptor(root: Path, receipt: dict[str, Any]) -> Path:
    descriptor = receipt.get("index")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        receipt.get("status") != INDEX_SHARD_STATUS
        or receipt.get("source_text_persisted") is not False
        or not isinstance(descriptor, dict)
        or descriptor.get("path") != "curriculum-index.parquet"
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise VirtualCurriculumCoverageError("curriculum index shard differs")
    return path


def _update_core(counts: Counter[str], row: dict[str, Any]) -> None:
    component = row["component"]
    split = row["corpus_split"]
    band = row["spiral_band"]
    size = row["output_text_utf8_bytes"]
    counts["documents"] += 1
    counts["output_text_utf8_bytes"] += size
    counts[f"component::{component}::documents"] += 1
    counts[f"component::{component}::output_text_utf8_bytes"] += size
    counts[f"split::{split}::documents"] += 1
    counts[f"split::{split}::output_text_utf8_bytes"] += size
    counts[f"component::{component}::split::{split}::documents"] += 1
    counts[f"band::{band}::documents"] += 1
    counts[f"band::{band}::output_text_utf8_bytes"] += size
    quality = row["quality_floor_milli"]
    counts[f"quality_floor::{quality}::documents"] += 1
    counts[f"quality_floor::{quality}::output_text_utf8_bytes"] += size
    for domain in row["semantic_domains"]:
        counts[f"domain::{domain}::documents"] += 1
        counts[f"domain::{domain}::output_text_utf8_bytes"] += size
        if quality >= POLICY["high_quality_floor_milli"]:
            counts[f"domain::{domain}::high_quality_output_text_utf8_bytes"] += size


def _publish(
    output: Path, durable_output: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    if (
        output.exists()
        or output.is_symlink()
        or durable_output.exists()
        or durable_output.is_symlink()
        or output.resolve() == durable_output.resolve()
    ):
        raise VirtualCurriculumCoverageError("coverage output differs")
    _atomic_create(output, payload)
    try:
        _atomic_create(durable_output, payload)
    except BaseException:
        output.unlink()
        raise
    return payload


def measure_coverage(
    index_root: Path,
    output: Path,
    durable_output: Path,
    *,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Replay 192 index shards and measure breadth, density, and prerequisites."""

    aggregate = _load_signed(index_root / "aggregate.json", INDEX_AGGREGATE_SCHEMA)
    if (
        aggregate.get("status") != INDEX_AGGREGATE_STATUS
        or aggregate.get("source_text_persisted") is not False
        or aggregate.get("exact_document_identity_unique") is not True
        or aggregate.get("exact_content_identity_unique") is not True
        or aggregate.get("source_disjoint_split_preserved") is not True
    ):
        raise VirtualCurriculumCoverageError("curriculum index aggregate differs")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise VirtualCurriculumCoverageError("pyarrow is required") from error

    counts: Counter[str] = Counter()
    receipt_digests: list[str] = []
    priority_digests = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        prefix="sai-virtual-curriculum-coverage-", dir=scratch_root
    ) as directory:
        database = sqlite3.connect(Path(directory) / "semantics.sqlite3")
        database.executescript(
            "CREATE TABLE concepts (value TEXT PRIMARY KEY) WITHOUT ROWID;"
            "CREATE TABLE prerequisites (value TEXT PRIMARY KEY) WITHOUT ROWID;"
            "CREATE TABLE domain_pairs (left_value TEXT NOT NULL, "
            "right_value TEXT NOT NULL, PRIMARY KEY(left_value, right_value)) "
            "WITHOUT ROWID;"
        )
        try:
            for component, component_root, logical_shards in (
                ("institutional_books", index_root / "books", 64),
                ("pleias_common_corpus", index_root / "pleias", 128),
            ):
                for shard_index in range(logical_shards):
                    root = component_root / "shards" / f"shard_{shard_index:05d}"
                    receipt = _load_signed(root / "receipt.json", INDEX_SHARD_SCHEMA)
                    if (
                        receipt.get("component") != component
                        or receipt.get("logical_shards") != logical_shards
                        or receipt.get("shard_index") != shard_index
                        or receipt.get("curriculum_policy_sha256")
                        != aggregate.get("curriculum_policy_sha256")
                    ):
                        raise VirtualCurriculumCoverageError(
                            "curriculum index shard custody differs"
                        )
                    path = _descriptor(root, receipt)
                    rows = 0
                    shard_priority = hashlib.sha256()
                    for batch in pq.ParquetFile(path).iter_batches(
                        batch_size=1024, use_threads=False
                    ):
                        for row in batch.to_pylist():
                            if (
                                row.get("component") != component
                                or row.get("component_shard") != shard_index
                                or row.get("component_row_index") != rows
                            ):
                                raise VirtualCurriculumCoverageError(
                                    "curriculum index row order differs"
                                )
                            _validated_index_row(row)
                            _update_core(counts, row)
                            concepts = row["concepts"]
                            prerequisites = row["prerequisites"]
                            domains = row["semantic_domains"]
                            if concepts:
                                counts["documents_with_concepts"] += 1
                                counts["concept_mentions"] += len(concepts)
                            if prerequisites:
                                counts["documents_with_prerequisites"] += 1
                                counts["prerequisite_mentions"] += len(prerequisites)
                            if concepts and prerequisites:
                                counts["documents_with_concepts_and_prerequisites"] += 1
                            if len(domains) >= 2:
                                counts["cross_domain_metadata_documents"] += 1
                                counts["cross_domain_metadata_utf8_bytes"] += row[
                                    "output_text_utf8_bytes"
                                ]
                            if (
                                row["quality_floor_milli"]
                                >= POLICY["high_quality_floor_milli"]
                            ):
                                counts["high_quality_documents"] += 1
                                counts["high_quality_utf8_bytes"] += row[
                                    "output_text_utf8_bytes"
                                ]
                            database.executemany(
                                "INSERT OR IGNORE INTO concepts VALUES (?)",
                                ((value,) for value in concepts),
                            )
                            database.executemany(
                                "INSERT OR IGNORE INTO prerequisites VALUES (?)",
                                ((value,) for value in prerequisites),
                            )
                            database.executemany(
                                "INSERT OR IGNORE INTO domain_pairs VALUES (?, ?)",
                                combinations(domains, 2),
                            )
                            digest = row["curriculum_priority_sha256"]
                            priority_digests.update(bytes.fromhex(digest))
                            shard_priority.update(bytes.fromhex(digest))
                            rows += 1
                    descriptor = receipt["index"]
                    if rows != descriptor.get(
                        "rows"
                    ) or shard_priority.hexdigest() != descriptor.get(
                        "ordered_priority_digests_sha256"
                    ):
                        raise VirtualCurriculumCoverageError(
                            "curriculum index shard coverage differs"
                        )
                    receipt_digests.append(receipt["receipt_sha256"])
                    database.commit()
            unique_concepts = database.execute(
                "SELECT COUNT(*) FROM concepts"
            ).fetchone()[0]
            unique_prerequisites = database.execute(
                "SELECT COUNT(*) FROM prerequisites"
            ).fetchone()[0]
            unique_domain_pairs = database.execute(
                "SELECT COUNT(*) FROM domain_pairs"
            ).fetchone()[0]
        finally:
            database.close()

    aggregate_totals = aggregate.get("totals")
    if not isinstance(aggregate_totals, dict):
        raise VirtualCurriculumCoverageError("curriculum index totals differ")
    for key, value in aggregate_totals.items():
        if counts.get(key, 0) != value:
            raise VirtualCurriculumCoverageError(
                f"curriculum index total {key} differs"
            )
    if canonical_sha256(receipt_digests) != aggregate.get("index_shards", {}).get(
        "ordered_receipts_sha256"
    ) or priority_digests.hexdigest() != aggregate.get(
        "ordered_curriculum_priority_digests_sha256"
    ):
        raise VirtualCurriculumCoverageError("curriculum index ordering differs")

    documents = counts["documents"]
    size = counts["output_text_utf8_bytes"]
    measurements = {
        "concept_document_coverage_ppm": _ppm(
            counts["documents_with_concepts"], documents
        ),
        "prerequisite_document_coverage_ppm": _ppm(
            counts["documents_with_prerequisites"], documents
        ),
        "cross_domain_metadata_document_coverage_ppm": _ppm(
            counts["cross_domain_metadata_documents"], documents
        ),
        "high_quality_byte_coverage_ppm": _ppm(counts["high_quality_utf8_bytes"], size),
        "unique_concepts": unique_concepts,
        "unique_prerequisites": unique_prerequisites,
        "unique_domain_pairs": unique_domain_pairs,
    }
    requirements = {
        "two_terabyte_window": POLICY["minimum_post_rewrite_utf8_bytes"]
        <= size
        <= POLICY["maximum_post_rewrite_utf8_bytes"],
        "all_polymath_domains_present": all(
            counts[f"domain::{domain}::documents"] > 0 for domain in DOMAINS
        ),
        "all_spiral_bands_present": all(
            counts[f"band::{band}::output_text_utf8_bytes"] > 0 for band in BANDS
        ),
        "development_present_in_every_component": all(
            counts[f"component::{component}::split::development::documents"] > 0
            for component in ("institutional_books", "pleias_common_corpus")
        ),
        "concept_coverage": measurements["concept_document_coverage_ppm"]
        >= POLICY["minimum_concept_document_coverage_ppm"],
        "prerequisite_coverage": measurements["prerequisite_document_coverage_ppm"]
        >= POLICY["minimum_prerequisite_document_coverage_ppm"],
        "cross_domain_metadata_coverage": measurements[
            "cross_domain_metadata_document_coverage_ppm"
        ]
        >= POLICY["minimum_cross_domain_metadata_document_coverage_ppm"],
        "high_quality_byte_coverage": measurements["high_quality_byte_coverage_ppm"]
        >= POLICY["minimum_high_quality_byte_coverage_ppm"],
    }
    failed = sorted(key for key, passed in requirements.items() if not passed)
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_index_aggregate_receipt_sha256": aggregate["receipt_sha256"],
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        "counts": dict(sorted(counts.items())),
        "measurements": measurements,
        "requirements": requirements,
        "failed_requirements": failed,
        "coverage_gate_passed": not failed,
        "cross_domain_metadata_is_verified_bridge_evidence": False,
        "verified_synthetic_bridge_component_complete": False,
        "selected_tokenizer_allocation_complete": False,
        "source_text_persisted": False,
        "final_corpus_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return _publish(output, durable_output, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durable-output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = measure_coverage(
        args.index_root,
        args.output,
        args.durable_output,
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
