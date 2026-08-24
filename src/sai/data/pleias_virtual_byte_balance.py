"""Balance final virtual PleIAs locators against exact private-book bytes.

The upstream PleIAs selection intentionally keeps enough quality-screened rows
to fill the corpus.  Once both components have completed all deletion passes,
this module assigns an exact per-shard PleIAs byte budget and records only the
held-over locator identities.  Source text is never copied into the balance
artifacts.
"""

from __future__ import annotations

import argparse
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
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA as PLEIAS_SCHEMA,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_STATUS as PLEIAS_STATUS,
)
from sai.data.pleias_virtual_cross_source_reconstruction import LOCATOR_SCHEMA
from sai.data.pleias_virtual_cross_source_reconstruction import (
    SHARD_SCHEMA as PLEIAS_SHARD_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ALLOCATION_SCHEMA = "sai-pleias-virtual-byte-allocation-v1"
SHARD_SCHEMA = "sai-pleias-virtual-byte-balance-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-virtual-byte-balance-aggregate-v1"
DEFAULT_BYTE_CEILING = 2_000_000_000_000
DEFAULT_RESERVED_BYTES = 1_000_000_000
MAXIMUM_SINGLE_STRATUM_PPM = 200_000


class PleiasVirtualByteBalanceError(RuntimeError):
    """Final component custody, selection, or byte accounting differs."""


def _regular_database(path: Path, descriptor: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PleiasVirtualByteBalanceError("balance database differs")


def _load_allocation(path: Path) -> dict[str, Any]:
    payload = _load_signed(path, ALLOCATION_SCHEMA)
    if (
        payload.get("status") != "complete_nontraining_pleias_virtual_byte_allocation"
        or payload.get("source_text_persisted") is not False
        or payload.get("training_ready") is not False
    ):
        raise PleiasVirtualByteBalanceError("byte allocation differs")
    return payload


def _bound_pleias(final_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate = _load_signed(final_root / "aggregate.json", PLEIAS_SCHEMA)
    shards = aggregate.get("shards", {})
    logical_shards = shards.get("logical_shards")
    if (
        aggregate.get("status") != PLEIAS_STATUS
        or aggregate.get("complete_final_pleias_document_coverage") is not True
        or aggregate.get("benchmark_decontamination_complete") is not True
        or aggregate.get("pleias_internal_subdocument_deduplication_complete")
        is not True
        or aggregate.get("cross_source_subdocument_deduplication_complete") is not True
        or aggregate.get("source_text_persisted") is not False
        or isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or logical_shards <= 0
    ):
        raise PleiasVirtualByteBalanceError("final PleIAs aggregate differs")
    receipts = []
    for index in range(logical_shards):
        receipt = _load_signed(
            final_root / "shards" / f"shard_{index:05d}" / "receipt.json",
            PLEIAS_SHARD_SCHEMA,
        )
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != index
            or receipt.get("complete_final_pleias_document_coverage") is not True
            or receipt.get("source_text_persisted") is not False
        ):
            raise PleiasVirtualByteBalanceError("final PleIAs shard differs")
        receipts.append(receipt)
    if shards.get("ordered_receipts_sha256") != canonical_sha256(
        [receipt["receipt_sha256"] for receipt in receipts]
    ):
        raise PleiasVirtualByteBalanceError("PleIAs shard custody differs")
    return aggregate, receipts


def build_allocation(
    book_aggregate_path: Path,
    pleias_final_root: Path,
    output: Path,
    *,
    byte_ceiling: int = DEFAULT_BYTE_CEILING,
    reserved_bytes: int = DEFAULT_RESERVED_BYTES,
) -> dict[str, Any]:
    """Allocate the exact residual byte ceiling proportionally across shards."""

    if (
        output.exists()
        or output.is_symlink()
        or isinstance(byte_ceiling, bool)
        or not isinstance(byte_ceiling, int)
        or byte_ceiling <= 0
        or isinstance(reserved_bytes, bool)
        or not isinstance(reserved_bytes, int)
        or reserved_bytes < 0
        or reserved_bytes >= byte_ceiling
    ):
        raise PleiasVirtualByteBalanceError("allocation arguments differ")
    books = _load_signed(book_aggregate_path, BOOK_SCHEMA)
    pleias, receipts = _bound_pleias(pleias_final_root)
    book_bytes = books.get("totals", {}).get("output_text_utf8_bytes")
    candidate_bytes = pleias.get("totals", {}).get("output_text_utf8_bytes")
    if (
        books.get("benchmark_decontamination_complete") is not True
        or books.get("cross_source_subdocument_deduplication_complete") is not True
        or books.get("private_storage_only") is not True
        or books.get("huggingface_redistribution_authorized") is not False
        or isinstance(book_bytes, bool)
        or not isinstance(book_bytes, int)
        or book_bytes <= 0
        or isinstance(candidate_bytes, bool)
        or not isinstance(candidate_bytes, int)
        or candidate_bytes <= 0
    ):
        raise PleiasVirtualByteBalanceError("final component bytes differ")
    available = byte_ceiling - reserved_bytes - book_bytes
    if available <= 0:
        raise PleiasVirtualByteBalanceError("books exhaust the corpus byte ceiling")
    admitted_ceiling = min(candidate_bytes, available)
    shard_bytes = [
        receipt.get("counts", {}).get("output_text_utf8_bytes") for receipt in receipts
    ]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shard_bytes
        )
        or sum(shard_bytes) != candidate_bytes
    ):
        raise PleiasVirtualByteBalanceError("PleIAs shard byte accounting differs")
    budgets = [admitted_ceiling * value // candidate_bytes for value in shard_bytes]
    remainder = admitted_ceiling - sum(budgets)
    order = sorted(
        range(len(shard_bytes)),
        key=lambda index: (
            -(admitted_ceiling * shard_bytes[index] % candidate_bytes),
            index,
        ),
    )
    for index in order[:remainder]:
        budgets[index] += 1
    allocations = [
        {
            "shard_index": index,
            "candidate_text_utf8_bytes": shard_bytes[index],
            "maximum_selected_text_utf8_bytes": budgets[index],
            "source_shard_receipt_sha256": receipts[index]["receipt_sha256"],
        }
        for index in range(len(receipts))
    ]
    payload = {
        "schema": ALLOCATION_SCHEMA,
        "status": "complete_nontraining_pleias_virtual_byte_allocation",
        "source": {
            "book_aggregate_receipt_sha256": books["receipt_sha256"],
            "pleias_aggregate_receipt_sha256": pleias["receipt_sha256"],
        },
        "policy": {
            "byte_ceiling": byte_ceiling,
            "reserved_bytes": reserved_bytes,
            "reserved_bytes_purpose": "verified_synthetic_bridge_and_rounding_headroom",
            "book_text_utf8_bytes": book_bytes,
            "maximum_pleias_text_utf8_bytes": admitted_ceiling,
            "ceiling_is_not_a_padding_floor": True,
            "quality_ranked_selection_required": True,
        },
        "candidate_pleias_text_utf8_bytes": candidate_bytes,
        "allocations": allocations,
        "ordered_allocations_sha256": canonical_sha256(allocations),
        "source_text_persisted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def _locator_path(final_root: Path, receipt: dict[str, Any], shard_index: int) -> Path:
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
        raise PleiasVirtualByteBalanceError("final locator file differs")
    return path


def select_shard(
    pleias_final_root: Path,
    allocation_path: Path,
    output_root: Path,
    *,
    logical_shards: int,
    shard_index: int,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Select one source-text-free locator shard under its exact byte budget."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasVirtualByteBalanceError("shard arguments differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualByteBalanceError("pyarrow is required") from error
    aggregate, receipts = _bound_pleias(pleias_final_root)
    if len(receipts) != logical_shards:
        raise PleiasVirtualByteBalanceError("logical shard count differs")
    allocation = _load_allocation(allocation_path)
    if allocation.get("source", {}).get(
        "pleias_aggregate_receipt_sha256"
    ) != aggregate.get("receipt_sha256") or allocation.get(
        "ordered_allocations_sha256"
    ) != canonical_sha256(allocation.get("allocations")):
        raise PleiasVirtualByteBalanceError("allocation lineage differs")
    assigned = allocation["allocations"][shard_index]
    source_receipt = receipts[shard_index]
    if assigned.get("shard_index") != shard_index or assigned.get(
        "source_shard_receipt_sha256"
    ) != source_receipt.get("receipt_sha256"):
        raise PleiasVirtualByteBalanceError("shard allocation differs")
    budget = assigned.get("maximum_selected_text_utf8_bytes")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise PleiasVirtualByteBalanceError("shard byte budget differs")
    locator_path = _locator_path(pleias_final_root, source_receipt, shard_index)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-pleias-byte-balance-", dir=scratch_root
        ) as directory:
            state = sqlite3.connect(Path(directory) / "candidates.sqlite3")
            state.execute("PRAGMA journal_mode=DELETE")
            state.execute("PRAGMA synchronous=OFF")
            state.execute("PRAGMA temp_store=FILE")
            state.execute(
                "CREATE TABLE candidates (identity TEXT PRIMARY KEY, bytes INTEGER "
                "NOT NULL, stratum TEXT NOT NULL, quality_floor INTEGER NOT NULL, "
                "quality_mean INTEGER NOT NULL, locator_json TEXT NOT NULL) "
                "WITHOUT ROWID"
            )
            candidate_counts: Counter[str] = Counter()
            for batch in pq.ParquetFile(locator_path).iter_batches(
                batch_size=4096, use_threads=False
            ):
                for locator in batch.to_pylist():
                    unsigned = {
                        key: value
                        for key, value in locator.items()
                        if key != "locator_sha256"
                    }
                    size = locator.get("output_text_utf8_bytes")
                    identity = locator.get("source_row_identity_sha256")
                    if (
                        locator.get("schema") != LOCATOR_SCHEMA
                        or locator.get("locator_sha256") != canonical_sha256(unsigned)
                        or locator.get("training_ready") is not False
                        or not isinstance(identity, str)
                        or len(identity) != 64
                        or isinstance(size, bool)
                        or not isinstance(size, int)
                        or size <= 0
                    ):
                        raise PleiasVirtualByteBalanceError("final locator differs")
                    state.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            identity,
                            size,
                            locator["semantic_stratum"],
                            locator["semantic_quality_floor_milli"],
                            locator["semantic_quality_mean_milli"],
                            json.dumps(locator, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                    candidate_counts["documents"] += 1
                    candidate_counts["output_text_utf8_bytes"] += size
            state.commit()
            if candidate_counts["documents"] != source_receipt.get("counts", {}).get(
                "documents"
            ) or candidate_counts["output_text_utf8_bytes"] != assigned.get(
                "candidate_text_utf8_bytes"
            ):
                raise PleiasVirtualByteBalanceError("candidate coverage differs")
            state.execute(
                "CREATE TABLE selected (identity TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            selected_bytes = 0
            strata = state.execute(
                "SELECT COUNT(DISTINCT stratum) FROM candidates"
            ).fetchone()[0]
            stratum_cap = min(
                budget * MAXIMUM_SINGLE_STRATUM_PPM // 1_000_000,
                budget // strata,
            )
            by_stratum: Counter[str] = Counter()
            ranked = (
                "SELECT identity, bytes, stratum FROM candidates ORDER BY "
                "quality_floor DESC, quality_mean DESC, identity"
            )
            for identity, size, stratum in state.execute(ranked):
                if (
                    selected_bytes + size <= budget
                    and by_stratum[stratum] + size <= stratum_cap
                ):
                    state.execute("INSERT INTO selected VALUES (?)", (identity,))
                    selected_bytes += size
                    by_stratum[stratum] += size
            refill = (
                "SELECT c.identity, c.bytes, c.stratum FROM candidates c LEFT JOIN "
                "selected s ON s.identity=c.identity WHERE s.identity IS NULL ORDER BY "
                "c.quality_floor DESC, c.quality_mean DESC, c.identity"
            )
            for identity, size, stratum in state.execute(refill):
                if selected_bytes + size <= budget:
                    state.execute("INSERT INTO selected VALUES (?)", (identity,))
                    selected_bytes += size
                    by_stratum[stratum] += size
            state.commit()
            exclusions = sqlite3.connect(stage / "excluded.sqlite3")
            exclusions.execute("PRAGMA journal_mode=DELETE")
            exclusions.execute("PRAGMA synchronous=FULL")
            exclusions.execute(
                "CREATE TABLE excluded (source_row_identity_sha256 TEXT PRIMARY KEY) "
                "WITHOUT ROWID"
            )
            selected_counts: Counter[str] = Counter()
            excluded_rows = 0
            for identity, encoded, chosen in state.execute(
                "SELECT c.identity, c.locator_json, s.identity IS NOT NULL FROM "
                "candidates c LEFT JOIN selected s ON s.identity=c.identity "
                "ORDER BY c.identity"
            ):
                locator = json.loads(encoded)
                if not chosen:
                    exclusions.execute("INSERT INTO excluded VALUES (?)", (identity,))
                    excluded_rows += 1
                    continue
                size = locator["output_text_utf8_bytes"]
                selected_counts["documents"] += 1
                selected_counts["output_text_utf8_bytes"] += size
                selected_counts[f"split::{locator['corpus_split']}::documents"] += 1
                selected_counts[
                    f"split::{locator['corpus_split']}::text_utf8_bytes"
                ] += size
                selected_counts[
                    f"semantic_stratum::{locator['semantic_stratum']}::documents"
                ] += 1
                selected_counts[
                    f"quality_floor_milli::{locator['semantic_quality_floor_milli']}::documents"
                ] += 1
                selected_counts[
                    f"difficulty_mean_milli::{locator['semantic_difficulty_mean_milli']}::documents"
                ] += 1
                selected_counts[
                    f"curriculum_phase::{locator['semantic_curriculum_phase']}::documents"
                ] += 1
                for domain in locator["semantic_domains"]:
                    selected_counts[f"semantic_domain::{domain}::documents"] += 1
            exclusions.commit()
            exclusions.execute("VACUUM")
            exclusions.close()
            state.close()
        database_path = stage / "excluded.sqlite3"
        descriptor = {
            "path": database_path.name,
            "bytes": database_path.stat().st_size,
            "sha256": sha256_file(database_path),
            "rows": excluded_rows,
        }
        payload = {
            "schema": SHARD_SCHEMA,
            "status": "complete_nontraining_pleias_virtual_byte_balance_shard",
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "source": {
                "allocation_receipt_sha256": allocation["receipt_sha256"],
                "pleias_shard_receipt_sha256": source_receipt["receipt_sha256"],
            },
            "policy": {
                "maximum_selected_text_utf8_bytes": budget,
                "maximum_single_stratum_ppm_first_pass": MAXIMUM_SINGLE_STRATUM_PPM,
                "selection_rank": "quality_floor_desc_quality_mean_desc_identity_asc",
                "second_pass_refill": True,
            },
            "candidate_counts": dict(sorted(candidate_counts.items())),
            "selected_counts": dict(sorted(selected_counts.items())),
            "excluded_rows": excluded_rows,
            "exclusion_database": descriptor,
            "byte_ceiling_respected": selected_counts["output_text_utf8_bytes"]
            <= budget,
            "padding_performed": False,
            "source_text_persisted": False,
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


def aggregate_shards(
    allocation_path: Path,
    shards_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify every balance shard and seal exact selected PleIAs totals."""

    if output.exists() or output.is_symlink():
        raise PleiasVirtualByteBalanceError("aggregate output exists")
    allocation = _load_allocation(allocation_path)
    allocations = allocation.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise PleiasVirtualByteBalanceError("allocation shards differ")
    totals: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    receipts = []
    for index, assigned in enumerate(allocations):
        root = shards_root / "shards" / f"shard_{index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        descriptor = receipt.get("exclusion_database")
        path = (
            root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
        )
        if (
            receipt.get("status")
            != "complete_nontraining_pleias_virtual_byte_balance_shard"
            or receipt.get("logical_shards") != len(allocations)
            or receipt.get("shard_index") != index
            or receipt.get("source", {}).get("allocation_receipt_sha256")
            != allocation["receipt_sha256"]
            or receipt.get("source", {}).get("pleias_shard_receipt_sha256")
            != assigned.get("source_shard_receipt_sha256")
            or receipt.get("policy", {}).get("maximum_selected_text_utf8_bytes")
            != assigned.get("maximum_selected_text_utf8_bytes")
            or receipt.get("byte_ceiling_respected") is not True
            or receipt.get("source_text_persisted") is not False
            or not isinstance(descriptor, dict)
        ):
            raise PleiasVirtualByteBalanceError("balance shard differs")
        _regular_database(path, descriptor)
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as database:
            rows = database.execute("SELECT COUNT(*) FROM excluded").fetchone()[0]
        if rows != descriptor.get("rows") or rows != receipt.get("excluded_rows"):
            raise PleiasVirtualByteBalanceError("excluded identity coverage differs")
        totals.update(receipt.get("selected_counts", {}))
        candidates.update(receipt.get("candidate_counts", {}))
        receipts.append(receipt["receipt_sha256"])
    maximum = allocation["policy"]["maximum_pleias_text_utf8_bytes"]
    if (
        candidates["output_text_utf8_bytes"]
        != allocation.get("candidate_pleias_text_utf8_bytes")
        or totals["output_text_utf8_bytes"] > maximum
        or totals["documents"] <= 0
        or totals["split::train::documents"] <= 0
        or totals["split::development::documents"] <= 0
    ):
        raise PleiasVirtualByteBalanceError("aggregate byte accounting differs")
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_virtual_byte_balance",
        "source": {
            "allocation_receipt_sha256": allocation["receipt_sha256"],
            **allocation["source"],
        },
        "shards": {
            "logical_shards": len(allocations),
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "candidate_counts": dict(sorted(candidates.items())),
        "selected_counts": dict(sorted(totals.items())),
        "maximum_selected_text_utf8_bytes": maximum,
        "remaining_pleias_byte_headroom": maximum - totals["output_text_utf8_bytes"],
        "byte_ceiling_respected": True,
        "padding_performed": False,
        "source_text_persisted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def excluded_database(
    balance_root: Path, shard_index: int
) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """Open one verified read-only exclusion database for a consumer."""

    aggregate = _load_signed(balance_root / "aggregate.json", AGGREGATE_SCHEMA)
    logical_shards = aggregate.get("shards", {}).get("logical_shards")
    if (
        aggregate.get("status") != "complete_nontraining_pleias_virtual_byte_balance"
        or aggregate.get("source_text_persisted") is not False
        or aggregate.get("training_ready") is not False
        or not isinstance(logical_shards, int)
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasVirtualByteBalanceError("balance aggregate differs")
    receipts = []
    target = None
    target_root = None
    for index in range(logical_shards):
        root = balance_root / "shards" / f"shard_{index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        receipts.append(receipt["receipt_sha256"])
        if index == shard_index:
            target = receipt
            target_root = root
    if (
        aggregate.get("shards", {}).get("ordered_receipts_sha256")
        != canonical_sha256(receipts)
        or target is None
        or target_root is None
        or target.get("status")
        != "complete_nontraining_pleias_virtual_byte_balance_shard"
        or target.get("logical_shards") != logical_shards
        or target.get("shard_index") != shard_index
        or target.get("source_text_persisted") is not False
        or target.get("training_ready") is not False
    ):
        raise PleiasVirtualByteBalanceError("balance shard custody differs")
    descriptor = target.get("exclusion_database")
    path = (
        target_root / descriptor.get("path", "")
        if isinstance(descriptor, dict)
        else target_root
    )
    if not isinstance(descriptor, dict):
        raise PleiasVirtualByteBalanceError("balance database descriptor differs")
    _regular_database(path, descriptor)
    database = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    rows = database.execute("SELECT COUNT(*) FROM excluded").fetchone()[0]
    if rows != descriptor.get("rows") or rows != target.get("excluded_rows"):
        database.close()
        raise PleiasVirtualByteBalanceError("excluded identity coverage differs")
    return database, target


def is_excluded(database: sqlite3.Connection, identity: str) -> bool:
    """Return whether an exact final locator was held over by the byte ceiling."""

    return (
        database.execute(
            "SELECT 1 FROM excluded WHERE source_row_identity_sha256=?", (identity,)
        ).fetchone()
        is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    allocation = commands.add_parser("allocate")
    allocation.add_argument("--book-aggregate", type=Path, required=True)
    allocation.add_argument("--pleias-final-root", type=Path, required=True)
    allocation.add_argument("--output", type=Path, required=True)
    allocation.add_argument("--byte-ceiling", type=int, default=DEFAULT_BYTE_CEILING)
    allocation.add_argument(
        "--reserved-bytes", type=int, default=DEFAULT_RESERVED_BYTES
    )
    shard = commands.add_parser("shard")
    shard.add_argument("--pleias-final-root", type=Path, required=True)
    shard.add_argument("--allocation", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--allocation", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "allocate":
        result = build_allocation(
            args.book_aggregate,
            args.pleias_final_root,
            args.output,
            byte_ceiling=args.byte_ceiling,
            reserved_bytes=args.reserved_bytes,
        )
    elif args.command == "shard":
        result = select_shard(
            args.pleias_final_root,
            args.allocation,
            args.output_root,
            logical_shards=args.logical_shards,
            shard_index=args.shard_index,
            scratch_root=args.scratch_root,
        )
    else:
        result = aggregate_shards(args.allocation, args.shards_root, args.output)
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
