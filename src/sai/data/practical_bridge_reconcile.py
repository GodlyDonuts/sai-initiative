"""Reconcile verified bridge lessons with the practical training foundation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.grounded_bridge_curriculum_candidates import (
    RECEIPT_SCHEMA as CANDIDATE_RECEIPT_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import ROW_SCHEMA
from sai.data.institutional_books_practical_admission import SCHEMA as BOOKS_SCHEMA
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-practical-bridge-foundation-reconciliation-v1"
STATUS = "complete_practical_bridge_foundation_reconciliation"


class PracticalBridgeReconcileError(RuntimeError):
    """A bridge, foundation, split, or exact-content invariant differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PracticalBridgeReconcileError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PracticalBridgeReconcileError("signed input differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise PracticalBridgeReconcileError("signed input differs")
    return payload


def _hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _candidate_rows(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _load_signed(root / "receipt.json", CANDIDATE_RECEIPT_SCHEMA)
    descriptor = receipt.get("curriculum_candidates")
    if (
        receipt.get("status")
        != "complete_nontraining_grounded_bridge_curriculum_candidates"
        or receipt.get("independent_model_family_verification_complete") is not True
        or receipt.get("benchmark_decontamination_complete") is not True
        or receipt.get("bridge_pair_disjoint_split_complete") is not True
        or receipt.get("global_deduplication_against_foundation_complete") is not False
        or receipt.get("transfer_ablation_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise PracticalBridgeReconcileError("bridge candidate receipt differs")
    path = root / str(descriptor.get("path"))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PracticalBridgeReconcileError("bridge candidate file differs")
    rows = []
    identities = []
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if (
                    row.get("schema") != ROW_SCHEMA
                    or row.get("corpus_split") not in {"train", "development"}
                    or not _hex(row.get("record_sha256"))
                    or not _hex(row.get("pair_identity_sha256"))
                    or not _hex(row.get("content_sha256"))
                    or not isinstance(row.get("anchor_source_content_sha256s"), list)
                    or len(row["anchor_source_content_sha256s"]) != 2
                    or any(
                        not _hex(value)
                        for value in row["anchor_source_content_sha256s"]
                    )
                    or row.get("training_ready") is not False
                    or row["record_sha256"]
                    != canonical_sha256(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "record_sha256"
                        }
                    )
                ):
                    raise PracticalBridgeReconcileError("bridge candidate row differs")
                rows.append(row)
                identities.append(row["record_sha256"])
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PracticalBridgeReconcileError("bridge candidate rows differ") from error
    if (
        len(rows) != descriptor.get("rows")
        or len(identities) != len(set(identities))
        or canonical_sha256(identities) != descriptor.get("ordered_records_sha256")
    ):
        raise PracticalBridgeReconcileError("bridge candidate coverage differs")
    return receipt, rows


def _targets(rows: list[dict[str, Any]]) -> set[str]:
    values = {row["content_sha256"] for row in rows}
    for row in rows:
        values.update(row["anchor_source_content_sha256s"])
    return values


def _book_hits(
    receipt_path: Path, targets: set[str]
) -> tuple[dict[str, Any], set[str]]:
    receipt = _load_signed(receipt_path, BOOKS_SCHEMA)
    descriptor = receipt.get("manifest")
    if (
        receipt.get("training_ready") is not True
        or receipt.get("practical_pretraining_ready") is not True
        or not isinstance(descriptor, dict)
    ):
        raise PracticalBridgeReconcileError("Books foundation differs")
    path = receipt_path.parent / str(descriptor.get("path"))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PracticalBridgeReconcileError("Books manifest differs")
    hits: set[str] = set()
    rows = 0
    ordered = []
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                value = row.get("source_content_sha256")
                if not _hex(value) or not _hex(row.get("record_sha256")):
                    raise PracticalBridgeReconcileError("Books manifest row differs")
                rows += 1
                ordered.append(row["record_sha256"])
                if value in targets:
                    hits.add(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PracticalBridgeReconcileError("Books manifest rows differ") from error
    if rows != descriptor.get("rows") or canonical_sha256(ordered) != descriptor.get(
        "ordered_records_sha256"
    ):
        raise PracticalBridgeReconcileError("Books manifest coverage differs")
    return receipt, hits


def _pleias_hits(
    receipt_path: Path, targets: set[str]
) -> tuple[dict[str, Any], set[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PracticalBridgeReconcileError("pyarrow is required") from error
    receipt = _load_signed(receipt_path, PLEIAS_SCHEMA)
    descriptors = receipt.get("outputs", {}).get("descriptors")
    if (
        receipt.get("training_ready") is not True
        or receipt.get("global_exact_content_deduplication_complete") is not True
        or not isinstance(descriptors, list)
        or not descriptors
    ):
        raise PracticalBridgeReconcileError("PleIAs foundation differs")
    hits: set[str] = set()
    rows = 0
    for descriptor in descriptors:
        path = receipt_path.parent / str(descriptor.get("path"))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise PracticalBridgeReconcileError("PleIAs locator differs")
        parquet = pq.ParquetFile(path)
        shard_rows = 0
        for batch in parquet.iter_batches(
            columns=["content_sha256"], batch_size=65_536
        ):
            for value in batch.column(0).to_pylist():
                if not _hex(value):
                    raise PracticalBridgeReconcileError("PleIAs content hash differs")
                shard_rows += 1
                if value in targets:
                    hits.add(value)
        if shard_rows != descriptor.get("rows"):
            raise PracticalBridgeReconcileError("PleIAs locator rows differ")
        rows += shard_rows
    if rows != receipt.get("counts", {}).get("admitted_rows"):
        raise PracticalBridgeReconcileError("PleIAs foundation coverage differs")
    return receipt, hits


def reconcile(
    candidate_root: Path,
    books_receipt_path: Path,
    pleias_receipt_path: Path,
    output_root: Path,
    durable_receipt: Path,
) -> dict[str, Any]:
    """Create exact-deduplicated train and source-disjoint development streams."""

    if output_root.exists() or output_root.is_symlink() or durable_receipt.exists():
        raise PracticalBridgeReconcileError("reconciliation output exists")
    candidate_receipt, rows = _candidate_rows(candidate_root)
    targets = _targets(rows)
    books, books_hits = _book_hits(books_receipt_path, targets)
    pleias, pleias_hits = _pleias_hits(pleias_receipt_path, targets)
    foundation_hits = books_hits | pleias_hits
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_identity_sha256"]].append(row)
    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    held = []
    counts: Counter[str] = Counter()
    pair_decisions = []
    for pair_identity in sorted(pairs):
        pair = pairs[pair_identity]
        provisional = {row["corpus_split"] for row in pair}
        anchors = set(pair[0]["anchor_source_content_sha256s"])
        if len(provisional) != 1 or any(
            set(row["anchor_source_content_sha256s"]) != anchors for row in pair
        ):
            raise PracticalBridgeReconcileError("bridge pair geometry differs")
        anchor_hits = sorted(anchors & foundation_hits)
        split = provisional.pop()
        if split == "development" and anchor_hits:
            split = "train"
            counts["development_pairs_promoted_to_train"] += 1
        duplicate_rows = 0
        for row in pair:
            if row["content_sha256"] in foundation_hits:
                duplicate_rows += 1
                held.append(
                    {
                        "record_sha256": row["record_sha256"],
                        "pair_identity_sha256": pair_identity,
                        "content_sha256": row["content_sha256"],
                        "reason": "exact_content_present_in_practical_foundation",
                    }
                )
                continue
            value = dict(row)
            value["provisional_corpus_split"] = row["corpus_split"]
            value["corpus_split"] = split
            value["exact_content_disjoint_from_practical_foundation"] = True
            value["development_source_disjoint_from_practical_foundation"] = (
                split != "development" or not anchor_hits
            )
            value["global_exact_content_deduplication_complete"] = True
            value["source_disjoint_split_complete"] = True
            value["training_ready"] = False
            value["record_sha256"] = canonical_sha256(
                {key: item for key, item in value.items() if key != "record_sha256"}
            )
            output_rows[split].append(value)
            counts[f"{split}_documents"] += 1
            counts[f"{split}_text_utf8_bytes"] += value["text_utf8_bytes"]
        counts["exact_duplicate_documents_held"] += duplicate_rows
        pair_decisions.append(
            {
                "pair_identity_sha256": pair_identity,
                "provisional_split": next(iter({row["corpus_split"] for row in pair})),
                "reconciled_split": split,
                "foundation_anchor_hits": anchor_hits,
                "exact_duplicate_documents_held": duplicate_rows,
            }
        )
        counts[f"{split}_pairs"] += 1
    development_anchors = {
        value
        for row in output_rows["development"]
        for value in row["anchor_source_content_sha256s"]
    }
    if development_anchors & foundation_hits:
        raise PracticalBridgeReconcileError("development source overlap differs")
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        descriptors = {}
        for split, split_rows in output_rows.items():
            path = stage / f"{split}.jsonl"
            _atomic_jsonl(path, split_rows)
            descriptors[split] = {
                "path": path.name,
                "rows": len(split_rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in split_rows]
                ),
            }
        held_path = stage / "held_exact_duplicates.jsonl"
        decisions_path = stage / "pair_decisions.jsonl"
        _atomic_jsonl(held_path, held)
        _atomic_jsonl(decisions_path, pair_decisions)
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "inputs": {
                "candidate_receipt_sha256": candidate_receipt["receipt_sha256"],
                "books_receipt_sha256": books["receipt_sha256"],
                "pleias_receipt_sha256": pleias["receipt_sha256"],
            },
            "foundation_target_hashes": len(targets),
            "foundation_matching_hashes": len(foundation_hits),
            "counts": dict(sorted(counts.items())),
            "outputs": descriptors,
            "held_exact_duplicates": {
                "path": held_path.name,
                "rows": len(held),
                "bytes": held_path.stat().st_size,
                "sha256": sha256_file(held_path),
            },
            "pair_decisions": {
                "path": decisions_path.name,
                "rows": len(pair_decisions),
                "bytes": decisions_path.stat().st_size,
                "sha256": sha256_file(decisions_path),
            },
            "foundation_overlap_reconciliation_complete": True,
            "global_exact_content_deduplication_complete": True,
            "development_source_disjoint_against_foundation_complete": True,
            "normalized_or_semantic_foundation_deduplication_complete": False,
            "transfer_ablation_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        durable_receipt.parent.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_create(durable_receipt, payload)
        except BaseException:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--books-receipt", type=Path, required=True)
    parser.add_argument("--pleias-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(
        args.candidate_root,
        args.books_receipt,
        args.pleias_receipt,
        args.output_root,
        args.durable_receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
