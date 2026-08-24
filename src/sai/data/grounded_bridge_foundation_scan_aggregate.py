"""Reconcile complete foundation scans into conservative bridge decisions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.grounded_bridge_foundation_query import (
    DATABASE_SCHEMA as QUERY_DATABASE_SCHEMA,
)
from sai.data.grounded_bridge_foundation_scan import (
    ANCHOR_MATCH_SCHEMA,
    QueryBoundary,
)
from sai.data.grounded_bridge_foundation_scan import SCHEMA as SCAN_SCHEMA
from sai.data.grounded_bridge_foundation_scan import STATUS as SCAN_STATUS
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-bridge-foundation-scan-aggregate-v1"
STATUS = "complete_nontraining_grounded_bridge_foundation_scan_aggregate"
DOCUMENT_DECISION_SCHEMA = "sai-grounded-bridge-foundation-document-decision-v1"
PAIR_DECISION_SCHEMA = "sai-grounded-bridge-foundation-pair-decision-v1"


class GroundedBridgeFoundationScanAggregateError(RuntimeError):
    """A scan shard, query owner, anchor split, or decision differs."""


def _binary_digests(root: Path, descriptor: Any) -> tuple[bytes, ...]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("record_bytes") != 32
        or descriptor.get("source_text_persisted") is not False
        or descriptor.get("bytes") != descriptor.get("rows", -1) * 32
    ):
        raise GroundedBridgeFoundationScanAggregateError(
            "scan digest descriptor differs"
        )
    path = _bound_file(root, descriptor)
    raw = path.read_bytes()
    rows = tuple(raw[index : index + 32] for index in range(0, len(raw), 32))
    if len(rows) != descriptor.get("rows") or any(
        current <= previous for previous, current in zip(rows, rows[1:], strict=False)
    ):
        raise GroundedBridgeFoundationScanAggregateError("scan digest ordering differs")
    return rows


def _anchor_matches(root: Path, descriptor: Any, receipt: dict[str, Any]) -> list[dict]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("source_text_persisted") is not False
    ):
        raise GroundedBridgeFoundationScanAggregateError("anchor descriptor differs")
    path = _bound_file(root, descriptor)
    rows = []
    ordered = []
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise GroundedBridgeFoundationScanAggregateError(
                    "anchor match row differs"
                ) from error
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if (
                row.get("schema") != ANCHOR_MATCH_SCHEMA
                or row.get("record_sha256") != canonical_sha256(unsigned)
                or row.get("component") != receipt["component"]
                or row.get("logical_shards") != receipt["logical_shards"]
                or row.get("shard_index") != receipt["shard_index"]
                or row.get("foundation_split") not in {"train", "development"}
                or row.get("source_text_persisted") is not False
                or row.get("training_ready") is not False
                or not isinstance(row.get("match_types"), list)
                or not row["match_types"]
                or row["match_types"] != sorted(set(row["match_types"]))
                or any(
                    value not in {"source_content_sha256", "source_key_sha256"}
                    for value in row["match_types"]
                )
            ):
                raise GroundedBridgeFoundationScanAggregateError(
                    "anchor match row differs"
                )
            rows.append(row)
            ordered.append(row["record_sha256"])
    if len(rows) != descriptor.get("rows") or canonical_sha256(
        ordered
    ) != descriptor.get("ordered_records_sha256"):
        raise GroundedBridgeFoundationScanAggregateError(
            "anchor match coverage differs"
        )
    return rows


def _load_scan(
    root: Path,
    *,
    component: str,
    logical_shards: int,
    shard_index: int,
    query_receipt_sha256: str,
) -> tuple[dict[str, Any], tuple[bytes, ...], tuple[bytes, ...], list[dict]]:
    receipt = _load_receipt(root / "receipt.json")
    if (
        receipt.get("schema") != SCAN_SCHEMA
        or receipt.get("status") != SCAN_STATUS
        or receipt.get("component") != component
        or receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("source_query_receipt_sha256") != query_receipt_sha256
        or receipt.get("source_text_persisted") is not False
        or receipt.get("foundation_shard_scan_complete") is not True
        or receipt.get("global_foundation_scan_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(receipt.get("source_custody"), dict)
        or not receipt["source_custody"]
        or receipt.get("counts", {}).get("documents", 0) <= 0
    ):
        raise GroundedBridgeFoundationScanAggregateError(
            "foundation scan receipt differs"
        )
    words = _binary_digests(root, receipt.get("matched_word_digests"))
    code = _binary_digests(root, receipt.get("matched_code_digests"))
    anchors = _anchor_matches(root, receipt.get("anchor_matches"), receipt)
    counts = receipt["counts"]
    if (
        len(words) != counts.get("unique_matched_word_signatures")
        or len(code) != counts.get("unique_matched_code_signatures")
        or len(anchors) != counts.get("anchor_match_records")
    ):
        raise GroundedBridgeFoundationScanAggregateError(
            "scan output accounting differs"
        )
    return receipt, words, code, anchors


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = []
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            ordered.append(row["record_sha256"])
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "ordered_records_sha256": canonical_sha256(ordered),
        "source_text_persisted": False,
    }


def aggregate_scans(
    query_root: Path,
    pleias_scan_root: Path,
    book_scan_root: Path,
    output_root: Path,
    *,
    pleias_logical_shards: int = 128,
    book_logical_shards: int = 64,
    durable_receipt: Path | None = None,
) -> dict[str, Any]:
    """Prove all-shard coverage and decide every bridge document and pair."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or pleias_logical_shards <= 0
        or book_logical_shards <= 0
        or (
            durable_receipt is not None
            and (durable_receipt.exists() or durable_receipt.is_symlink())
        )
    ):
        raise GroundedBridgeFoundationScanAggregateError("aggregate arguments differ")
    boundary = QueryBoundary(query_root)
    query_receipt = boundary.receipt
    scan_receipts: dict[str, list[str]] = defaultdict(list)
    matched_word: set[bytes] = set()
    matched_code: set[bytes] = set()
    all_anchor_matches = []
    foundation_counts: Counter[str] = Counter()
    for component, root, logical_shards in (
        ("pleias_common_corpus", pleias_scan_root, pleias_logical_shards),
        ("institutional_books", book_scan_root, book_logical_shards),
    ):
        for shard_index in range(logical_shards):
            shard_root = root / "shards" / f"shard_{shard_index:05d}"
            receipt, words, code, anchors = _load_scan(
                shard_root,
                component=component,
                logical_shards=logical_shards,
                shard_index=shard_index,
                query_receipt_sha256=query_receipt["receipt_sha256"],
            )
            scan_receipts[component].append(receipt["receipt_sha256"])
            matched_word.update(words)
            matched_code.update(code)
            all_anchor_matches.extend(anchors)
            for key, value in receipt["counts"].items():
                if isinstance(value, int):
                    foundation_counts[f"{component}::{key}"] += value

    descriptor = query_receipt["query_database"]
    database_path = _bound_file(query_root, descriptor)
    if descriptor.get("schema") != QUERY_DATABASE_SCHEMA:
        raise GroundedBridgeFoundationScanAggregateError("query database differs")
    database = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        documents = {
            row[0]: {
                "document_identity_sha256": row[0],
                "pair_identity_sha256": row[1],
                "provisional_split": row[2],
                "candidate_record_sha256": row[3],
            }
            for row in database.execute(
                "SELECT document_identity_sha256, pair_identity_sha256, "
                "provisional_split, record_sha256 FROM documents "
                "ORDER BY document_identity_sha256"
            )
        }
        anchors = {
            (row[0], row[1]): {
                "candidate_identity_sha256": row[2],
                "provisional_split": row[3],
            }
            for row in database.execute(
                "SELECT pair_identity_sha256, anchor_index, "
                "candidate_identity_sha256, provisional_split FROM anchors "
                "ORDER BY pair_identity_sha256, anchor_index"
            )
        }
        overlap_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for kind, digests, table in (
            ("word", matched_word, "word_signatures"),
            ("code", matched_code, "code_signatures"),
        ):
            for digest in sorted(digests):
                owners = database.execute(
                    f"SELECT document_identity_sha256 FROM {table} WHERE digest=?",
                    (digest,),
                ).fetchall()
                if not owners:
                    raise GroundedBridgeFoundationScanAggregateError(
                        "matched digest has no query owner"
                    )
                for (identity,) in owners:
                    overlap_counts[identity][kind] += 1
    finally:
        database.close()
    if len(documents) != query_receipt.get("counts", {}).get("documents"):
        raise GroundedBridgeFoundationScanAggregateError(
            "query document coverage differs"
        )

    anchor_splits: dict[tuple[str, int], set[str]] = defaultdict(set)
    anchor_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    seen_anchor_records = set()
    for row in all_anchor_matches:
        key = (row["pair_identity_sha256"], row["anchor_index"])
        expected = anchors.get(key)
        if (
            expected is None
            or row["anchor_candidate_identity_sha256"]
            != expected["candidate_identity_sha256"]
            or row["provisional_bridge_split"] != expected["provisional_split"]
            or row["record_sha256"] in seen_anchor_records
        ):
            raise GroundedBridgeFoundationScanAggregateError(
                "anchor query binding differs"
            )
        seen_anchor_records.add(row["record_sha256"])
        anchor_splits[key].add(row["foundation_split"])
        anchor_groups[key].add(row["foundation_source_group_sha256"])

    by_pair: dict[str, list[str]] = defaultdict(list)
    for identity, document in documents.items():
        by_pair[document["pair_identity_sha256"]].append(identity)
    pair_rows = []
    document_rows = []
    counts: Counter[str] = Counter()
    for pair in sorted(by_pair):
        identities = sorted(by_pair[pair])
        provisional = {
            documents[identity]["provisional_split"] for identity in identities
        }
        if len(provisional) != 1:
            raise GroundedBridgeFoundationScanAggregateError(
                "pair provisional split differs"
            )
        provisional_split = provisional.pop()
        splits_by_anchor = {
            str(index): sorted(anchor_splits.get((pair, index), set()))
            for index in (0, 1)
        }
        groups_by_anchor = {
            str(index): sorted(anchor_groups.get((pair, index), set()))
            for index in (0, 1)
        }
        observed_splits = set(splits_by_anchor["0"]) | set(splits_by_anchor["1"])
        conflict = len(observed_splits) > 1
        resolved_split = (
            None if conflict else next(iter(observed_splits), provisional_split)
        )
        overlapped = [identity for identity in identities if overlap_counts[identity]]
        retained = [identity for identity in identities if not overlap_counts[identity]]
        if conflict:
            pair_status = "exclude_anchor_foundation_split_conflict"
        elif not retained:
            pair_status = "exclude_all_representations_overlap_foundation"
        else:
            pair_status = "retain_pending_positive_transfer_ablation"
        pair_row = {
            "schema": PAIR_DECISION_SCHEMA,
            "pair_identity_sha256": pair,
            "provisional_split": provisional_split,
            "resolved_split": resolved_split,
            "foundation_splits_by_anchor": splits_by_anchor,
            "foundation_source_groups_by_anchor": groups_by_anchor,
            "representations": {
                "total": len(identities),
                "exact_overlap": len(overlapped),
                "retained": len(retained) if not conflict else 0,
            },
            "decision": pair_status,
            "source_disjoint_against_foundation_complete": True,
            "global_deduplication_against_foundation_complete": True,
            "positive_transfer_ablation_complete": False,
            "training_ready": False,
        }
        pair_row["record_sha256"] = canonical_sha256(pair_row)
        pair_rows.append(pair_row)
        counts[f"pair_decision::{pair_status}"] += 1
        for identity in identities:
            overlap = overlap_counts[identity]
            if conflict:
                decision = "exclude_pair_anchor_foundation_split_conflict"
            elif pair_status == "exclude_all_representations_overlap_foundation":
                decision = "exclude_pair_all_representations_overlap_foundation"
            elif overlap:
                decision = "hold_exact_foundation_overlap"
            else:
                decision = "retain_pending_positive_transfer_ablation"
            document_row = {
                "schema": DOCUMENT_DECISION_SCHEMA,
                "document_identity_sha256": identity,
                "pair_identity_sha256": pair,
                "candidate_record_sha256": documents[identity][
                    "candidate_record_sha256"
                ],
                "provisional_split": provisional_split,
                "resolved_split": resolved_split,
                "exact_foundation_overlap": {
                    "word_signatures": overlap["word"],
                    "code_signatures": overlap["code"],
                },
                "decision": decision,
                "source_text_persisted": False,
                "positive_transfer_ablation_complete": False,
                "training_ready": False,
            }
            document_row["record_sha256"] = canonical_sha256(document_row)
            document_rows.append(document_row)
            counts[f"document_decision::{decision}"] += 1

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        document_descriptor = _write_jsonl(
            stage / "document_decisions.jsonl", document_rows
        )
        pair_descriptor = _write_jsonl(stage / "pair_decisions.jsonl", pair_rows)
        counts["foundation_documents"] = sum(
            value
            for key, value in foundation_counts.items()
            if key.endswith("::documents") and "::split::" not in key
        )
        counts["foundation_text_utf8_bytes"] = sum(
            value
            for key, value in foundation_counts.items()
            if key.endswith("::text_utf8_bytes") and "::split::" not in key
        )
        counts["unique_matched_word_signatures"] = len(matched_word)
        counts["unique_matched_code_signatures"] = len(matched_code)
        counts["anchor_match_records"] = len(all_anchor_matches)
        counts["bridge_documents"] = len(document_rows)
        counts["bridge_pairs"] = len(pair_rows)
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "source_query_receipt_sha256": query_receipt["receipt_sha256"],
            "source_scan_receipts": {
                component: {
                    "logical_shards": len(receipts),
                    "ordered_receipts_sha256": canonical_sha256(receipts),
                }
                for component, receipts in sorted(scan_receipts.items())
            },
            "counts": dict(sorted(counts.items())),
            "foundation_counts": dict(sorted(foundation_counts.items())),
            "document_decisions": document_descriptor,
            "pair_decisions": pair_descriptor,
            "source_text_persisted": False,
            "global_foundation_scan_complete": True,
            "global_deduplication_against_foundation_complete": True,
            "source_disjoint_against_foundation_complete": True,
            "positive_transfer_ablation_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        if durable_receipt is not None:
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
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--pleias-scan-root", type=Path, required=True)
    parser.add_argument("--book-scan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pleias-logical-shards", type=int, default=128)
    parser.add_argument("--book-logical-shards", type=int, default=64)
    parser.add_argument("--durable-receipt", type=Path)
    args = parser.parse_args()
    result = aggregate_scans(
        args.query_root,
        args.pleias_scan_root,
        args.book_scan_root,
        args.output_root,
        pleias_logical_shards=args.pleias_logical_shards,
        book_logical_shards=args.book_logical_shards,
        durable_receipt=args.durable_receipt,
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
