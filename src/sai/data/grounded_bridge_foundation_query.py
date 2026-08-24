"""Build source-text-free bridge queries for the final foundation scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.decontamination import (
    _CODE,
    _WORD,
    _code_shingles,
    _normalize,
    _shingles,
)
from sai.data.decontamination import (
    POLICY as SHINGLE_POLICY,
)
from sai.data.grounded_bridge_curriculum_candidates import (
    RECEIPT_SCHEMA as CANDIDATE_RECEIPT_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import (
    ROW_SCHEMA as CANDIDATE_ROW_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import (
    STATUS as CANDIDATE_STATUS,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-bridge-foundation-query-v1"
STATUS = "complete_nontraining_grounded_bridge_foundation_query"
DATABASE_SCHEMA = "sai-grounded-bridge-foundation-query-sqlite-v1"


class GroundedBridgeFoundationQueryError(RuntimeError):
    """Bridge candidates, shingle signatures, or anchor registry differs."""


def source_key(source: dict[str, Any]) -> str:
    """Normalize the three stable source fields available across components."""

    if not isinstance(source, dict):
        raise GroundedBridgeFoundationQueryError("bridge anchor source differs")
    dataset = source.get("dataset")
    revision = source.get("revision", "")
    row_id = source.get("row_id")
    if (
        not isinstance(dataset, str)
        or not dataset
        or not isinstance(revision, str)
        or not isinstance(row_id, str)
        or not row_id
    ):
        raise GroundedBridgeFoundationQueryError("bridge anchor source differs")
    return canonical_sha256(
        {"dataset": dataset, "revision": revision, "row_id": row_id}
    )


def _validate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
    text = row.get("text")
    if (
        row.get("schema") != CANDIDATE_ROW_SCHEMA
        or row.get("record_sha256") != canonical_sha256(unsigned)
        or not isinstance(text, str)
        or not text
        or hashlib.sha256(text.encode()).hexdigest() != row.get("content_sha256")
        or hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
        != row.get("normalized_content_sha256")
        or row.get("bridge_pair_disjoint_split_complete") is not True
        or row.get("source_disjoint_against_foundation_complete") is not False
        or row.get("global_deduplication_against_foundation_complete") is not False
        or row.get("bridge_verified") is not False
        or row.get("training_ready") is not False
    ):
        raise GroundedBridgeFoundationQueryError("bridge candidate row differs")
    anchors = row.get("anchor_sources")
    anchor_identities = row.get("anchor_candidate_identity_sha256s")
    anchor_contents = row.get("anchor_source_content_sha256s")
    if (
        not isinstance(anchors, list)
        or len(anchors) != 2
        or not isinstance(anchor_identities, list)
        or len(anchor_identities) != 2
        or not isinstance(anchor_contents, list)
        or len(anchor_contents) != 2
    ):
        raise GroundedBridgeFoundationQueryError("bridge anchor registry differs")
    for source in anchors:
        source_key(source)
    return row


def _database(path: Path) -> sqlite3.Connection:
    database = sqlite3.connect(path)
    database.executescript(
        "PRAGMA journal_mode=DELETE;"
        "PRAGMA synchronous=FULL;"
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) "
        "WITHOUT ROWID;"
        "CREATE TABLE documents (document_identity_sha256 TEXT PRIMARY KEY, "
        "pair_identity_sha256 TEXT NOT NULL, provisional_split TEXT NOT NULL, "
        "record_sha256 TEXT NOT NULL UNIQUE) WITHOUT ROWID;"
        "CREATE TABLE word_signatures (digest BLOB NOT NULL, "
        "document_identity_sha256 TEXT NOT NULL, PRIMARY KEY(digest, "
        "document_identity_sha256)) WITHOUT ROWID;"
        "CREATE TABLE code_signatures (digest BLOB NOT NULL, "
        "document_identity_sha256 TEXT NOT NULL, PRIMARY KEY(digest, "
        "document_identity_sha256)) WITHOUT ROWID;"
        "CREATE TABLE anchors (pair_identity_sha256 TEXT NOT NULL, "
        "anchor_index INTEGER NOT NULL, candidate_identity_sha256 TEXT NOT NULL, "
        "source_content_sha256 TEXT NOT NULL, source_key_sha256 TEXT NOT NULL, "
        "provisional_split TEXT NOT NULL, PRIMARY KEY(pair_identity_sha256, "
        "anchor_index)) WITHOUT ROWID;"
        "CREATE INDEX word_digest ON word_signatures(digest);"
        "CREATE INDEX code_digest ON code_signatures(digest);"
        "CREATE INDEX anchor_content ON anchors(source_content_sha256);"
        "CREATE INDEX anchor_source ON anchors(source_key_sha256);"
    )
    return database


def build_query(
    candidate_root: Path,
    output_root: Path,
    durable_receipt: Path | None = None,
) -> dict[str, Any]:
    """Index every bridge shingle and anchor without retaining source text."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or (
            durable_receipt is not None
            and (durable_receipt.exists() or durable_receipt.is_symlink())
        )
    ):
        raise GroundedBridgeFoundationQueryError("bridge query output differs")
    receipt = _load_receipt(candidate_root / "receipt.json")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    descriptor = receipt.get("curriculum_candidates")
    if (
        receipt.get("schema") != CANDIDATE_RECEIPT_SCHEMA
        or receipt.get("status") != CANDIDATE_STATUS
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned_receipt)
        or receipt.get("source_disjoint_against_foundation_complete") is not False
        or receipt.get("global_deduplication_against_foundation_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedBridgeFoundationQueryError("bridge candidate receipt differs")
    candidate_path = _bound_file(candidate_root, descriptor)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir()
    database_path = stage / "queries.sqlite3"
    database = _database(database_path)
    counts: Counter[str] = Counter()
    ordered_records = []
    ordered_anchors = []
    seen_pairs = set()
    pair_bindings = {}
    try:
        with candidate_path.open() as handle:
            for line in handle:
                row = _validate_candidate(json.loads(line))
                identity = row["document_identity_sha256"]
                pair = row["pair_identity_sha256"]
                split = row["corpus_split"]
                database.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?)",
                    (identity, pair, split, row["record_sha256"]),
                )
                normalized = _normalize(row["text"])
                words = _shingles(
                    _WORD.findall(normalized), SHINGLE_POLICY["word_shingle_tokens"]
                )
                code = _code_shingles(_CODE.findall(normalized))
                database.executemany(
                    "INSERT INTO word_signatures VALUES (?, ?)",
                    ((digest, identity) for digest in sorted(words)),
                )
                database.executemany(
                    "INSERT INTO code_signatures VALUES (?, ?)",
                    ((digest, identity) for digest in sorted(code)),
                )
                counts["documents"] += 1
                counts["word_signature_occurrences"] += len(words)
                counts["code_signature_occurrences"] += len(code)
                counts[f"split::{split}::documents"] += 1
                ordered_records.append(row["record_sha256"])
                binding = canonical_sha256(
                    {
                        "anchor_candidate_identity_sha256s": row[
                            "anchor_candidate_identity_sha256s"
                        ],
                        "anchor_source_content_sha256s": row[
                            "anchor_source_content_sha256s"
                        ],
                        "anchor_sources": row["anchor_sources"],
                        "provisional_split": split,
                    }
                )
                if pair in pair_bindings and pair_bindings[pair] != binding:
                    raise GroundedBridgeFoundationQueryError(
                        "bridge pair anchor binding differs"
                    )
                pair_bindings[pair] = binding
                if pair not in seen_pairs:
                    for index, (anchor_identity, content, source) in enumerate(
                        zip(
                            row["anchor_candidate_identity_sha256s"],
                            row["anchor_source_content_sha256s"],
                            row["anchor_sources"],
                            strict=True,
                        )
                    ):
                        anchor = {
                            "pair_identity_sha256": pair,
                            "anchor_index": index,
                            "candidate_identity_sha256": anchor_identity,
                            "source_content_sha256": content,
                            "source_key_sha256": source_key(source),
                            "provisional_split": split,
                        }
                        database.execute(
                            "INSERT INTO anchors VALUES (?, ?, ?, ?, ?, ?)",
                            tuple(anchor.values()),
                        )
                        ordered_anchors.append(anchor)
                        counts["anchors"] += 1
                    seen_pairs.add(pair)
        if (
            counts["documents"] != descriptor.get("rows")
            or canonical_sha256(ordered_records)
            != descriptor.get("ordered_records_sha256")
            or counts["anchors"] != len(seen_pairs) * 2
        ):
            raise GroundedBridgeFoundationQueryError("bridge query coverage differs")
        counts["pairs"] = len(seen_pairs)
        counts["unique_word_signatures"] = database.execute(
            "SELECT COUNT(DISTINCT digest) FROM word_signatures"
        ).fetchone()[0]
        counts["unique_code_signatures"] = database.execute(
            "SELECT COUNT(DISTINCT digest) FROM code_signatures"
        ).fetchone()[0]
        metadata = {
            "schema": DATABASE_SCHEMA,
            "candidate_receipt_sha256": receipt["receipt_sha256"],
            "shingle_policy_sha256": canonical_sha256(SHINGLE_POLICY),
            "ordered_candidate_records_sha256": canonical_sha256(ordered_records),
            "ordered_anchors_sha256": canonical_sha256(ordered_anchors),
        }
        database.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                (key, json.dumps(value, sort_keys=True))
                for key, value in metadata.items()
            ),
        )
        database.commit()
        database.execute("PRAGMA wal_checkpoint(FULL)")
        database.close()
        database = None
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "source_candidate_receipt_sha256": receipt["receipt_sha256"],
            "shingle_policy": SHINGLE_POLICY,
            "shingle_policy_sha256": canonical_sha256(SHINGLE_POLICY),
            "counts": dict(sorted(counts.items())),
            "query_database": {
                "path": database_path.name,
                "bytes": database_path.stat().st_size,
                "sha256": sha256_file(database_path),
                "schema": DATABASE_SCHEMA,
                "source_text_persisted": False,
            },
            "ordered_candidate_records_sha256": canonical_sha256(ordered_records),
            "ordered_anchors_sha256": canonical_sha256(ordered_anchors),
            "source_text_persisted": False,
            "foundation_scan_complete": False,
            "global_deduplication_against_foundation_complete": False,
            "source_disjoint_against_foundation_complete": False,
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
        if database is not None:
            database.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path)
    args = parser.parse_args()
    result = build_query(args.candidate_root, args.output_root, args.durable_receipt)
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
