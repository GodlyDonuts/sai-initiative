"""Replay cross-source deletion maps and coherently rewrite exact text spans."""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from pathlib import Path

from sai.data.cross_source_subdocument_decision import (
    DELETE_RECORD,
)
from sai.data.cross_source_subdocument_decision import (
    SCHEMA as DECISION_SCHEMA,
)
from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.frequency_length_subdocument_deduplication import (
    DEFAULT_DELETE_CHARACTERS,
    DEFAULT_SEGMENT_CHARACTERS,
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_subdocument_decision import _records
from sai.data.pleias_subdocument_signature import HASH_BUCKETS
from sai.data.token_stream import canonical_sha256, sha256_file


class CrossSourceSubdocumentRewriteError(RuntimeError):
    """Decision custody, source identity, chunk replay, or rewrite differs."""


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def rewrite_text(
    *,
    text: str,
    content_sha256: str,
    identity: str,
    source_row_index: int,
    decisions: list[tuple[str, int, int, int, str, int, int]],
    code_document: bool,
    delete_characters: int = DEFAULT_DELETE_CHARACTERS,
) -> tuple[str, dict[str, int], str]:
    """Delete exact decided chunks while preserving short or empty documents."""

    if (
        not isinstance(text, str)
        or not text
        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
        or not _valid_hash(content_sha256)
        or not _valid_hash(identity)
        or isinstance(source_row_index, bool)
        or not isinstance(source_row_index, int)
        or source_row_index < 0
        or not isinstance(code_document, bool)
        or isinstance(delete_characters, bool)
        or not isinstance(delete_characters, int)
        or delete_characters <= 0
    ):
        raise CrossSourceSubdocumentRewriteError("rewrite source differs")
    chunks = segment_subdocuments(
        text,
        minimum_characters=DEFAULT_SEGMENT_CHARACTERS,
        code_document=code_document,
    )
    selected = {}
    decision_digests = []
    for (
        expected_identity,
        chunk_index,
        character_start,
        character_end,
        normalized_sha256,
        frequency,
        budget,
    ) in decisions:
        if (
            expected_identity != identity
            or chunk_index in selected
            or not 0 <= chunk_index < len(chunks)
            or not _valid_hash(expected_identity)
            or not _valid_hash(normalized_sha256)
            or isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or isinstance(character_start, bool)
            or not isinstance(character_start, int)
            or isinstance(character_end, bool)
            or not isinstance(character_end, int)
            or isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or frequency <= budget
            or budget <= 0
        ):
            raise CrossSourceSubdocumentRewriteError("rewrite decision differs")
        chunk = chunks[chunk_index]
        normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
        decision = {
            "source_row_index": source_row_index,
            "document_identity_sha256": identity,
            "chunk_index": chunk_index,
            "character_start": character_start,
            "character_end": character_end,
            "normalized_sha256": normalized_sha256,
            "frequency": frequency,
            "budget": budget,
        }
        if (
            chunk["character_start"] != character_start
            or chunk["character_end"] != character_end
            or hashlib.sha256(normalized.encode()).hexdigest() != normalized_sha256
        ):
            raise CrossSourceSubdocumentRewriteError("rewrite chunk replay differs")
        selected[chunk_index] = chunk
        decision_digests.append(canonical_sha256(decision))
    delete_total = sum(len(chunk["text"]) for chunk in selected.values())
    counts = Counter(
        {
            "candidate_deletion_chunks": len(selected),
            "candidate_deletion_characters": delete_total,
            "deleted_chunks": 0,
            "deleted_characters": 0,
            "coherence_restored_chunks": 0,
        }
    )
    rewritten = text
    if selected and delete_total >= delete_characters:
        spans = sorted(
            (chunk["character_start"], chunk["character_end"])
            for chunk in selected.values()
        )
        if any(
            first[1] > second[0]
            for first, second in zip(spans, spans[1:], strict=False)
        ):
            raise CrossSourceSubdocumentRewriteError("rewrite spans overlap")
        parts = []
        cursor = 0
        for start, end in spans:
            parts.append(text[cursor:start])
            cursor = end
        parts.append(text[cursor:])
        candidate = "".join(parts)
        if candidate.strip():
            rewritten = candidate
            counts["deleted_chunks"] = len(selected)
            counts["deleted_characters"] = delete_total
        else:
            counts["coherence_restored_chunks"] = len(selected)
    elif selected:
        counts["coherence_restored_chunks"] = len(selected)
    transform = {
        "document_identity_sha256": identity,
        "source_content_sha256": content_sha256,
        "ordered_decision_digests_sha256": canonical_sha256(decision_digests),
        "output_content_sha256": hashlib.sha256(rewritten.encode()).hexdigest(),
    }
    return rewritten, dict(counts), canonical_sha256(transform)


def decision_database(
    decision_root: Path,
    component: str,
    component_priority: int,
    source_shard: int,
    logical_shards: int,
    database_path: Path,
) -> tuple[sqlite3.Connection, list[str], int]:
    """Build one shard-local SQLite index from all sixteen verified buckets."""

    aggregate = _load_signed(decision_root / "aggregate.json", AGGREGATE_SCHEMA)
    bindings = aggregate.get("components")
    if (
        aggregate.get("cross_source_subdocument_decision_complete") is not True
        or aggregate.get("decision_contains_source_text") is not False
        or aggregate.get("hash_partition", {}).get("complete") is not True
        or aggregate.get("hash_partition", {}).get("required_buckets")
        != HASH_BUCKETS
        or not isinstance(bindings, list)
        or not any(
            row.get("component") == component
            and row.get("priority") == component_priority
            and row.get("logical_shards") == logical_shards
            for row in bindings
        )
        or not 0 <= source_shard < logical_shards
    ):
        raise CrossSourceSubdocumentRewriteError("decision aggregate differs")
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE deletions ("
        "source_row_index INTEGER NOT NULL, document_identity_sha256 TEXT NOT NULL, "
        "chunk_index INTEGER NOT NULL, character_start INTEGER NOT NULL, "
        "character_end INTEGER NOT NULL, normalized_sha256 TEXT NOT NULL, "
        "frequency INTEGER NOT NULL, budget INTEGER NOT NULL, "
        "PRIMARY KEY(source_row_index, chunk_index)"
        ") WITHOUT ROWID"
    )
    receipts = []
    total = 0
    try:
        for bucket_index in range(HASH_BUCKETS):
            root = decision_root / "buckets" / f"bucket_{bucket_index:02x}"
            receipt = _load_signed(root / "receipt.json", DECISION_SCHEMA)
            descriptors = receipt.get("deletions")
            matches = [
                row
                for row in descriptors
                if isinstance(row, dict)
                and row.get("component") == component
                and row.get("component_priority") == component_priority
                and row.get("source_shard") == source_shard
            ] if isinstance(descriptors, list) else []
            descriptor = matches[0] if len(matches) == 1 else {}
            deletion_root = root / "deletions"
            relative = descriptor.get("path")
            path = deletion_root / relative if isinstance(relative, str) else root
            try:
                path.resolve().relative_to(deletion_root.resolve())
            except ValueError as error:
                raise CrossSourceSubdocumentRewriteError(
                    "deletion path escapes its root"
                ) from error
            if (
                receipt.get("hash_bucket", {}).get("index") != bucket_index
                or receipt.get("hash_bucket", {}).get("buckets") != HASH_BUCKETS
                or receipt.get("cross_source_subdocument_decision_complete") is not True
                or receipt.get("decision_contains_source_text") is not False
                or len(matches) != 1
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
                or descriptor.get("bytes")
                != descriptor.get("rows") * DELETE_RECORD.size
            ):
                raise CrossSourceSubdocumentRewriteError(
                    "deletion descriptor differs"
                )
            rows = 0
            with path.open("rb") as handle:
                for record in _records(handle, DELETE_RECORD):
                    if record[0] != component_priority or record[1] != source_shard:
                        raise CrossSourceSubdocumentRewriteError(
                            "deletion record partition differs"
                        )
                    connection.execute(
                        "INSERT INTO deletions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            record[2],
                            record[3].hex(),
                            record[4],
                            record[5],
                            record[6],
                            record[7].hex(),
                            record[8],
                            record[9],
                        ),
                    )
                    rows += 1
                    total += 1
            if rows != descriptor.get("rows"):
                raise CrossSourceSubdocumentRewriteError(
                    "deletion row coverage differs"
                )
            connection.commit()
            receipts.append(receipt["receipt_sha256"])
        connection.execute("CREATE INDEX deletions_row ON deletions(source_row_index)")
        connection.commit()
    except BaseException:
        connection.close()
        raise
    return connection, receipts, total
