"""Replay sealed virtual PleIAs locators into a transient tokenizer input stream.

The source corpus is intentionally not materialized a second time.  Every row is
downloaded from its pinned parent, replayed through the internal and cross-source
deletion decisions, checked against the final locator, and written only to the
caller-provided stream.  The durable output is a source-text-free receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from sai.data.agent_labeling import _atomic_create
from sai.data.cross_source_subdocument_rewrite import decision_database
from sai.data.pleias_bounded_mechanical_candidates import _download
from sai.data.pleias_cross_source_subdocument_rewrite import (
    COMPONENT_PRIORITY,
    rewrite_row,
)
from sai.data.pleias_final_subdocument_signature import COMPONENT
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_production_materializer import (
    _load_signed,
    _selection_database,
    _semantic_metadata,
    replay_selected_row,
)
from sai.data.pleias_subdocument_rewrite import _decision_database, rewrite_candidate
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA,
    AGGREGATE_STATUS,
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    SHARD_STATUS,
    _decision_rows,
    final_locator_row,
)
from sai.data.pleias_virtual_internal_rewrite_signature import (
    TRANSFORMED_LOCATOR_SCHEMA,
)
from sai.data.pleias_virtual_subdocument_signature import _selected_rows
from sai.data.token_stream import (
    ROW_SCHEMA,
    canonical_sha256,
    normalize_document,
    sha256_file,
)

ENVELOPE_SCHEMA = "sai-pleias-transient-tokenizer-envelope-v1"
RECEIPT_SCHEMA = "sai-pleias-transient-tokenizer-stream-receipt-v1"
STATUS = "complete_nontraining_transient_pleias_tokenizer_stream"


class PleiasVirtualTransientStreamError(RuntimeError):
    """A sealed locator, replayed row, stream identity, or coverage differs."""


def _locator_database(
    final_root: Path,
    logical_shards: int,
    shard_index: int,
    database_path: Path,
) -> tuple[sqlite3.Connection, dict[str, Any], dict[str, Any], int]:
    """Validate aggregate/shard custody and index source-text-free locators."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualTransientStreamError("pyarrow is required") from error
    aggregate = _load_signed(final_root / "aggregate.json", AGGREGATE_SCHEMA)
    if (
        aggregate.get("status") != AGGREGATE_STATUS
        or aggregate.get("complete_final_pleias_document_coverage") is not True
        or aggregate.get("benchmark_decontamination_complete") is not True
        or aggregate.get("cross_source_subdocument_deduplication_complete") is not True
        or aggregate.get("source_disjoint_split_complete") is not True
        or aggregate.get("source_text_persisted") is not False
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
    ):
        raise PleiasVirtualTransientStreamError("final aggregate differs")
    receipts = []
    receipt = None
    for index in range(logical_shards):
        candidate = _load_signed(
            final_root / "shards" / f"shard_{index:05d}" / "receipt.json",
            SHARD_SCHEMA,
        )
        receipts.append(candidate["receipt_sha256"])
        if index == shard_index:
            receipt = candidate
    if receipt is None or aggregate.get("shards", {}).get(
        "ordered_receipts_sha256"
    ) != canonical_sha256(receipts):
        raise PleiasVirtualTransientStreamError("aggregate shard custody differs")
    root = final_root / "shards" / f"shard_{shard_index:05d}"
    descriptor = receipt.get("final_locators")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        receipt.get("status") != SHARD_STATUS
        or receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("complete_final_pleias_document_coverage") is not True
        or receipt.get("benchmark_decontamination_complete") is not True
        or receipt.get("cross_source_subdocument_deduplication_complete") is not True
        or receipt.get("source_disjoint_split_complete") is not True
        or receipt.get("source_text_persisted") is not False
        or not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PleiasVirtualTransientStreamError("final shard differs")
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE locators ("
        "virtual_row_index INTEGER PRIMARY KEY, source_path TEXT NOT NULL, "
        "source_row_index INTEGER NOT NULL, locator_json TEXT NOT NULL, "
        "UNIQUE(source_path, source_row_index))"
    )
    rows = 0
    ordered = hashlib.sha256()
    try:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1024, use_threads=False
        ):
            for locator in batch.to_pylist():
                unsigned = {
                    key: value
                    for key, value in locator.items()
                    if key != "locator_sha256"
                }
                if (
                    locator.get("schema") != LOCATOR_SCHEMA
                    or locator.get("training_ready") is not False
                    or locator.get("virtual_row_index") != rows
                    or locator.get("locator_sha256") != canonical_sha256(unsigned)
                ):
                    raise PleiasVirtualTransientStreamError("final locator differs")
                connection.execute(
                    "INSERT INTO locators VALUES (?, ?, ?, ?)",
                    (
                        rows,
                        locator["source_path"],
                        locator["source_row_index"],
                        json.dumps(locator, sort_keys=True, separators=(",", ":")),
                    ),
                )
                ordered.update(bytes.fromhex(locator["locator_sha256"]))
                rows += 1
        connection.commit()
        if (
            rows != descriptor.get("rows")
            or rows != receipt.get("counts", {}).get("documents")
            or ordered.hexdigest() != descriptor.get("ordered_locator_digests_sha256")
        ):
            raise PleiasVirtualTransientStreamError("final locator coverage differs")
    except BaseException:
        connection.close()
        raise
    return connection, aggregate, receipt, rows


def _internal_locator(
    locator: dict[str, Any],
    candidate: dict[str, Any],
    rewritten: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the exact intermediate locator so final_locator_row can replay it."""

    row = {
        "schema": TRANSFORMED_LOCATOR_SCHEMA,
        "virtual_row_index": locator["virtual_row_index"],
        "source_repository": locator["source_repository"],
        "source_revision": locator["source_revision"],
        "source_path": locator["source_path"],
        "source_parent_sha256": locator["source_parent_sha256"],
        "source_row_index": locator["source_row_index"],
        "source_row_identity_sha256": locator["source_row_identity_sha256"],
        "pre_internal_content_sha256": candidate["content_sha256"],
        "content_sha256": rewritten["content_sha256"],
        "source_text_utf8_bytes": len(candidate["text"].encode()),
        "output_text_utf8_bytes": len(rewritten["text"].encode()),
        "source_text_characters": len(candidate["text"]),
        "output_text_characters": len(rewritten["text"]),
        "source_word_count": candidate["word_count"],
        "output_word_count": rewritten["word_count"],
        "source_token_count": candidate["token_count"],
        "token_count_requires_recomputation": True,
        "collection": candidate["collection"],
        "open_type": candidate["open_type"],
        "license": candidate["license"],
        "language": candidate["language"],
        "semantic_stratum": candidate["semantic_stratum"],
        "semantic_quality_floor_milli": candidate["semantic_quality_floor_milli"],
        "semantic_quality_mean_milli": candidate["semantic_quality_mean_milli"],
        "semantic_difficulty_mean_milli": candidate["semantic_difficulty_mean_milli"],
        "semantic_prerequisite_burden_mean_milli": candidate[
            "semantic_prerequisite_burden_mean_milli"
        ],
        "semantic_curriculum_phase": candidate["semantic_curriculum_phase"],
        "semantic_domains": candidate["semantic_domains"],
        "semantic_recurring_concepts": candidate["semantic_recurring_concepts"],
        "semantic_recurring_prerequisites": candidate[
            "semantic_recurring_prerequisites"
        ],
        "code_document": locator["code_document"],
        "internal_subdocument_transform_sha256": rewritten[
            "subdocument_transform_sha256"
        ],
        "training_ready": False,
    }
    row["locator_sha256"] = canonical_sha256(row)
    return row


def training_envelope(
    locator: dict[str, Any], text: str, shard_receipt_sha256: str
) -> dict[str, Any]:
    """Construct a standard pretraining row plus exact curriculum metadata."""

    if (
        locator.get("schema") != LOCATOR_SCHEMA
        or not isinstance(text, str)
        or not text
        or hashlib.sha256(text.encode()).hexdigest() != locator.get("content_sha256")
        or locator.get("corpus_split") not in {"train", "development"}
        or not isinstance(shard_receipt_sha256, str)
        or len(shard_receipt_sha256) != 64
    ):
        raise PleiasVirtualTransientStreamError("training envelope source differs")
    evidence = canonical_sha256(
        {
            "final_reconstruction_shard_receipt_sha256": shard_receipt_sha256,
            "final_locator_sha256": locator["locator_sha256"],
        }
    )
    semantic_key = " ".join(locator["semantic_domains"]).casefold()
    if locator["code_document"]:
        tokenizer_domain = "code"
    elif any(
        marker in semantic_key
        for marker in (
            "math",
            "algebra",
            "geometry",
            "statistics",
            "probability",
        )
    ):
        tokenizer_domain = "math"
    elif any(
        marker in semantic_key
        for marker in (
            "physics",
            "chemistry",
            "biology",
            "astronomy",
            "earth_science",
            "medicine",
            "health",
        )
    ):
        tokenizer_domain = "science"
    elif any(
        marker in semantic_key
        for marker in (
            "computer",
            "software",
            "technical",
            "technology",
            "engineering",
            "electronics",
            "robotics",
        )
    ):
        tokenizer_domain = "technical"
    else:
        tokenizer_domain = "english"
    document = normalize_document(
        {
            "schema": ROW_SCHEMA,
            "text": text,
            "source": {
                "dataset": (
                    f"{locator['source_repository']}@{locator['source_revision']}"
                ),
                "row_id": f"{locator['source_path']}#{locator['source_row_index']}",
                "license": locator["license"],
                "domain": tokenizer_domain,
            },
            "verification": {
                "benchmark_disjoint": True,
                "evidence_sha256": evidence,
            },
        }
    )
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "document": document,
        "corpus_split": locator["corpus_split"],
        "semantic_curriculum_phase": locator["semantic_curriculum_phase"],
        "semantic_difficulty_mean_milli": locator["semantic_difficulty_mean_milli"],
        "semantic_prerequisite_burden_mean_milli": locator[
            "semantic_prerequisite_burden_mean_milli"
        ],
        "semantic_quality_floor_milli": locator["semantic_quality_floor_milli"],
        "semantic_domains": locator["semantic_domains"],
        "semantic_recurring_concepts": locator["semantic_recurring_concepts"],
        "semantic_recurring_prerequisites": locator["semantic_recurring_prerequisites"],
        "final_locator_sha256": locator["locator_sha256"],
        "tokenization_ready": True,
        "training_ready": False,
    }
    envelope["envelope_sha256"] = canonical_sha256(envelope)
    return envelope


def iter_reconstructed_shard(
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    internal_decision_root: Path,
    cross_decision_root: Path,
    final_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield only rows whose exact two-layer replay matches the sealed locator."""

    if not token or logical_shards <= 0 or not 0 <= shard_index < logical_shards:
        raise PleiasVirtualTransientStreamError("transient stream arguments differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasVirtualTransientStreamError("pyarrow is required") from error
    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    _selection, selection_path = _selection_database(selection_root)
    semantic_by_stratum, _semantic_receipt = _semantic_metadata(semantic_decision_path)
    selection = sqlite3.connect(f"file:{selection_path.resolve()}?mode=ro", uri=True)
    connections: list[sqlite3.Connection] = [selection]
    emitted = 0
    seen_virtual: set[int] = set()
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-pleias-transient-state-", dir=scratch_root
        ) as state_directory:
            state = Path(state_directory)
            locators, _aggregate, shard_receipt, expected = _locator_database(
                final_root,
                logical_shards,
                shard_index,
                state / "final-locators.sqlite3",
            )
            connections.append(locators)
            internal, _internal_receipts, _internal_total = _decision_database(
                internal_decision_root,
                shard_index,
                logical_shards,
                state / "internal-deletions.sqlite3",
            )
            connections.append(internal)
            cross, _cross_receipts, _cross_total = decision_database(
                cross_decision_root,
                COMPONENT,
                COMPONENT_PRIORITY,
                shard_index,
                logical_shards,
                state / "cross-deletions.sqlite3",
            )
            connections.append(cross)
            for parent in parents:
                selected_rows = _selected_rows(selection, parent["source_path"])
                if not selected_rows:
                    continue
                by_index = {row[0]: row[1:] for row in selected_rows}
                seen_source_rows: set[int] = set()
                with tempfile.TemporaryDirectory(
                    prefix="sai-pleias-transient-parent-", dir=scratch_root
                ) as source_directory:
                    source = _download(parent, token, Path(source_directory))
                    parquet = pq.ParquetFile(source)
                    row_offset = 0
                    for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                        for relative, source_row in enumerate(batch.to_pylist()):
                            source_row_index = row_offset + relative
                            selected = by_index.get(source_row_index)
                            if selected is None:
                                continue
                            seen_source_rows.add(source_row_index)
                            match = locators.execute(
                                "SELECT virtual_row_index, locator_json FROM locators "
                                "WHERE source_path=? AND source_row_index=?",
                                (parent["source_path"], source_row_index),
                            ).fetchone()
                            if match is None:
                                continue
                            virtual_row_index, encoded_locator = match
                            locator = json.loads(encoded_locator)
                            candidate = replay_selected_row(
                                source_row,
                                parent,
                                source_row_index,
                                selected,
                                semantic_by_stratum.get(selected[3], {}),
                            )
                            internally_rewritten, _ = rewrite_candidate(
                                candidate,
                                virtual_row_index,
                                _decision_rows(internal, virtual_row_index),
                            )
                            final, _ = rewrite_row(
                                internally_rewritten,
                                virtual_row_index,
                                _decision_rows(cross, virtual_row_index),
                            )
                            intermediate = _internal_locator(
                                locator, candidate, internally_rewritten
                            )
                            replayed_locator = final_locator_row(intermediate, final)
                            if (
                                replayed_locator != locator
                                or virtual_row_index in seen_virtual
                            ):
                                raise PleiasVirtualTransientStreamError(
                                    "final row replay differs"
                                )
                            seen_virtual.add(virtual_row_index)
                            emitted += 1
                            yield training_envelope(
                                locator, final["text"], shard_receipt["receipt_sha256"]
                            )
                        row_offset += batch.num_rows
                if seen_source_rows != set(by_index):
                    raise PleiasVirtualTransientStreamError(
                        "selected source coverage differs"
                    )
            if emitted != expected or seen_virtual != set(range(expected)):
                raise PleiasVirtualTransientStreamError(
                    "transient final document coverage differs"
                )
    finally:
        for connection in reversed(connections):
            connection.close()


def stream_shard(
    output: TextIO,
    receipt_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write transient JSONL and atomically preserve only hash/accounting evidence."""

    if receipt_path.exists() or receipt_path.is_symlink():
        raise PleiasVirtualTransientStreamError("transient receipt already exists")
    digest = hashlib.sha256()
    envelope_digests = hashlib.sha256()
    counts: Counter[str] = Counter()
    for envelope in iter_reconstructed_shard(**kwargs):
        encoded = (
            json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        output.write(encoded)
        digest.update(encoded.encode())
        envelope_digests.update(bytes.fromhex(envelope["envelope_sha256"]))
        document = envelope["document"]
        counts["documents"] += 1
        counts["text_utf8_bytes"] += len(document["text"].encode())
        counts[f"split::{envelope['corpus_split']}::documents"] += 1
        counts[
            f"curriculum_phase::{envelope['semantic_curriculum_phase']}::documents"
        ] += 1
    if not counts["documents"]:
        raise PleiasVirtualTransientStreamError("transient stream is empty")
    output.flush()
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": STATUS,
        "logical_shards": kwargs["logical_shards"],
        "shard_index": kwargs["shard_index"],
        "counts": dict(sorted(counts.items())),
        "ordered_jsonl_sha256": digest.hexdigest(),
        "ordered_envelope_digests_sha256": envelope_digests.hexdigest(),
        "benchmark_decontamination_replayed": True,
        "internal_subdocument_deduplication_replayed": True,
        "cross_source_subdocument_deduplication_replayed": True,
        "source_text_persisted_by_compiler": False,
        "tokenization_ready": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(receipt_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--semantic-decision", type=Path, required=True)
    parser.add_argument("--internal-decision-root", type=Path, required=True)
    parser.add_argument("--cross-decision-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = stream_shard(
        sys.stdout,
        args.receipt,
        manifest_path=args.manifest,
        selection_root=args.selection_root,
        semantic_decision_path=args.semantic_decision,
        internal_decision_root=args.internal_decision_root,
        cross_decision_root=args.cross_decision_root,
        final_root=args.final_root,
        logical_shards=args.logical_shards,
        shard_index=args.shard_index,
        token=os.environ.get(args.token_env, ""),
        scratch_root=args.scratch_root,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
