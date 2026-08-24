"""Scan final foundation shards against grounded-bridge queries without source text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_yield_ledger import DataYieldLedgerError, _bound_file, _load_receipt
from sai.data.decontamination import _CODE, _WORD, _code_shingles, _normalize, _shingles
from sai.data.decontamination import POLICY as SHINGLE_POLICY
from sai.data.foundation_source_split import POLICY_SHA256 as SPLIT_POLICY_SHA256
from sai.data.grounded_bridge_foundation_query import (
    DATABASE_SCHEMA as QUERY_DATABASE_SCHEMA,
)
from sai.data.grounded_bridge_foundation_query import SCHEMA as QUERY_SCHEMA
from sai.data.grounded_bridge_foundation_query import STATUS as QUERY_STATUS
from sai.data.grounded_bridge_foundation_query import source_key
from sai.data.institutional_books import ENRICHED_REPOSITORY, ENRICHED_REVISION
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    OUTPUT_SCHEMA as BOOK_ROW_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    SHARD_SCHEMA as BOOK_SHARD_SCHEMA,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_transient_tokenizer_stream import _selection
from sai.data.pleias_production_materializer import _load_signed
from sai.data.pleias_virtual_transient_stream import (
    _locator_database,
    iter_reconstructed_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-bridge-foundation-shard-scan-v1"
STATUS = "complete_nontraining_grounded_bridge_foundation_shard_scan"
ANCHOR_MATCH_SCHEMA = "sai-grounded-bridge-foundation-anchor-match-v1"


class GroundedBridgeFoundationScanError(RuntimeError):
    """Query custody, foundation coverage, or source-text-free evidence differs."""


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GroundedBridgeFoundationScanError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise GroundedBridgeFoundationScanError(f"{field} differs") from error
    return value


def source_key_aliases(source: dict[str, Any]) -> set[str]:
    """Return both separated and repository@revision source-coordinate hashes."""

    dataset = source.get("dataset") if isinstance(source, dict) else None
    revision = source.get("revision", "") if isinstance(source, dict) else None
    row_id = source.get("row_id") if isinstance(source, dict) else None
    if (
        not isinstance(dataset, str)
        or not dataset
        or not isinstance(revision, str)
        or not isinstance(row_id, str)
        or not row_id
    ):
        raise GroundedBridgeFoundationScanError("foundation source coordinates differ")
    aliases = {source_key({"dataset": dataset, "revision": revision, "row_id": row_id})}
    if revision:
        aliases.add(
            source_key(
                {"dataset": f"{dataset}@{revision}", "revision": "", "row_id": row_id}
            )
        )
    elif "@" in dataset:
        repository, observed_revision = dataset.rsplit("@", 1)
        if repository and observed_revision:
            aliases.add(
                source_key(
                    {
                        "dataset": repository,
                        "revision": observed_revision,
                        "row_id": row_id,
                    }
                )
            )
    return aliases


@dataclass(frozen=True)
class FoundationDocument:
    """The minimum transient fields required for exact global bridge checks."""

    component: str
    document_identity_sha256: str
    text: str
    corpus_split: str
    source_group_sha256: str
    source: dict[str, Any]
    source_content_sha256s: tuple[str, ...]
    source_custody_sha256: str

    def validate(self) -> FoundationDocument:
        if (
            not isinstance(self.component, str)
            or not self.component
            or not isinstance(self.text, str)
            or not self.text
            or self.corpus_split not in {"train", "development"}
        ):
            raise GroundedBridgeFoundationScanError("foundation document differs")
        _sha256(self.document_identity_sha256, "foundation document identity")
        _sha256(self.source_group_sha256, "foundation source group")
        _sha256(self.source_custody_sha256, "foundation source custody")
        if not self.source_content_sha256s:
            raise GroundedBridgeFoundationScanError(
                "foundation content custody differs"
            )
        for value in self.source_content_sha256s:
            _sha256(value, "foundation source content")
        source_key_aliases(self.source)
        return self


class QueryBoundary:
    """Validated in-memory view of the deliberately small bridge query boundary."""

    def __init__(self, root: Path) -> None:
        try:
            receipt = _load_receipt(root / "receipt.json")
        except (DataYieldLedgerError, OSError, json.JSONDecodeError) as error:
            raise GroundedBridgeFoundationScanError(
                "bridge query receipt differs"
            ) from error
        unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        descriptor = receipt.get("query_database")
        if (
            receipt.get("schema") != QUERY_SCHEMA
            or receipt.get("status") != QUERY_STATUS
            or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
            or receipt.get("shingle_policy") != SHINGLE_POLICY
            or receipt.get("shingle_policy_sha256") != canonical_sha256(SHINGLE_POLICY)
            or receipt.get("source_text_persisted") is not False
            or receipt.get("foundation_scan_complete") is not False
            or receipt.get("training_ready") is not False
            or not isinstance(descriptor, dict)
            or descriptor.get("schema") != QUERY_DATABASE_SCHEMA
            or descriptor.get("source_text_persisted") is not False
        ):
            raise GroundedBridgeFoundationScanError("bridge query receipt differs")
        try:
            path = _bound_file(root, descriptor)
        except DataYieldLedgerError as error:
            raise GroundedBridgeFoundationScanError(
                "bridge query database differs"
            ) from error
        database = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            metadata = {
                key: json.loads(value)
                for key, value in database.execute("SELECT key, value FROM metadata")
            }
            counts = receipt.get("counts", {})
            if (
                metadata.get("schema") != QUERY_DATABASE_SCHEMA
                or metadata.get("candidate_receipt_sha256")
                != receipt.get("source_candidate_receipt_sha256")
                or metadata.get("shingle_policy_sha256")
                != receipt.get("shingle_policy_sha256")
                or database.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                != counts.get("documents")
                or database.execute("SELECT COUNT(*) FROM anchors").fetchone()[0]
                != counts.get("anchors")
            ):
                raise GroundedBridgeFoundationScanError("bridge query database differs")
            self.word = frozenset(
                bytes(value)
                for (value,) in database.execute(
                    "SELECT DISTINCT digest FROM word_signatures"
                )
            )
            self.code = frozenset(
                bytes(value)
                for (value,) in database.execute(
                    "SELECT DISTINCT digest FROM code_signatures"
                )
            )
            if (
                len(self.word) != counts.get("unique_word_signatures")
                or len(self.code) != counts.get("unique_code_signatures")
                or any(len(value) != 32 for value in self.word | self.code)
            ):
                raise GroundedBridgeFoundationScanError(
                    "bridge query signatures differ"
                )
            self.anchor_by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
            self.anchor_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in database.execute(
                "SELECT pair_identity_sha256, anchor_index, "
                "candidate_identity_sha256, source_content_sha256, "
                "source_key_sha256, provisional_split FROM anchors "
                "ORDER BY pair_identity_sha256, anchor_index"
            ):
                anchor = {
                    "pair_identity_sha256": row[0],
                    "anchor_index": row[1],
                    "candidate_identity_sha256": row[2],
                    "source_content_sha256": row[3],
                    "source_key_sha256": row[4],
                    "provisional_split": row[5],
                }
                self.anchor_by_content[row[3]].append(anchor)
                self.anchor_by_source[row[4]].append(anchor)
        finally:
            database.close()
        self.receipt = receipt


def _write_digest_index(path: Path, values: set[bytes]) -> dict[str, Any]:
    with path.open("xb") as handle:
        for value in sorted(values):
            if len(value) != 32:
                raise GroundedBridgeFoundationScanError("matched digest differs")
            handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "rows": len(values),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_bytes": 32,
        "source_text_persisted": False,
    }


def scan_documents(
    query_root: Path,
    documents: Iterable[FoundationDocument],
    output_root: Path,
    *,
    component: str,
    logical_shards: int,
    shard_index: int,
    source_custody: dict[str, Any],
) -> dict[str, Any]:
    """Scan one complete foundation shard and preserve only matching digests."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or logical_shards <= 0
        or not 0 <= shard_index < logical_shards
        or not isinstance(source_custody, dict)
        or not source_custody
    ):
        raise GroundedBridgeFoundationScanError("foundation scan arguments differ")
    boundary = QueryBoundary(query_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir()
    matched_word: set[bytes] = set()
    matched_code: set[bytes] = set()
    anchor_records: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    ordered_documents = hashlib.sha256()
    ordered_custody = hashlib.sha256()
    try:
        for item in documents:
            document = item.validate()
            if document.component != component:
                raise GroundedBridgeFoundationScanError("foundation component differs")
            normalized = _normalize(document.text)
            words = _shingles(
                _WORD.findall(normalized), SHINGLE_POLICY["word_shingle_tokens"]
            )
            code = _code_shingles(_CODE.findall(normalized))
            word_matches = words & boundary.word
            code_matches = code & boundary.code
            matched_word.update(word_matches)
            matched_code.update(code_matches)
            counts["documents"] += 1
            counts["text_utf8_bytes"] += len(document.text.encode())
            counts[f"split::{document.corpus_split}::documents"] += 1
            counts["word_signature_occurrences"] += len(word_matches)
            counts["code_signature_occurrences"] += len(code_matches)
            counts["documents_with_word_overlap"] += bool(word_matches)
            counts["documents_with_code_overlap"] += bool(code_matches)
            ordered_documents.update(bytes.fromhex(document.document_identity_sha256))
            ordered_custody.update(bytes.fromhex(document.source_custody_sha256))
            candidates: dict[tuple[str, int], tuple[dict[str, Any], set[str]]] = {}
            for content in document.source_content_sha256s:
                for anchor in boundary.anchor_by_content.get(content, []):
                    candidates.setdefault(
                        (anchor["pair_identity_sha256"], anchor["anchor_index"]),
                        (anchor, set()),
                    )[1].add("source_content_sha256")
            for alias in source_key_aliases(document.source):
                for anchor in boundary.anchor_by_source.get(alias, []):
                    candidates.setdefault(
                        (anchor["pair_identity_sha256"], anchor["anchor_index"]),
                        (anchor, set()),
                    )[1].add("source_key_sha256")
            for anchor, match_types in candidates.values():
                row = {
                    "schema": ANCHOR_MATCH_SCHEMA,
                    "component": component,
                    "logical_shards": logical_shards,
                    "shard_index": shard_index,
                    "pair_identity_sha256": anchor["pair_identity_sha256"],
                    "anchor_index": anchor["anchor_index"],
                    "anchor_candidate_identity_sha256": anchor[
                        "candidate_identity_sha256"
                    ],
                    "provisional_bridge_split": anchor["provisional_split"],
                    "foundation_document_identity_sha256": (
                        document.document_identity_sha256
                    ),
                    "foundation_source_group_sha256": document.source_group_sha256,
                    "foundation_split": document.corpus_split,
                    "match_types": sorted(match_types),
                    "source_text_persisted": False,
                    "training_ready": False,
                }
                row["record_sha256"] = canonical_sha256(row)
                anchor_records[row["record_sha256"]] = row
        if not counts["documents"]:
            raise GroundedBridgeFoundationScanError("foundation shard scan is empty")
        word_descriptor = _write_digest_index(
            stage / "matched_word_digests.bin", matched_word
        )
        code_descriptor = _write_digest_index(
            stage / "matched_code_digests.bin", matched_code
        )
        anchor_path = stage / "anchor_matches.jsonl"
        ordered_anchor_records = []
        with anchor_path.open("x") as handle:
            for key in sorted(anchor_records):
                row = anchor_records[key]
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
                ordered_anchor_records.append(row["record_sha256"])
            handle.flush()
            os.fsync(handle.fileno())
        anchor_descriptor = {
            "path": anchor_path.name,
            "rows": len(anchor_records),
            "bytes": anchor_path.stat().st_size,
            "sha256": sha256_file(anchor_path),
            "ordered_records_sha256": canonical_sha256(ordered_anchor_records),
            "source_text_persisted": False,
        }
        counts["unique_matched_word_signatures"] = len(matched_word)
        counts["unique_matched_code_signatures"] = len(matched_code)
        counts["anchor_match_records"] = len(anchor_records)
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "component": component,
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "source_query_receipt_sha256": boundary.receipt["receipt_sha256"],
            "source_custody": source_custody,
            "shingle_policy": SHINGLE_POLICY,
            "shingle_policy_sha256": canonical_sha256(SHINGLE_POLICY),
            "counts": dict(sorted(counts.items())),
            "ordered_foundation_document_identities_sha256": (
                ordered_documents.hexdigest()
            ),
            "ordered_foundation_source_custody_sha256": ordered_custody.hexdigest(),
            "matched_word_digests": word_descriptor,
            "matched_code_digests": code_descriptor,
            "anchor_matches": anchor_descriptor,
            "source_text_persisted": False,
            "foundation_shard_scan_complete": True,
            "global_foundation_scan_complete": False,
            "global_deduplication_against_foundation_complete": False,
            "source_disjoint_against_foundation_complete": False,
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


def iter_pleias_documents(
    *,
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    internal_decision_root: Path,
    cross_decision_root: Path,
    final_root: Path,
    balance_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None,
) -> tuple[Iterator[FoundationDocument], dict[str, Any]]:
    """Replay final PleIAs text while binding every row to its sealed locator."""

    temporary = tempfile.TemporaryDirectory(
        prefix="sai-bridge-pleias-scan-", dir=scratch_root
    )
    database, aggregate, shard, balance_shard, expected = _locator_database(
        final_root,
        balance_root,
        logical_shards,
        shard_index,
        Path(temporary.name) / "locators.sqlite3",
    )

    def iterator() -> Iterator[FoundationDocument]:
        observed = 0
        try:
            envelopes = iter_reconstructed_shard(
                manifest_path=manifest_path,
                selection_root=selection_root,
                semantic_decision_path=semantic_decision_path,
                internal_decision_root=internal_decision_root,
                cross_decision_root=cross_decision_root,
                final_root=final_root,
                balance_root=balance_root,
                logical_shards=logical_shards,
                shard_index=shard_index,
                token=token,
                scratch_root=scratch_root,
            )
            for envelope in envelopes:
                observed += 1
                value = database.execute(
                    "SELECT locator_json FROM locators WHERE locator_sha256=?",
                    (envelope.get("final_locator_sha256"),),
                ).fetchone()
                if value is None:
                    raise GroundedBridgeFoundationScanError(
                        "PleIAs locator coverage differs"
                    )
                locator = json.loads(value[0])
                document = envelope.get("document")
                if (
                    not isinstance(document, dict)
                    or envelope.get("final_locator_sha256")
                    != locator.get("locator_sha256")
                    or envelope.get("corpus_split") != locator.get("corpus_split")
                    or locator.get("source_split_policy_sha256") != SPLIT_POLICY_SHA256
                    or document.get("identity_sha256") is None
                    or hashlib.sha256(document.get("text", "").encode()).hexdigest()
                    != locator.get("content_sha256")
                ):
                    raise GroundedBridgeFoundationScanError("PleIAs envelope differs")
                yield FoundationDocument(
                    component="pleias_common_corpus",
                    document_identity_sha256=document["identity_sha256"],
                    text=document["text"],
                    corpus_split=locator["corpus_split"],
                    source_group_sha256=locator["source_group_sha256"],
                    source={
                        "dataset": locator["source_repository"],
                        "revision": locator["source_revision"],
                        "row_id": (
                            f"{locator['source_path']}#{locator['source_row_index']}"
                        ),
                    },
                    source_content_sha256s=tuple(
                        dict.fromkeys(
                            [
                                locator["pre_internal_content_sha256"],
                                locator["post_internal_content_sha256"],
                                locator["content_sha256"],
                            ]
                        )
                    ),
                    source_custody_sha256=locator["locator_sha256"],
                )
            if observed != expected:
                raise GroundedBridgeFoundationScanError(
                    "PleIAs document coverage differs"
                )
        finally:
            database.close()
            temporary.cleanup()

    custody = {
        "final_aggregate_receipt_sha256": aggregate["receipt_sha256"],
        "final_shard_receipt_sha256": shard["receipt_sha256"],
        "byte_balance_shard_receipt_sha256": balance_shard["receipt_sha256"],
    }
    return iterator(), custody


def iter_book_documents(
    *, final_root: Path, selection_root: Path, logical_shards: int, shard_index: int
) -> tuple[Iterator[FoundationDocument], dict[str, Any]]:
    """Read and validate both final Institutional Books partitions."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise GroundedBridgeFoundationScanError("pyarrow is required") from error
    aggregate = _load_signed(final_root / "aggregate.json", BOOK_AGGREGATE_SCHEMA)
    if (
        aggregate.get("status")
        != "complete_nontraining_institutional_books_cross_source_rewritten"
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("complete_benchmark_disjoint_book_coverage") is not True
        or aggregate.get("cross_source_subdocument_deduplication_complete") is not True
    ):
        raise GroundedBridgeFoundationScanError("final book aggregate differs")
    receipts = []
    shard = None
    for index in range(logical_shards):
        value = _load_signed(
            final_root / "shards" / f"shard_{index:05d}" / "receipt.json",
            BOOK_SHARD_SCHEMA,
        )
        receipts.append(value["receipt_sha256"])
        if index == shard_index:
            shard = value
    if (
        shard is None
        or aggregate.get("shards", {}).get("ordered_receipts_sha256")
        != canonical_sha256(receipts)
        or shard.get("source_disjoint_split_complete") is not True
        or shard.get("source_disjoint_split_policy_sha256") != SPLIT_POLICY_SHA256
    ):
        raise GroundedBridgeFoundationScanError("final book shard custody differs")
    selected, selection_receipt = _selection(selection_root)

    def iterator() -> Iterator[FoundationDocument]:
        observed = 0
        for split in ("train", "development"):
            descriptor = shard.get("outputs", {}).get(split)
            expected = shard.get("counts", {}).get(f"split::{split}::documents", 0)
            if descriptor is None:
                if expected:
                    raise GroundedBridgeFoundationScanError("book split output differs")
                continue
            root = final_root / "shards" / f"shard_{shard_index:05d}"
            path = root / descriptor.get("path", "")
            if (
                descriptor.get("path") != f"{split}.parquet"
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
                or descriptor.get("rows") != expected
            ):
                raise GroundedBridgeFoundationScanError("book split descriptor differs")
            split_rows = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=16, use_threads=False
            ):
                for row in batch.to_pylist():
                    text = row.get("text")
                    barcode = row.get("barcode_src")
                    selection = selected.get(barcode)
                    if (
                        row.get("schema") != BOOK_ROW_SCHEMA
                        or row.get("training_ready") is not False
                        or row.get("corpus_split") != split
                        or row.get("source_split_policy_sha256") != SPLIT_POLICY_SHA256
                        or not isinstance(text, str)
                        or not text
                        or hashlib.sha256(text.encode()).hexdigest()
                        != row.get("content_sha256")
                        or selection is None
                        or row.get("selection_row_sha256")
                        != selection.get("row_sha256")
                    ):
                        raise GroundedBridgeFoundationScanError(
                            "final book row differs"
                        )
                    identity = canonical_sha256(
                        {
                            "component": "institutional_books",
                            "barcode_src": barcode,
                            "content_sha256": row["content_sha256"],
                            "source_group_sha256": row["source_group_sha256"],
                            "cross_source_subdocument_transform_sha256": row[
                                "cross_source_subdocument_transform_sha256"
                            ],
                        }
                    )
                    yield FoundationDocument(
                        component="institutional_books",
                        document_identity_sha256=identity,
                        text=text,
                        corpus_split=split,
                        source_group_sha256=row["source_group_sha256"],
                        source={
                            "dataset": ENRICHED_REPOSITORY,
                            "revision": ENRICHED_REVISION,
                            "row_id": barcode,
                        },
                        source_content_sha256s=tuple(
                            dict.fromkeys(
                                [row["source_content_sha256"], row["content_sha256"]]
                            )
                        ),
                        source_custody_sha256=row[
                            "cross_source_subdocument_transform_sha256"
                        ],
                    )
                    observed += 1
                    split_rows += 1
            if split_rows != expected:
                raise GroundedBridgeFoundationScanError("book split coverage differs")
        if observed != shard.get("counts", {}).get("documents"):
            raise GroundedBridgeFoundationScanError("book document coverage differs")

    custody = {
        "final_aggregate_receipt_sha256": aggregate["receipt_sha256"],
        "final_shard_receipt_sha256": shard["receipt_sha256"],
        "selection_receipt_sha256": selection_receipt["receipt_sha256"],
    }
    return iterator(), custody


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("pleias_common_corpus", "institutional_books"),
        required=True,
    )
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--semantic-decision", type=Path)
    parser.add_argument("--balance-root", type=Path)
    parser.add_argument("--internal-decision-root", type=Path)
    parser.add_argument("--cross-decision-root", type=Path)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    if args.component == "institutional_books":
        documents, custody = iter_book_documents(
            final_root=args.final_root,
            selection_root=args.selection_root,
            logical_shards=args.logical_shards,
            shard_index=args.shard_index,
        )
    else:
        if any(
            value is None
            for value in (
                args.manifest,
                args.semantic_decision,
                args.internal_decision_root,
                args.cross_decision_root,
                args.balance_root,
            )
        ):
            raise GroundedBridgeFoundationScanError("PleIAs scan inputs differ")
        documents, custody = iter_pleias_documents(
            manifest_path=args.manifest,
            selection_root=args.selection_root,
            semantic_decision_path=args.semantic_decision,
            internal_decision_root=args.internal_decision_root,
            cross_decision_root=args.cross_decision_root,
            final_root=args.final_root,
            balance_root=args.balance_root,
            logical_shards=args.logical_shards,
            shard_index=args.shard_index,
            token=os.environ.get(args.token_env, ""),
            scratch_root=args.scratch_root,
        )
    result = scan_documents(
        args.query_root,
        documents,
        args.output_root,
        component=args.component,
        logical_shards=args.logical_shards,
        shard_index=args.shard_index,
        source_custody=custody,
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
