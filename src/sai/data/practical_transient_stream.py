"""Reconstruct practical PleIAs locators as a verified transient text stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_bounded_mechanical_candidates import REQUIRED_COLUMNS, _download
from sai.data.pleias_metadata_census import load_manifest
from sai.data.pleias_practical_admission import SCHEMA as ADMISSION_SCHEMA
from sai.data.pleias_practical_locator_scan import LOCATOR_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-practical-pretraining-document-v1"
RECEIPT_SCHEMA = "sai-practical-transient-stream-receipt-v1"


class PracticalTransientStreamError(RuntimeError):
    """An admission, locator, parent, or reconstructed text differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PracticalTransientStreamError("signed stream input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PracticalTransientStreamError("signed stream input is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise PracticalTransientStreamError("signed stream input differs")
    return payload


def _valid_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _locator_database(
    path: Path, database_path: Path
) -> tuple[sqlite3.Connection, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PracticalTransientStreamError("pyarrow is required") from error
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE locators (source_path TEXT NOT NULL, row_index INTEGER NOT NULL, "
        "row_json TEXT NOT NULL, PRIMARY KEY(source_path, row_index)) WITHOUT ROWID"
    )
    rows = 0
    try:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096, use_threads=False
        ):
            values = []
            for row in batch.to_pylist():
                if (
                    row.get("schema") != LOCATOR_SCHEMA
                    or row.get("source_id") != "pleias_common_corpus"
                    or not isinstance(row.get("source_path"), str)
                    or not row["source_path"]
                    or not isinstance(row.get("source_row_index"), int)
                    or row["source_row_index"] < 0
                    or not _valid_hex(row.get("source_parent_sha256"))
                    or not _valid_hex(row.get("source_row_identity_sha256"))
                    or not _valid_hex(row.get("content_sha256"))
                    or row.get("language", "").strip().casefold() != "english"
                ):
                    raise PracticalTransientStreamError("stream locator differs")
                values.append(
                    (
                        row["source_path"],
                        row["source_row_index"],
                        json.dumps(row, sort_keys=True, separators=(",", ":")),
                    )
                )
                rows += 1
            try:
                connection.executemany("INSERT INTO locators VALUES (?, ?, ?)", values)
            except sqlite3.IntegrityError as error:
                raise PracticalTransientStreamError(
                    "stream locator overlaps"
                ) from error
        connection.commit()
    except BaseException:
        connection.close()
        raise
    return connection, rows


def _document(locator: dict[str, Any], text: str) -> dict[str, Any]:
    content = text.encode()
    if (
        hashlib.sha256(content).hexdigest() != locator["content_sha256"]
        or len(content) != locator["text_utf8_bytes"]
    ):
        raise PracticalTransientStreamError("reconstructed text differs")
    payload = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": locator["source_repository"],
            "revision": locator["source_revision"],
            "path": locator["source_path"],
            "row_index": locator["source_row_index"],
            "identifier": locator["identifier"],
            "parent_sha256": locator["source_parent_sha256"],
            "content_sha256": locator["content_sha256"],
            "license": locator["license"],
        },
        "metadata": {
            "collection": locator["collection"],
            "open_type": locator["open_type"],
            "language": "English",
            "word_count": locator["word_count"],
            "source_token_count": locator["source_token_count"],
        },
        "quality_route": "pass_mechanical_gate",
        "practical_pretraining_ready": True,
        "training_ready": True,
    }
    return payload


def stream_shard(
    manifest_path: Path,
    admission_root: Path,
    shard_index: int,
    token: str,
    output: TextIO,
    receipt_path: Path,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct one source-local locator partition without persisting its text."""

    if (
        not token
        or shard_index < 0
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise PracticalTransientStreamError("stream arguments differ")
    admission = _load_signed(admission_root / "receipt.json", ADMISSION_SCHEMA)
    descriptors = [
        row
        for row in admission.get("outputs", {}).get("descriptors", [])
        if row.get("shard_index") == shard_index
    ]
    if (
        admission.get("status") != "complete_practical_pleias_pretraining_admission"
        or admission.get("practical_pretraining_ready") is not True
        or admission.get("training_ready") is not True
        or admission.get("policy", {}).get("output_partition_policy")
        != "canonical_source_path_sha256_modulo"
        or len(descriptors) != 1
    ):
        raise PracticalTransientStreamError("stream admission differs")
    descriptor = descriptors[0]
    locator_path = admission_root / descriptor.get("path", "")
    if (
        not locator_path.is_file()
        or locator_path.is_symlink()
        or locator_path.stat().st_nlink != 1
        or locator_path.stat().st_size != descriptor.get("bytes")
        or sha256_file(locator_path) != descriptor.get("sha256")
    ):
        raise PracticalTransientStreamError("stream locator payload differs")
    manifest = {row["source_path"]: row for row in load_manifest(manifest_path)}
    documents = text_bytes = source_tokens = parents = 0
    ordered = hashlib.sha256()
    with tempfile.TemporaryDirectory(
        prefix="sai-practical-stream-", dir=scratch_root
    ) as directory:
        connection, locator_rows = _locator_database(
            locator_path, Path(directory) / "locators.sqlite3"
        )
        try:
            source_paths = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT source_path FROM locators ORDER BY source_path"
                )
            ]
            for source_path in source_paths:
                parent = manifest.get(source_path)
                if parent is None:
                    raise PracticalTransientStreamError("stream parent is absent")
                parent_root = Path(directory) / f"parent-{parents:05d}"
                parent_root.mkdir()
                local = _download(parent, token, parent_root)
                try:
                    import pyarrow.parquet as pq
                except ImportError as error:
                    raise PracticalTransientStreamError(
                        "pyarrow is required"
                    ) from error
                parquet = pq.ParquetFile(local)
                if not REQUIRED_COLUMNS.issubset(parquet.schema_arrow.names):
                    raise PracticalTransientStreamError("stream parent columns differ")
                cursor = iter(
                    connection.execute(
                        "SELECT row_index, row_json FROM locators "
                        "WHERE source_path=? ORDER BY row_index",
                        (source_path,),
                    )
                )
                wanted = next(cursor, None)
                row_index = 0
                for batch in parquet.iter_batches(
                    batch_size=256,
                    columns=sorted(REQUIRED_COLUMNS),
                    use_threads=False,
                ):
                    for row in batch.to_pylist():
                        if wanted is not None and row_index == wanted[0]:
                            locator = json.loads(wanted[1])
                            if (
                                locator["source_repository"]
                                != parent["source_repository"]
                                or locator["source_revision"]
                                != parent["source_revision"]
                                or locator["source_parent_sha256"] != parent["sha256"]
                                or row.get("identifier") != locator["identifier"]
                                or row.get("collection") != locator["collection"]
                                or row.get("open_type") != locator["open_type"]
                                or row.get("license") != locator["license"]
                                or row.get("word_count") != locator["word_count"]
                                or row.get("token_count")
                                != locator["source_token_count"]
                                or row.get("language", "").strip().casefold()
                                != "english"
                                or not isinstance(row.get("text"), str)
                            ):
                                raise PracticalTransientStreamError(
                                    "reconstructed row metadata differs"
                                )
                            document = _document(locator, row["text"])
                            output.write(
                                json.dumps(
                                    document, sort_keys=True, separators=(",", ":")
                                )
                                + "\n"
                            )
                            documents += 1
                            text_bytes += locator["text_utf8_bytes"]
                            source_tokens += locator["source_token_count"]
                            ordered.update(bytes.fromhex(locator["content_sha256"]))
                            ordered.update(
                                bytes.fromhex(locator["source_row_identity_sha256"])
                            )
                            wanted = next(cursor, None)
                        row_index += 1
                if wanted is not None or row_index != parquet.metadata.num_rows:
                    raise PracticalTransientStreamError(
                        "reconstructed parent coverage differs"
                    )
                parents += 1
                shutil.rmtree(parent_root)
        finally:
            connection.close()
    if (
        documents != locator_rows
        or documents != descriptor["rows"]
        or text_bytes != descriptor["text_utf8_bytes"]
        or source_tokens != descriptor["source_token_count"]
    ):
        raise PracticalTransientStreamError("transient stream accounting differs")
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_practical_transient_training_stream",
        "admission_receipt_sha256": admission["receipt_sha256"],
        "shard_index": shard_index,
        "locator_descriptor": descriptor,
        "counts": {
            "source_parents": parents,
            "documents": documents,
            "text_utf8_bytes": text_bytes,
            "source_token_count": source_tokens,
        },
        "ordered_content_and_identity_sha256": ordered.hexdigest(),
        "source_text_persisted": False,
        "practical_pretraining_ready": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(receipt_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = stream_shard(
        args.manifest,
        args.admission_root,
        args.shard_index,
        os.environ.get(args.token_env, ""),
        sys.stdout,
        args.receipt,
        args.scratch_root,
    )
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
