"""Pack source-verified Sai documents into uint16 foundation-window parts."""

from __future__ import annotations

import argparse
import array
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_component_admission import SCHEMA as BRIDGE_SCHEMA
from sai.data.institutional_books_practical_admission import SCHEMA as BOOK_SCHEMA
from sai.data.one_b_curriculum_index import SHARD_SCHEMA as INDEX_SCHEMA
from sai.data.one_b_foundation_window_plan import SCHEMA as PLAN_SCHEMA
from sai.data.one_b_parent_partition import PARTITION_COUNT
from sai.data.token_stream import canonical_sha256, sha256_file, sha256_tree

SCHEMA = "sai-1b-foundation-pack-shard-v1"
SEQUENCE_LENGTH = 4_096
PART_SEQUENCES = 10_000
PART_TOKENS = SEQUENCE_LENGTH * PART_SEQUENCES
BANDS = ("foundation", "intermediate", "advanced", "expert")


class OneBFoundationPackError(RuntimeError):
    """A source parent, selected document, token, part, or receipt differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBFoundationPackError("signed pack input differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBFoundationPackError("signed pack input differs")
    return value


def _selected(row: dict[str, Any], plan: dict[str, Any]) -> bool:
    band = row["curriculum_band"]
    priority = row["curriculum_priority_sha256"]
    return (
        band in BANDS
        and row["split"] == "train"
        and int(priority[:16], 16) % 1_000_000 < plan["bands"][band]["selection_ppm"]
    )


class _PartWriter:
    def __init__(self, root: Path, band: str) -> None:
        self.root = root / band
        self.root.mkdir(parents=True)
        self.band = band
        self.part_index = 0
        self.handle: Any = None
        self.temporary: Path | None = None
        self.tokens_in_part = 0
        self.total_tokens = 0
        self.parts: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.temporary = self.root / f".part_{self.part_index:05d}.partial"
        self.handle = self.temporary.open("xb")

    def append(self, token_ids: list[int]) -> None:
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < 48_000
            for token in token_ids
        ):
            raise OneBFoundationPackError("production token ID differs")
        offset = 0
        while offset < len(token_ids):
            if self.handle is None:
                self._open()
            available = PART_TOKENS - self.tokens_in_part
            chunk = token_ids[offset : offset + available]
            values = array.array("H", chunk)
            if values.itemsize != 2 or sys.byteorder != "little":
                raise OneBFoundationPackError("uint16 storage differs")
            values.tofile(self.handle)
            self.tokens_in_part += len(chunk)
            self.total_tokens += len(chunk)
            offset += len(chunk)
            if self.tokens_in_part == PART_TOKENS:
                self._close(keep_tokens=PART_TOKENS)

    def _close(self, *, keep_tokens: int) -> None:
        if self.handle is None or self.temporary is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.truncate(keep_tokens * 2)
        self.handle.close()
        if keep_tokens:
            final = self.root / f"part_{self.part_index:05d}.bin"
            os.replace(self.temporary, final)
            self.parts.append(
                {
                    "path": str(final.relative_to(self.root.parent)),
                    "tokens": keep_tokens,
                    "sequences": keep_tokens // SEQUENCE_LENGTH,
                    "bytes": final.stat().st_size,
                    "sha256": sha256_file(final),
                }
            )
            self.part_index += 1
        else:
            self.temporary.unlink()
        self.handle = None
        self.temporary = None
        self.tokens_in_part = 0

    def finish(self) -> dict[str, Any]:
        retained = self.tokens_in_part // SEQUENCE_LENGTH * SEQUENCE_LENGTH
        dropped = self.tokens_in_part - retained
        self._close(keep_tokens=retained)
        return {
            "band": self.band,
            "parts": self.parts,
            "parts_sha256": canonical_sha256(self.parts),
            "retained_tokens": sum(row["tokens"] for row in self.parts),
            "retained_sequences": sum(row["sequences"] for row in self.parts),
            "tail_tokens_dropped": dropped,
        }


def _tokenizer(root: Path) -> tuple[Any, str]:
    if not root.is_dir() or root.is_symlink():
        raise OneBFoundationPackError("production tokenizer root differs")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise OneBFoundationPackError("transformers is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if (
        not getattr(tokenizer, "is_fast", False)
        or tokenizer.vocab_size != 48_000
        or not isinstance(tokenizer.eos_token_id, int)
    ):
        raise OneBFoundationPackError("production tokenizer differs")
    return tokenizer, sha256_tree(root)


def _manifest(path: Path) -> dict[str, dict[str, Any]]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_path = row.get("source_path")
            if not isinstance(source_path, str) or source_path in values:
                raise OneBFoundationPackError("source manifest differs")
            values[source_path] = row
    if not values:
        raise OneBFoundationPackError("source manifest is empty")
    return values


def _download(parent: dict[str, Any], token: str, root: Path) -> Path:
    if not token:
        raise OneBFoundationPackError("Hugging Face token is required")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise OneBFoundationPackError("huggingface_hub is required") from error
    try:
        path = Path(
            hf_hub_download(
                repo_id=parent["source_repository"],
                filename=parent["source_path"],
                repo_type="dataset",
                revision=parent["source_revision"],
                token=token,
                local_dir=root,
            )
        )
    except Exception as error:
        raise OneBFoundationPackError("source parent download failed") from error
    if path.stat().st_size != parent["bytes"] or sha256_file(path) != parent["sha256"]:
        raise OneBFoundationPackError("source parent bytes differ")
    return path


def _pack_text(
    row: dict[str, Any], text: str, tokenizer: Any, writers: dict[str, _PartWriter]
) -> int:
    if hashlib.sha256(text.encode()).hexdigest() != row["content_sha256"]:
        raise OneBFoundationPackError("selected source content differs")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens.append(tokenizer.eos_token_id)
    writers[row["curriculum_band"]].append(tokens)
    return len(tokens)


def _account(counts: Counter[str], band: str, token_count: int) -> None:
    counts["documents"] += 1
    counts["tokens_before_tail_trim"] += token_count
    counts[f"band::{band}::documents"] += 1
    counts[f"band::{band}::tokens"] += token_count


def pack_remote_bucket(
    component: str,
    partition_root: Path,
    bucket: int,
    manifest_path: Path,
    plan_path: Path,
    tokenizer_root: Path,
    output_root: Path,
    token: str,
    scratch_root: Path,
) -> dict[str, Any]:
    """Download each selected remote parent once and pack its admitted rows."""

    if (
        component not in {"pleias", "code"}
        or not 0 <= bucket < PARTITION_COUNT
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise OneBFoundationPackError("remote packing arguments differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBFoundationPackError("pyarrow is required") from error
    plan = _load_signed(plan_path, PLAN_SCHEMA)
    tokenizer, tokenizer_identity = _tokenizer(tokenizer_root)
    parents = _manifest(manifest_path)
    locator_files = sorted(
        partition_root.glob(f"shard_*/parent_bucket={bucket}/*.parquet")
    )
    rows_by_parent: dict[str, list[dict[str, Any]]] = {}
    for path in locator_files:
        for row in pq.read_table(path).to_pylist():
            if _selected(row, plan):
                rows_by_parent.setdefault(row["source_path"], []).append(row)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    writers = {band: _PartWriter(stage, band) for band in BANDS}
    counts: Counter[str] = Counter()
    identities = set()
    parent_receipts = []
    try:
        for parent_number, (source_path, locators) in enumerate(
            sorted(rows_by_parent.items()), start=1
        ):
            parent = parents.get(source_path)
            if parent is None:
                raise OneBFoundationPackError("selected parent is absent from manifest")
            wanted = {row["source_row_index"]: row for row in locators}
            if len(wanted) != len(locators):
                raise OneBFoundationPackError("selected parent row overlaps")
            seen = set()
            with tempfile.TemporaryDirectory(
                prefix=f"sai-foundation-{bucket:03d}-{parent_number:03d}-",
                dir=scratch_root,
            ) as temporary:
                local = _download(parent, token, Path(temporary))
                if component == "code":
                    with gzip.open(local, "rt", encoding="utf-8") as handle:
                        for row_index, line in enumerate(handle):
                            locator = wanted.get(row_index)
                            if locator is None:
                                continue
                            text = json.loads(line).get("text")
                            if not isinstance(text, str):
                                raise OneBFoundationPackError("code source row differs")
                            token_count = _pack_text(locator, text, tokenizer, writers)
                            seen.add(row_index)
                            identity = locator["document_identity_sha256"]
                            if identity in identities:
                                raise OneBFoundationPackError(
                                    "selected document identity overlaps"
                                )
                            identities.add(identity)
                            _account(counts, locator["curriculum_band"], token_count)
                            if len(seen) == len(wanted):
                                break
                else:
                    next_row = 0
                    for batch in pq.ParquetFile(local).iter_batches(
                        batch_size=128, columns=("text",), use_threads=False
                    ):
                        for raw in batch.to_pylist():
                            locator = wanted.get(next_row)
                            next_row += 1
                            if locator is None:
                                continue
                            text = raw.get("text")
                            if not isinstance(text, str):
                                raise OneBFoundationPackError(
                                    "PleIAs source row differs"
                                )
                            token_count = _pack_text(locator, text, tokenizer, writers)
                            seen.add(locator["source_row_index"])
                            identity = locator["document_identity_sha256"]
                            if identity in identities:
                                raise OneBFoundationPackError(
                                    "selected document identity overlaps"
                                )
                            identities.add(identity)
                            _account(counts, locator["curriculum_band"], token_count)
            if seen != set(wanted):
                raise OneBFoundationPackError("selected parent coverage differs")
            parent_receipts.append(
                {
                    "source_path": source_path,
                    "source_sha256": parent["sha256"],
                    "selected_rows": len(wanted),
                }
            )
        band_outputs = {band: writers[band].finish() for band in BANDS}
        retained_tokens = sum(row["retained_tokens"] for row in band_outputs.values())
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_1b_foundation_pack_shard",
            "component": component,
            "parent_bucket": bucket,
            "partition_count": PARTITION_COUNT,
            "plan_receipt_sha256": plan["receipt_sha256"],
            "tokenizer_identity_sha256": tokenizer_identity,
            "source_manifest_sha256": sha256_file(manifest_path),
            "locator_files": [str(path.resolve()) for path in locator_files],
            "locator_files_sha256": canonical_sha256(
                [sha256_file(path) for path in locator_files]
            ),
            "parents": parent_receipts,
            "parents_sha256": canonical_sha256(parent_receipts),
            "counts": dict(sorted(counts.items())),
            "unique_document_identities": len(identities),
            "band_outputs": band_outputs,
            "retained_tokens": retained_tokens,
            "retained_sequences": retained_tokens // SEQUENCE_LENGTH,
            "sequence_length": SEQUENCE_LENGTH,
            "part_sequences": PART_SEQUENCES,
            "uint16_little_endian": True,
            "eos_between_documents": True,
            "development_rows_excluded": True,
            "model_training_started": False,
            "one_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _finish_local(
    stage: Path,
    output_root: Path,
    component: str,
    source_shard: int,
    source_receipt_sha256: str,
    plan: dict[str, Any],
    tokenizer_identity: str,
    writers: dict[str, _PartWriter],
    counts: Counter[str],
    identities: set[str],
) -> dict[str, Any]:
    band_outputs = {band: writers[band].finish() for band in BANDS}
    retained_tokens = sum(row["retained_tokens"] for row in band_outputs.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_1b_foundation_pack_shard",
        "component": component,
        "source_shard": source_shard,
        "source_receipt_sha256": source_receipt_sha256,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "tokenizer_identity_sha256": tokenizer_identity,
        "counts": dict(sorted(counts.items())),
        "unique_document_identities": len(identities),
        "band_outputs": band_outputs,
        "retained_tokens": retained_tokens,
        "retained_sequences": retained_tokens // SEQUENCE_LENGTH,
        "sequence_length": SEQUENCE_LENGTH,
        "part_sequences": PART_SEQUENCES,
        "uint16_little_endian": True,
        "eos_between_documents": True,
        "development_rows_excluded": True,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(stage / "receipt.json", payload)
    os.replace(stage, output_root)
    return payload


def pack_book_shard(
    admission_root: Path,
    index_root: Path,
    source_shard: int,
    plan_path: Path,
    tokenizer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Pack one private Institutional Books shard into selected band parts."""

    if output_root.exists() or output_root.is_symlink() or not 0 <= source_shard < 64:
        raise OneBFoundationPackError("book packing arguments differ")
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBFoundationPackError("pyarrow is required") from error
    plan = _load_signed(plan_path, PLAN_SCHEMA)
    admission = _load_signed(admission_root / "receipt.json", BOOK_SCHEMA)
    index = _load_signed(index_root / "receipt.json", INDEX_SCHEMA)
    descriptor = index.get("output", {})
    index_path = index_root / descriptor.get("path", "")
    if (
        admission.get("training_ready") is not True
        or index.get("component") != "books"
        or index.get("source_receipt_sha256") != admission["receipt_sha256"]
        or sha256_file(index_path) != descriptor.get("sha256")
    ):
        raise OneBFoundationPackError("book packing source differs")
    table = pq.read_table(index_path)
    table = table.filter(pc.equal(table["source_shard"], source_shard))
    selected = {
        row["content_sha256"]: row for row in table.to_pylist() if _selected(row, plan)
    }
    relative = f"shards/shard_{source_shard:05d}/data.parquet"
    if {row["source_path"] for row in selected.values()} - {relative}:
        raise OneBFoundationPackError("book source path differs")
    source = Path(admission["source_text_location"]) / relative
    if not source.is_file() or source.is_symlink():
        raise OneBFoundationPackError("book source shard differs")
    tokenizer, tokenizer_identity = _tokenizer(tokenizer_root)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    writers = {band: _PartWriter(stage, band) for band in BANDS}
    counts: Counter[str] = Counter()
    identities = set()
    try:
        for batch in pq.ParquetFile(source).iter_batches(
            batch_size=32,
            columns=("text", "source_content_sha256"),
            use_threads=False,
        ):
            for raw in batch.to_pylist():
                row = selected.get(raw["source_content_sha256"])
                if row is None:
                    continue
                identity = row["document_identity_sha256"]
                if identity in identities:
                    raise OneBFoundationPackError("book identity overlaps")
                text = raw.get("text")
                if not isinstance(text, str):
                    raise OneBFoundationPackError("book source text differs")
                token_count = _pack_text(row, text, tokenizer, writers)
                identities.add(identity)
                _account(counts, row["curriculum_band"], token_count)
        if identities != {row["document_identity_sha256"] for row in selected.values()}:
            raise OneBFoundationPackError("book selected coverage differs")
        return _finish_local(
            stage,
            output_root,
            "books",
            source_shard,
            admission["receipt_sha256"],
            plan,
            tokenizer_identity,
            writers,
            counts,
            identities,
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def pack_connections(
    admission_root: Path,
    index_root: Path,
    plan_path: Path,
    tokenizer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Pack every admitted train-only connection exactly once."""

    if output_root.exists() or output_root.is_symlink():
        raise OneBFoundationPackError("connection packing arguments differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBFoundationPackError("pyarrow is required") from error
    plan = _load_signed(plan_path, PLAN_SCHEMA)
    admission = _load_signed(admission_root / "receipt.json", BRIDGE_SCHEMA)
    index = _load_signed(index_root / "receipt.json", INDEX_SCHEMA)
    descriptor = index.get("output", {})
    index_path = index_root / descriptor.get("path", "")
    source_descriptor = admission.get("train", {})
    source = admission_root / source_descriptor.get("path", "")
    if (
        admission.get("training_ready") is not True
        or index.get("component") != "connections"
        or index.get("source_receipt_sha256") != admission["receipt_sha256"]
        or sha256_file(index_path) != descriptor.get("sha256")
        or sha256_file(source) != source_descriptor.get("sha256")
    ):
        raise OneBFoundationPackError("connection packing source differs")
    indexed = {
        row["document_identity_sha256"]: row
        for row in pq.read_table(index_path).to_pylist()
    }
    tokenizer, tokenizer_identity = _tokenizer(tokenizer_root)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    writers = {band: _PartWriter(stage, band) for band in BANDS}
    counts: Counter[str] = Counter()
    identities = set()
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                identity = raw.get("document_identity_sha256")
                row = indexed.get(identity)
                text = raw.get("text")
                if row is None or not isinstance(text, str) or identity in identities:
                    raise OneBFoundationPackError("connection row differs")
                token_count = _pack_text(row, text, tokenizer, writers)
                identities.add(identity)
                _account(counts, row["curriculum_band"], token_count)
        if identities != set(indexed):
            raise OneBFoundationPackError("connection coverage differs")
        return _finish_local(
            stage,
            output_root,
            "connections",
            -1,
            admission["receipt_sha256"],
            plan,
            tokenizer_identity,
            writers,
            counts,
            identities,
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("pleias", "code", "books", "connections"))
    parser.add_argument("--partition-root", type=Path)
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--admission-root", type=Path)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--source-shard", type=int)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    if args.component in {"pleias", "code"}:
        if None in (
            args.partition_root,
            args.bucket,
            args.manifest,
            args.token_env,
            args.scratch_root,
        ):
            raise OneBFoundationPackError("remote packing arguments are incomplete")
        value = pack_remote_bucket(
            args.component,
            args.partition_root,
            args.bucket,
            args.manifest,
            args.plan,
            args.tokenizer_root,
            args.output_root,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    elif args.component == "books":
        if (
            args.admission_root is None
            or args.index_root is None
            or args.source_shard is None
        ):
            raise OneBFoundationPackError("book packing arguments are incomplete")
        value = pack_book_shard(
            args.admission_root,
            args.index_root,
            args.source_shard,
            args.plan,
            args.tokenizer_root,
            args.output_root,
        )
    else:
        if args.admission_root is None or args.index_root is None:
            raise OneBFoundationPackError("connection packing arguments are incomplete")
        value = pack_connections(
            args.admission_root,
            args.index_root,
            args.plan,
            args.tokenizer_root,
            args.output_root,
        )
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
