"""Build bounded tokenizer-training samples from the released Sai 1B corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_component_admission import SCHEMA as BRIDGE_SCHEMA
from sai.data.common_pile_stack_edu_practical_admission import (
    SCHEMA as CODE_SCHEMA,
)
from sai.data.institutional_books_practical_admission import SCHEMA as BOOK_SCHEMA
from sai.data.one_b_curriculum_index import SHARD_SCHEMA as INDEX_SHARD_SCHEMA
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_SCHEMA
from sai.data.token_stream import (
    TOKENIZER_ROW_SCHEMA,
    canonical_sha256,
    normalize_tokenizer_document,
    sha256_file,
)

SCHEMA = "sai-1b-production-tokenizer-sample-v1"
STATUS = "complete_nontraining_1b_tokenizer_sample"
SAMPLE_NAME = "sample.jsonl"


class OneBTokenizerSampleError(RuntimeError):
    """A released source, curriculum edge, sample, or identity differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise OneBTokenizerSampleError("signed tokenizer-sample input is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBTokenizerSampleError(
            "signed tokenizer-sample input differs"
        ) from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("schema") != schema or value.get("receipt_sha256") != canonical_sha256(
        unsigned
    ):
        raise OneBTokenizerSampleError("signed tokenizer-sample input differs")
    return value


def _rank(identity: str) -> int:
    return int(
        hashlib.sha256(b"sai-1b-tokenizer-v1\0" + bytes.fromhex(identity)).hexdigest(),
        16,
    )


def _tokenizer_domain(value: str) -> str:
    folded = value.casefold()
    if any(marker in folded for marker in ("math", "algebra", "geometry", "statistic")):
        return "math"
    if any(
        marker in folded
        for marker in ("science", "physics", "chemistry", "biology", "medicine")
    ):
        return "science"
    if any(
        marker in folded
        for marker in ("technology", "engineering", "computer", "system", "technical")
    ):
        return "technical"
    return "english"


def _bounded_by_stratum(
    rows: Iterable[tuple[str, str, dict[str, Any]]], maximum_jsonl_bytes: int
) -> tuple[list[bytes], Counter[str]]:
    if maximum_jsonl_bytes < 1_000_000:
        raise OneBTokenizerSampleError("tokenizer sample byte cap differs")
    heaps: dict[str, list[tuple[int, str, int, bytes]]] = {}
    heap_bytes: Counter[str] = Counter()
    input_counts: Counter[str] = Counter()

    def rebalance() -> None:
        cap = maximum_jsonl_bytes // len(heaps)
        for stratum, heap in heaps.items():
            while heap and heap_bytes[stratum] > cap:
                _negative_rank, _identity, size, _encoded = heapq.heappop(heap)
                heap_bytes[stratum] -= size

    for stratum, identity, document in rows:
        normalized = normalize_tokenizer_document(document)
        if len(identity) != 64:
            raise OneBTokenizerSampleError("sample selection identity differs")
        try:
            bytes.fromhex(identity)
        except ValueError as error:
            raise OneBTokenizerSampleError(
                "sample selection identity differs"
            ) from error
        encoded = (
            json.dumps(
                normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode()
        input_counts["documents"] += 1
        input_counts["text_utf8_bytes"] += len(normalized["text"].encode())
        input_counts[f"stratum::{stratum}::documents"] += 1
        if stratum not in heaps:
            heaps[stratum] = []
            rebalance()
        cap = maximum_jsonl_bytes // len(heaps)
        if len(encoded) > cap:
            input_counts["oversize_documents_excluded"] += 1
            continue
        rank = _rank(identity)
        heapq.heappush(heaps[stratum], (-rank, identity, len(encoded), encoded))
        heap_bytes[stratum] += len(encoded)
        rebalance()
    selected = [
        (stratum, -negative_rank, identity, encoded)
        for stratum, heap in heaps.items()
        for negative_rank, identity, _size, encoded in heap
    ]
    selected.sort(key=lambda row: (row[0], row[1], row[2]))
    encoded_rows = [row[3] for row in selected]
    if not encoded_rows or sum(map(len, encoded_rows)) > maximum_jsonl_bytes:
        raise OneBTokenizerSampleError("bounded tokenizer sample differs")
    return encoded_rows, input_counts


def _write(
    output_root: Path,
    encoded_rows: list[bytes],
    input_counts: Counter[str],
    *,
    component: str,
    source_receipt_sha256: str,
    source_shard: int,
    maximum_jsonl_bytes: int,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise OneBTokenizerSampleError("tokenizer sample output exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    try:
        sample = stage / SAMPLE_NAME
        digest = hashlib.sha256()
        text_bytes = 0
        with sample.open("xb") as handle:
            for encoded in encoded_rows:
                handle.write(encoded)
                digest.update(encoded)
                text_bytes += len(json.loads(encoded)["text"].encode())
            handle.flush()
            os.fsync(handle.fileno())
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "component": component,
            "source_shard": source_shard,
            "source_receipt_sha256": source_receipt_sha256,
            "selection": {
                "method": "equal-stratum-bottom-hash-v1",
                "maximum_jsonl_bytes": maximum_jsonl_bytes,
                "development_rows_excluded": True,
            },
            "input_counts": dict(sorted(input_counts.items())),
            "sample": {
                "path": SAMPLE_NAME,
                "documents": len(encoded_rows),
                "text_utf8_bytes": text_bytes,
                "bytes": sample.stat().st_size,
                "sha256": sha256_file(sample),
                "ordered_jsonl_sha256": digest.hexdigest(),
            },
            "source_text_persisted_only_in_bounded_sample": True,
            "tokenizer_training_only": True,
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


def sample_book_shard(
    admission_root: Path,
    index_root: Path,
    source_shard: int,
    output_root: Path,
    *,
    maximum_jsonl_bytes: int,
) -> dict[str, Any]:
    """Sample one exact private Books shard using its production index rows."""

    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBTokenizerSampleError("pyarrow is required") from error
    admission = _load_signed(admission_root / "receipt.json", BOOK_SCHEMA)
    index_receipt = _load_signed(index_root / "receipt.json", INDEX_SHARD_SCHEMA)
    descriptor = index_receipt.get("output", {})
    index_path = index_root / descriptor.get("path", "")
    if (
        admission.get("training_ready") is not True
        or index_receipt.get("component") != "books"
        or index_receipt.get("source_receipt_sha256") != admission["receipt_sha256"]
        or not index_path.is_file()
        or index_path.stat().st_size != descriptor.get("bytes")
        or sha256_file(index_path) != descriptor.get("sha256")
    ):
        raise OneBTokenizerSampleError("book tokenizer source differs")
    index = pq.read_table(index_path)
    index = index.filter(pc.equal(index["source_shard"], source_shard))
    rows = {
        row["content_sha256"]: row
        for row in index.to_pylist()
        if row["split"] == "train"
    }
    if not rows:
        raise OneBTokenizerSampleError("book tokenizer index shard is empty")
    relative = f"shards/shard_{source_shard:05d}/data.parquet"
    if {row["source_path"] for row in rows.values()} != {relative}:
        raise OneBTokenizerSampleError("book source paths differ")
    source_path = Path(admission["source_text_location"]) / relative
    if not source_path.is_file() or source_path.is_symlink():
        raise OneBTokenizerSampleError("book source shard is unsafe")

    def documents() -> Iterable[tuple[str, str, dict[str, Any]]]:
        seen = set()
        for batch in pq.ParquetFile(source_path).iter_batches(
            batch_size=32,
            columns=("text", "source_content_sha256", "barcode_src"),
            use_threads=False,
        ):
            for source in batch.to_pylist():
                indexed = rows.get(source["source_content_sha256"])
                if indexed is None:
                    continue
                text = source["text"]
                if (
                    hashlib.sha256(text.encode()).hexdigest()
                    != indexed["content_sha256"]
                ):
                    raise OneBTokenizerSampleError("book source content differs")
                identity = indexed["document_identity_sha256"]
                seen.add(identity)
                yield indexed["curriculum_band"], identity, {
                    "schema": TOKENIZER_ROW_SCHEMA,
                    "text": text,
                    "selection_identity_sha256": identity,
                    "text_sha256": indexed["content_sha256"],
                    "tokenizer_training_only": True,
                    "source": {
                        "dataset": "institutional/institutional-books-hl-enriched-text",
                        "row_id": source["barcode_src"],
                        "license": "private admitted public-domain-or-cc0",
                        "domain": _tokenizer_domain(indexed["domain"]),
                    },
                }
        if seen != {row["document_identity_sha256"] for row in rows.values()}:
            raise OneBTokenizerSampleError("book tokenizer row coverage differs")

    encoded, counts = _bounded_by_stratum(documents(), maximum_jsonl_bytes)
    return _write(
        output_root,
        encoded,
        counts,
        component="books",
        source_receipt_sha256=admission["receipt_sha256"],
        source_shard=source_shard,
        maximum_jsonl_bytes=maximum_jsonl_bytes,
    )


def sample_connections(
    admission_root: Path, output_root: Path, *, maximum_jsonl_bytes: int
) -> dict[str, Any]:
    """Include the admitted train-only connection component in tokenizer fitting."""

    admission = _load_signed(admission_root / "receipt.json", BRIDGE_SCHEMA)
    descriptor = admission.get("train", {})
    source = admission_root / descriptor.get("path", "")
    if (
        admission.get("training_ready") is not True
        or descriptor.get("compression") != "gzip-mtime-0-no-filename"
        or not source.is_file()
        or source.stat().st_size != descriptor.get("bytes")
        or sha256_file(source) != descriptor.get("sha256")
    ):
        raise OneBTokenizerSampleError("connection tokenizer source differs")

    def documents() -> Iterable[tuple[str, str, dict[str, Any]]]:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get("text")
                identity = row.get("document_identity_sha256")
                if (
                    row.get("training_ready") is not True
                    or row.get("corpus_split") != "train"
                    or not isinstance(text, str)
                    or hashlib.sha256(text.encode()).hexdigest()
                    != row.get("content_sha256")
                    or not isinstance(identity, str)
                ):
                    raise OneBTokenizerSampleError("connection tokenizer row differs")
                domains = row.get("semantic_domains") or ["cross_domain"]
                yield f"difficulty-{row['difficulty_milli'] // 1000}", identity, {
                    "schema": TOKENIZER_ROW_SCHEMA,
                    "text": text,
                    "selection_identity_sha256": identity,
                    "text_sha256": row["content_sha256"],
                    "tokenizer_training_only": True,
                    "source": {
                        "dataset": "Godlydonuts/Sai::verified-connections",
                        "row_id": identity,
                        "license": "mixed reusable rights; source-bound",
                        "domain": _tokenizer_domain(" ".join(sorted(domains))),
                    },
                }

    encoded, counts = _bounded_by_stratum(documents(), maximum_jsonl_bytes)
    if counts["documents"] != admission["counts"]["train_documents"]:
        raise OneBTokenizerSampleError("connection tokenizer coverage differs")
    return _write(
        output_root,
        encoded,
        counts,
        component="connections",
        source_receipt_sha256=admission["receipt_sha256"],
        source_shard=-1,
        maximum_jsonl_bytes=maximum_jsonl_bytes,
    )


def _locator_descriptor(admission: dict[str, Any], shard: int) -> dict[str, Any]:
    descriptors = [
        row
        for row in admission.get("outputs", {}).get("descriptors", [])
        if row.get("shard_index") == shard
    ]
    if len(descriptors) != 1:
        raise OneBTokenizerSampleError("tokenizer locator descriptor differs")
    return descriptors[0]


def _download_parent(locator: dict[str, Any], token: str, root: Path) -> Path:
    if not token:
        raise OneBTokenizerSampleError("Hugging Face token is required")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise OneBTokenizerSampleError("huggingface_hub is required") from error
    try:
        path = Path(
            hf_hub_download(
                repo_id=locator["source_repository"],
                filename=locator["source_path"],
                repo_type="dataset",
                revision=locator["source_revision"],
                token=token,
                local_dir=root,
            )
        )
    except Exception as error:
        raise OneBTokenizerSampleError("tokenizer parent download failed") from error
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != locator["source_parent_sha256"]
    ):
        raise OneBTokenizerSampleError("tokenizer parent identity differs")
    return path


def _chosen_parent_paths(
    rows: list[dict[str, Any]], component: str, maximum_parents: int
) -> set[str]:
    if component == "code":
        totals: Counter[str] = Counter()
        for row in rows:
            totals[row["source_path"]] += row["text_utf8_bytes"]
        return {totals.most_common(1)[0][0]}
    by_open_type: dict[str, Counter[str]] = {}
    for row in rows:
        by_open_type.setdefault(row["open_type"], Counter())[row["source_path"]] += row[
            "text_utf8_bytes"
        ]
    choices = [
        (counter.most_common(1)[0][1], open_type, counter.most_common(1)[0][0])
        for open_type, counter in by_open_type.items()
    ]
    choices.sort(key=lambda row: (-row[0], row[1], row[2]))
    return {row[2] for row in choices[:maximum_parents]}


def _remote_documents(
    component: str,
    locators: list[dict[str, Any]],
    token: str,
    scratch_root: Path,
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in locators:
        by_parent.setdefault(row["source_path"], []).append(row)
    for parent_number, (source_path, selected) in enumerate(
        sorted(by_parent.items()), start=1
    ):
        first = selected[0]
        if any(
            row[key] != first[key]
            for row in selected
            for key in (
                "source_repository",
                "source_revision",
                "source_parent_sha256",
            )
        ):
            raise OneBTokenizerSampleError("tokenizer parent custody differs")
        with tempfile.TemporaryDirectory(
            prefix=f"sai-1b-tokenizer-{parent_number}-", dir=scratch_root
        ) as temporary:
            parent = _download_parent(first, token, Path(temporary))
            wanted = {row["source_row_index"]: row for row in selected}
            seen = set()
            if component == "code":
                with gzip.open(parent, "rt", encoding="utf-8") as handle:
                    for row_index, line in enumerate(handle):
                        locator = wanted.get(row_index)
                        if locator is None:
                            continue
                        raw = json.loads(line)
                        text = raw.get("text")
                        if (
                            not isinstance(text, str)
                            or hashlib.sha256(text.encode()).hexdigest()
                            != locator["content_sha256"]
                        ):
                            raise OneBTokenizerSampleError(
                                "code tokenizer content differs"
                            )
                        identity = locator["source_row_identity_sha256"]
                        seen.add(row_index)
                        yield locator["language"], identity, {
                            "schema": TOKENIZER_ROW_SCHEMA,
                            "text": text,
                            "selection_identity_sha256": identity,
                            "text_sha256": locator["content_sha256"],
                            "tokenizer_training_only": True,
                            "source": {
                                "dataset": (
                                    f"{locator['source_repository']}@"
                                    f"{locator['source_revision']}"
                                ),
                                "row_id": locator["blob_id"],
                                "license": "+".join(locator["licenses"]),
                                "domain": "code",
                            },
                        }
                        if len(seen) == len(wanted):
                            break
            else:
                try:
                    import pyarrow.parquet as pq
                except ImportError as error:
                    raise OneBTokenizerSampleError("pyarrow is required") from error
                next_row = 0
                for batch in pq.ParquetFile(parent).iter_batches(
                    batch_size=128,
                    columns=("text", "identifier"),
                    use_threads=False,
                ):
                    for raw in batch.to_pylist():
                        locator = wanted.get(next_row)
                        next_row += 1
                        if locator is None:
                            continue
                        text = raw.get("text")
                        if (
                            not isinstance(text, str)
                            or raw.get("identifier") != locator["identifier"]
                            or hashlib.sha256(text.encode()).hexdigest()
                            != locator["content_sha256"]
                        ):
                            raise OneBTokenizerSampleError(
                                "PleIAs tokenizer content differs"
                            )
                        identity = locator["source_row_identity_sha256"]
                        seen.add(locator["source_row_index"])
                        stratum = (
                            f"{locator['open_type']}::{locator['collection']}"
                        )
                        yield stratum, identity, {
                            "schema": TOKENIZER_ROW_SCHEMA,
                            "text": text,
                            "selection_identity_sha256": identity,
                            "text_sha256": locator["content_sha256"],
                            "tokenizer_training_only": True,
                            "source": {
                                "dataset": (
                                    f"{locator['source_repository']}@"
                                    f"{locator['source_revision']}"
                                ),
                                "row_id": locator["identifier"],
                                "license": locator["license"],
                                "domain": _tokenizer_domain(locator["collection"]),
                            },
                        }
            if seen != set(wanted):
                raise OneBTokenizerSampleError(
                    f"{component} tokenizer locator coverage differs: {source_path}"
                )


def sample_remote_component(
    component: str,
    admission_root: Path,
    locator_shard: int,
    output_root: Path,
    token: str,
    scratch_root: Path,
    *,
    maximum_jsonl_bytes: int,
    maximum_parents: int,
) -> dict[str, Any]:
    """Rehydrate bounded admitted Code or PleIAs rows from pinned parents."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBTokenizerSampleError("pyarrow is required") from error
    schema = CODE_SCHEMA if component == "code" else PLEIAS_SCHEMA
    admission = _load_signed(admission_root / "receipt.json", schema)
    descriptor = _locator_descriptor(admission, locator_shard)
    locator_path = admission_root / descriptor["path"]
    if (
        admission.get("training_ready") is not True
        or locator_path.stat().st_size != descriptor["bytes"]
        or sha256_file(locator_path) != descriptor["sha256"]
    ):
        raise OneBTokenizerSampleError("remote tokenizer admission differs")
    rows = [
        row
        for row in pq.read_table(locator_path).to_pylist()
        if int(row["source_row_identity_sha256"][:16], 16) % 1_000_000 >= 1_000
    ]
    parents = _chosen_parent_paths(rows, component, maximum_parents)
    selected = [row for row in rows if row["source_path"] in parents]
    encoded, counts = _bounded_by_stratum(
        _remote_documents(component, selected, token, scratch_root),
        maximum_jsonl_bytes,
    )
    counts["selected_source_parents"] = len(parents)
    counts["selected_admitted_locators"] = len(selected)
    return _write(
        output_root,
        encoded,
        counts,
        component=component,
        source_receipt_sha256=admission["receipt_sha256"],
        source_shard=locator_shard,
        maximum_jsonl_bytes=maximum_jsonl_bytes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    books = subparsers.add_parser("books")
    books.add_argument("--admission-root", type=Path, required=True)
    books.add_argument("--index-root", type=Path, required=True)
    books.add_argument("--source-shard", type=int, required=True)
    books.add_argument("--output-root", type=Path, required=True)
    books.add_argument("--maximum-jsonl-bytes", type=int, required=True)
    connections = subparsers.add_parser("connections")
    connections.add_argument("--admission-root", type=Path, required=True)
    connections.add_argument("--output-root", type=Path, required=True)
    connections.add_argument("--maximum-jsonl-bytes", type=int, required=True)
    for name in ("code", "pleias"):
        remote = subparsers.add_parser(name)
        remote.add_argument("--admission-root", type=Path, required=True)
        remote.add_argument("--locator-shard", type=int, required=True)
        remote.add_argument("--output-root", type=Path, required=True)
        remote.add_argument("--maximum-jsonl-bytes", type=int, required=True)
        remote.add_argument("--maximum-parents", type=int, required=True)
        remote.add_argument("--token-env", required=True)
        remote.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "books":
        result = sample_book_shard(
            args.admission_root,
            args.index_root,
            args.source_shard,
            args.output_root,
            maximum_jsonl_bytes=args.maximum_jsonl_bytes,
        )
    elif args.command == "connections":
        result = sample_connections(
            args.admission_root,
            args.output_root,
            maximum_jsonl_bytes=args.maximum_jsonl_bytes,
        )
    else:
        result = sample_remote_component(
            args.command,
            args.admission_root,
            args.locator_shard,
            args.output_root,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
            maximum_jsonl_bytes=args.maximum_jsonl_bytes,
            maximum_parents=args.maximum_parents,
        )
    print(json.dumps({"receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
