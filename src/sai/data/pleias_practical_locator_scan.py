"""Build English, reusable, mechanically clean PleIAs training locators.

This is the fast bulk path.  It intentionally does not wait for model-written
semantic labels: raw rows are admitted as candidates when their source bytes
are pinned, their language and rights are explicit, and the deterministic
non-slop gate passes.  Text is not copied into the output; the locator binds the
exact upstream parent, row index, and content hash for transient training reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_bounded_mechanical_candidates import (
    MAXIMUM_TEXT_BYTES,
    MINIMUM_TEXT_BYTES,
    MINIMUM_WORD_COUNT,
    REQUIRED_COLUMNS,
    license_allowed,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.source_quality_gate import POLICY_SHA256, mechanical_quality_evidence
from sai.data.token_stream import canonical_sha256, sha256_file

LOCATOR_SCHEMA = "sai-pleias-practical-training-locator-v1"
SHARD_SCHEMA = "sai-pleias-practical-locator-scan-shard-v1"


class PleiasPracticalLocatorScanError(RuntimeError):
    """A pinned parent, row, or practical admission invariant differs."""


def _download_pinned_parent(
    row: dict[str, Any], token: str, scratch: Path, attempts: int = 5
) -> Path:
    """Stream one exact parent with bounded reads and closed retry responses."""

    if not token or attempts < 1:
        raise PleiasPracticalLocatorScanError("parent download arguments differ")
    try:
        import httpx
    except ImportError as error:
        raise PleiasPracticalLocatorScanError("httpx is required") from error
    url = (
        "https://huggingface.co/datasets/"
        f"{quote(row['source_repository'], safe='/')}/resolve/"
        f"{quote(row['source_revision'], safe='')}/"
        f"{quote(row['source_path'], safe='/')}"
    )
    scratch.mkdir(parents=True, exist_ok=True)
    target = scratch / "parent.parquet"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept-Encoding": "identity",
        "User-Agent": "sai-practical-locator/1.0",
    }
    last_error: BaseException | None = None
    for attempt in range(attempts):
        target.unlink(missing_ok=True)
        try:
            timeout = httpx.Timeout(180.0, connect=30.0, write=30.0, pool=30.0)
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    with target.open("xb") as handle:
                        for chunk in response.iter_bytes(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
            if (
                target.stat().st_size != row["bytes"]
                or sha256_file(target) != row["sha256"]
            ):
                raise PleiasPracticalLocatorScanError("PleIAs parent identity differs")
            return target
        except (OSError, httpx.HTTPError, PleiasPracticalLocatorScanError) as error:
            last_error = error
            target.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(30, 2**attempt * 2))
    raise PleiasPracticalLocatorScanError(
        "bounded parent download failed"
    ) from last_error


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasPracticalLocatorScanError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("source_id", pa.string()),
            ("source_repository", pa.string()),
            ("source_revision", pa.string()),
            ("source_path", pa.string()),
            ("source_parent_sha256", pa.string()),
            ("source_row_index", pa.int64()),
            ("source_row_identity_sha256", pa.string()),
            ("identifier", pa.string()),
            ("collection", pa.string()),
            ("open_type", pa.string()),
            ("license", pa.string()),
            ("language", pa.string()),
            ("word_count", pa.int64()),
            ("source_token_count", pa.int64()),
            ("text_utf8_bytes", pa.int64()),
            ("content_sha256", pa.string()),
        ]
    )


def _hash_selected(identity: str, sample_ppm: int) -> bool:
    if not 1 <= sample_ppm <= 1_000_000:
        raise PleiasPracticalLocatorScanError("sample fraction differs")
    return int(identity[:16], 16) % 1_000_000 < sample_ppm


def _ordered_parents(
    parents: list[dict[str, Any]], stop_at_byte_cap: bool
) -> tuple[list[dict[str, Any]], str]:
    ordered = list(parents)
    if not stop_at_byte_cap:
        return ordered, "source_path"
    ordered.sort(
        key=lambda row: canonical_sha256(
            {"source_path": row["source_path"], "source_sha256": row["sha256"]}
        )
    )
    return ordered, "canonical_source_identity_sha256"


def _route(row: dict[str, Any]) -> tuple[str, bytes | None, str | None]:
    language = row.get("language")
    if not isinstance(language, str) or language.strip().casefold() != "english":
        return "hold_nonenglish", None, None
    if not license_allowed(row.get("license")):
        return "hold_rights", None, None
    if any(
        not isinstance(row.get(key), str) or not row[key]
        for key in ("identifier", "collection", "open_type", "text")
    ):
        return "hold_missing_required_value", None, None
    words = row.get("word_count")
    tokens = row.get("token_count")
    if (
        isinstance(words, bool)
        or not isinstance(words, int)
        or words < MINIMUM_WORD_COUNT
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens <= 0
    ):
        return "hold_structural", None, None
    content = row["text"].encode()
    if len(content) < MINIMUM_TEXT_BYTES:
        return "hold_too_short", None, None
    if len(content) > MAXIMUM_TEXT_BYTES:
        return "hold_too_large", None, None
    evidence = mechanical_quality_evidence(row["text"])
    if evidence["decision"] != "pass_mechanical_gate":
        return f"hold_{evidence['decision']}", None, None
    digest = hashlib.sha256(content).hexdigest()
    return "pass_mechanical_gate", content, digest


def run_shard(
    manifest_path: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    sample_ppm: int,
    maximum_text_bytes: int,
    stop_at_byte_cap: bool = False,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Scan an identity-disjoint source shard and emit source-safe locators."""

    if (
        not token
        or not 0 <= shard_index < logical_shards
        or maximum_text_bytes <= 0
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise PleiasPracticalLocatorScanError("practical scan arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasPracticalLocatorScanError("pyarrow is required") from error

    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    if not parents:
        raise PleiasPracticalLocatorScanError("practical scan shard is empty")
    assigned_paths_sha256 = canonical_sha256([row["source_path"] for row in parents])
    parents, parent_scan_order = _ordered_parents(parents, stop_at_byte_cap)
    output_root.mkdir(parents=True)
    output_path = output_root / "locators.parquet"
    temporary = output_root / f".locators.partial.{uuid.uuid4().hex}.parquet"
    schema = _schema()
    counts: Counter[str] = Counter()
    selected_bytes = 0
    selected_tokens = 0
    selected_identities: list[str] = []
    parent_receipts = []
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        for parent_number, parent in enumerate(parents, start=1):
            with tempfile.TemporaryDirectory(
                prefix="sai-pleias-practical-", dir=scratch_root
            ) as directory:
                local = _download_pinned_parent(parent, token, Path(directory))
                parquet = pq.ParquetFile(local)
                if not REQUIRED_COLUMNS.issubset(parquet.schema_arrow.names):
                    raise PleiasPracticalLocatorScanError("parent columns differ")
                parent_rows = 0
                parent_selected = 0
                pending = []
                for batch in parquet.iter_batches(
                    batch_size=32,
                    columns=sorted(REQUIRED_COLUMNS),
                    use_threads=False,
                ):
                    for row in batch.to_pylist():
                        row_index = parent_rows
                        parent_rows += 1
                        counts["source_rows"] += 1
                        route, content, content_sha256 = _route(row)
                        counts[route] += 1
                        if content is None or content_sha256 is None:
                            continue
                        identity = canonical_sha256(
                            {
                                "source_path": parent["source_path"],
                                "source_row_index": row_index,
                                "identifier": row["identifier"],
                                "content_sha256": content_sha256,
                            }
                        )
                        if not _hash_selected(identity, sample_ppm):
                            counts["pass_not_hash_selected"] += 1
                            continue
                        if selected_bytes + len(content) > maximum_text_bytes:
                            counts["pass_over_shard_byte_cap"] += 1
                            continue
                        pending.append(
                            {
                                "schema": LOCATOR_SCHEMA,
                                "source_id": "pleias_common_corpus",
                                "source_repository": parent["source_repository"],
                                "source_revision": parent["source_revision"],
                                "source_path": parent["source_path"],
                                "source_parent_sha256": parent["sha256"],
                                "source_row_index": row_index,
                                "source_row_identity_sha256": identity,
                                "identifier": row["identifier"],
                                "collection": row["collection"],
                                "open_type": row["open_type"],
                                "license": row["license"],
                                "language": row["language"],
                                "word_count": row["word_count"],
                                "source_token_count": row["token_count"],
                                "text_utf8_bytes": len(content),
                                "content_sha256": content_sha256,
                            }
                        )
                        selected_bytes += len(content)
                        selected_tokens += row["token_count"]
                        selected_identities.append(identity)
                        parent_selected += 1
                        counts["selected_locators"] += 1
                        if len(pending) >= 128:
                            writer.write_table(
                                pa.Table.from_pylist(pending, schema=schema)
                            )
                            pending.clear()
                if pending:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                if parent_rows != parquet.metadata.num_rows:
                    raise PleiasPracticalLocatorScanError("parent row coverage differs")
                parent_receipts.append(
                    {
                        "source_path": parent["source_path"],
                        "source_sha256": parent["sha256"],
                        "rows": parent_rows,
                        "selected_locators": parent_selected,
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "pleias_practical_scan_progress",
                        "shard_index": shard_index,
                        "complete_parents": parent_number,
                        "remaining_parents": len(parents) - parent_number,
                        "selected_text_bytes": selected_bytes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if (
                stop_at_byte_cap
                and maximum_text_bytes - selected_bytes < MINIMUM_TEXT_BYTES
            ):
                break
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temporary, output_path)
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_pleias_practical_locator_scan_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "selected_parent_count": len(parents),
            "scanned_parent_count": len(parent_receipts),
            "selected_paths_sha256": assigned_paths_sha256,
            "ordered_scanned_paths_sha256": canonical_sha256(
                [row["source_path"] for row in parent_receipts]
            ),
            "ordered_parent_receipts_sha256": canonical_sha256(parent_receipts),
        },
        "policy": {
            "english_only": True,
            "explicit_reusable_rights_only": True,
            "mechanical_quality_policy_sha256": POLICY_SHA256,
            "minimum_text_bytes": MINIMUM_TEXT_BYTES,
            "minimum_word_count": MINIMUM_WORD_COUNT,
            "maximum_text_bytes_per_row": MAXIMUM_TEXT_BYTES,
            "sample_ppm": sample_ppm,
            "maximum_text_bytes_per_shard": maximum_text_bytes,
            "parent_scan_order": parent_scan_order,
            "stop_at_byte_cap": stop_at_byte_cap,
            "model_semantic_gate_required": False,
        },
        "counts": dict(sorted(counts.items())),
        "selected": {
            "rows": counts["selected_locators"],
            "text_utf8_bytes": selected_bytes,
            "source_token_count": selected_tokens,
            "ordered_identity_sha256": canonical_sha256(selected_identities),
        },
        "output": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "all_source_rows_accounted": counts["source_rows"]
        == sum(row["rows"] for row in parent_receipts),
        "complete_assigned_parent_scan": len(parent_receipts) == len(parents),
        "deterministic_byte_cap_sampling_complete": bool(
            not stop_at_byte_cap
            or len(parent_receipts) == len(parents)
            or maximum_text_bytes - selected_bytes < MINIMUM_TEXT_BYTES
        ),
        "byte_cap_respected": selected_bytes <= maximum_text_bytes,
        "practical_candidate_complete": True,
        "global_exact_deduplication_complete": False,
        "official_benchmark_decontamination_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--sample-ppm", type=int, required=True)
    parser.add_argument("--maximum-text-bytes", type=int, required=True)
    parser.add_argument("--stop-at-byte-cap", action="store_true")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = run_shard(
        args.manifest,
        args.output_root,
        args.logical_shards,
        args.shard_index,
        os.environ.get(args.token_env, ""),
        args.sample_ppm,
        args.maximum_text_bytes,
        args.stop_at_byte_cap,
        args.scratch_root,
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
