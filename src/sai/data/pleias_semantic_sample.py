"""Select a diverse private Hermès sample from bounded PleIAs candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import CANDIDATE_SCHEMA, normalize_candidate
from sai.data.pleias_bounded_mechanical_candidates import (
    AGGREGATE_SCHEMA as BOUNDED_AGGREGATE_SCHEMA,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    CANDIDATE_SCHEMA as BOUNDED_CANDIDATE_SCHEMA,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    SHARD_SCHEMA as BOUNDED_SHARD_SCHEMA,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.reservoir_audit_population import (
    LINEAGE_SCHEMA,
    MAX_EXCERPT_BYTES,
)
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

DEFAULT_SEED = "sai-pleias-semantic-sample-20260826-r1"
TOKEN_BANDS = (512, 4_096, 32_768)


class PleiasSemanticSampleError(RuntimeError):
    """The bounded population, diversity sample, or lineage differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasSemanticSampleError("semantic sample input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasSemanticSampleError("semantic sample input is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise PleiasSemanticSampleError("semantic sample input differs")
    return payload


def _token_band(token_count: int) -> str:
    if token_count < TOKEN_BANDS[0]:
        return "lt512"
    if token_count < TOKEN_BANDS[1]:
        return "512to4095"
    if token_count < TOKEN_BANDS[2]:
        return "4096to32767"
    return "ge32768"


def _excerpt(text: str) -> tuple[str, str]:
    encoded = text.encode()
    if len(encoded) <= MAX_EXCERPT_BYTES:
        return text, "full_text_within_32768_utf8_bytes"
    character_budget = max(256, MAX_EXCERPT_BYTES // 4)
    while character_budget > 0:
        middle = max(0, (len(text) - character_budget) // 2)
        parts = (
            text[:character_budget].strip(),
            text[middle : middle + character_budget].strip(),
            text[-character_budget:].strip(),
        )
        result = "\n\n[... middle sample ...]\n\n".join(parts)
        if len(result.encode()) <= MAX_EXCERPT_BYTES:
            return result, "deterministic_beginning_middle_end_32768_utf8_bytes"
        character_budget -= 128
    raise PleiasSemanticSampleError("PleIAs excerpt cannot meet byte cap")


def _source_type(open_type: str, collection: str) -> str:
    value = f"{open_type} {collection}".casefold()
    if "github" in value or "open source" in value:
        return "code_repository"
    if "stackexchange" in value or "forum" in value:
        return "forum"
    if "science" in value or "paper" in value or "arxiv" in value:
        return "research_paper"
    if "web" in value or "wikipedia" in value:
        return "educational_web"
    return "reference"


def _candidate_and_lineage(
    row: dict[str, Any], parent: dict[str, Any], seed: str, stratum: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = row["text"]
    excerpt, excerpt_method = _excerpt(text)
    excerpt_sha256 = hashlib.sha256(excerpt.encode()).hexdigest()
    locator = {
        "source_path": row["source_path"],
        "source_row_index": row["source_row_index"],
        "identifier": row["identifier"],
        "source_row_identity_sha256": row["source_row_identity_sha256"],
    }
    provenance = canonical_sha256(
        {
            "parent_sha256": parent["sha256"],
            "locator": locator,
            "full_text_sha256": row["content_sha256"],
            "excerpt_method": excerpt_method,
            "excerpt_sha256": excerpt_sha256,
        }
    )
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text": excerpt,
        "source": {
            "dataset": row["source_repository"],
            "revision": row["source_revision"],
            "row_id": row["source_row_identity_sha256"],
            "license": row["license"],
            "source_type": _source_type(row["open_type"], row["collection"]),
        },
        "source_content_sha256": excerpt_sha256,
        "provenance_sha256": provenance,
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    candidate = normalize_candidate(candidate)
    selection_key = hashlib.sha256(
        f"{seed}:{row['source_row_identity_sha256']}".encode()
    ).hexdigest()
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "ordinal": -1,
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_id": "pleias_common_corpus",
        "stratum": stratum,
        "selection_key": selection_key,
        "repository": row["source_repository"],
        "revision": row["source_revision"],
        "license": row["license"],
        "access": "huggingface_pinned_revision",
        "path": row["source_path"],
        "parent_file_bytes": parent["bytes"],
        "parent_file_sha256": parent["sha256"],
        "locator": locator,
        "full_file_content_verified": True,
        "full_text_bytes": len(text.encode()),
        "full_text_sha256": row["content_sha256"],
        "excerpt_method": excerpt_method,
        "excerpt_bytes": len(excerpt.encode()),
        "excerpt_sha256": excerpt_sha256,
        "raw_source_is_training_ready": False,
    }
    return candidate, lineage


def _validate_bounded_inputs(
    manifest_path: Path,
    bounded_root: Path,
    bounded_aggregate_path: Path,
    logical_shards: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path]]:
    manifest = load_manifest(manifest_path)
    aggregate = _load_signed(bounded_aggregate_path, BOUNDED_AGGREGATE_SCHEMA)
    if (
        aggregate.get("status")
        != "complete_nontraining_bounded_pleias_mechanical_candidates"
        or aggregate.get("complete_source_parent_coverage") is not True
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("semantic_admission_complete") is not False
    ):
        raise PleiasSemanticSampleError("bounded aggregate differs")
    receipts = []
    paths = []
    for shard_index in range(logical_shards):
        root = bounded_root / "shards" / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", BOUNDED_SHARD_SCHEMA)
        path = root / receipt.get("output", {}).get("path", "")
        expected = select_shard(manifest, logical_shards, shard_index)
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("selected_paths_sha256")
            != canonical_sha256([row["source_path"] for row in expected])
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != receipt.get("output", {}).get("bytes")
            or sha256_file(path) != receipt.get("output", {}).get("sha256")
        ):
            raise PleiasSemanticSampleError("bounded shard differs")
        receipts.append(receipt["receipt_sha256"])
        paths.append(path)
    if canonical_sha256(receipts) != aggregate.get("shards", {}).get(
        "ordered_receipts_sha256"
    ):
        raise PleiasSemanticSampleError("bounded shard custody differs")
    return manifest, aggregate, paths


def build_population(
    manifest_path: Path,
    bounded_root: Path,
    bounded_aggregate_path: Path,
    output_root: Path,
    logical_shards: int,
    maximum_rows: int,
    maximum_rows_per_stratum: int,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Stream bounded candidates and retain a balanced deterministic sample."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not 1 <= maximum_rows <= 65_536
        or not 1 <= maximum_rows_per_stratum <= 64
        or not isinstance(seed, str)
        or not seed
    ):
        raise PleiasSemanticSampleError("semantic sample geometry differs")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasSemanticSampleError("pyarrow is required") from error
    manifest, aggregate, paths = _validate_bounded_inputs(
        manifest_path, bounded_root, bounded_aggregate_path, logical_shards
    )
    parents = {row["source_path"]: row for row in manifest}
    by_stratum: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    scanned = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                scanned += 1
                if (
                    row.get("schema") != BOUNDED_CANDIDATE_SCHEMA
                    or row.get("training_ready") is not False
                    or row.get("source_path") not in parents
                    or hashlib.sha256(row.get("text", "").encode()).hexdigest()
                    != row.get("content_sha256")
                ):
                    raise PleiasSemanticSampleError("bounded candidate row differs")
                stratum = "::".join(
                    (
                        row["collection"],
                        row["open_type"],
                        _token_band(row["token_count"]),
                    )
                )
                candidate, lineage = _candidate_and_lineage(
                    row, parents[row["source_path"]], seed, stratum
                )
                rank = lineage["selection_key"]
                bucket = by_stratum[stratum]
                bucket.append((rank, candidate, lineage))
                bucket.sort(key=lambda item: item[0])
                if len(bucket) > maximum_rows_per_stratum:
                    bucket.pop()
    if scanned != aggregate.get("totals", {}).get("selected_rows"):
        raise PleiasSemanticSampleError("bounded candidate row coverage differs")
    ordered_strata = sorted(
        by_stratum,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    selected = []
    offset = 0
    while len(selected) < maximum_rows:
        added = False
        for stratum in ordered_strata:
            bucket = by_stratum[stratum]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == maximum_rows:
                    break
        if not added:
            break
        offset += 1
    if not selected:
        raise PleiasSemanticSampleError("semantic sample is empty")
    candidates = []
    lineage = []
    for ordinal, (_rank, candidate, row) in enumerate(selected):
        row["ordinal"] = ordinal
        row["lineage_sha256"] = canonical_sha256(row)
        candidates.append(candidate)
        lineage.append(row)
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(identities) != len(set(identities)):
        raise PleiasSemanticSampleError("semantic sample identities overlap")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        with candidate_path.open("x") as handle:
            for row in candidates:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        with lineage_path.open("x") as handle:
            for row in lineage:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        by_stratum_counts = Counter(row["stratum"] for row in lineage)
        receipt = {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "seed": seed,
            "selection_method": (
                "per_collection_open_type_token_band_lowest_identity_hash_"
                "then_seeded_stratum_round_robin"
            ),
            "statistically_representative": False,
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_sha256": aggregate["receipt_sha256"],
                "selected_files": logical_shards,
                "selected_bytes": aggregate["totals"]["output_bytes"],
            },
            "population": {
                "path": candidate_path.name,
                "rows": len(candidates),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage),
            },
            "by_source": {"pleias_common_corpus": len(candidates)},
            "by_stratum": dict(sorted(by_stratum_counts.items())),
            "bounded_candidate_rows_scanned": scanned,
            "available_strata": len(by_stratum),
            "maximum_rows": maximum_rows,
            "maximum_rows_per_stratum": maximum_rows_per_stratum,
            "range_read_parent_files": 0,
            "fully_verified_parent_files": len(manifest),
            "hermes_judgments_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bounded-root", type=Path, required=True)
    parser.add_argument("--bounded-aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--maximum-rows", type=int, default=8192)
    parser.add_argument("--maximum-rows-per-stratum", type=int, default=32)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    receipt = build_population(
        args.manifest,
        args.bounded_root,
        args.bounded_aggregate,
        args.output_root,
        args.logical_shards,
        args.maximum_rows,
        args.maximum_rows_per_stratum,
        args.seed,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
