"""Census full PleIAs production candidates without persisting source text."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import _WORD
from sai.data.pleias_bounded_mechanical_candidates import (
    POLICY_SCHEMA,
    REQUIRED_COLUMNS,
    _download,
    _load_signed,
    _routes,
    evaluate_row,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_semantic_sample import _token_band
from sai.data.pleias_semantic_stratum_decision import CORE_SCORES
from sai.data.pleias_semantic_stratum_decision import SCHEMA as DECISION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-production-descriptor-census-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-production-descriptor-census-aggregate-v1"
DESCRIPTOR_SCHEMA = "sai-pleias-production-candidate-descriptor-v1"
NEAR_SHINGLE_WORDS = 5
NEAR_BOTTOM_K = 32


class PleiasProductionDescriptorCensusError(RuntimeError):
    """The production evidence, parent identity, or descriptor replay differs."""


def _signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasProductionDescriptorCensusError("signed input is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasProductionDescriptorCensusError(
            "signed input is invalid"
        ) from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasProductionDescriptorCensusError("signed input differs")
    return value


def advanced_stratum_quality(decision: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Replay advanced strata and their conservative semantic-quality ranks."""

    rows = decision.get("decisions")
    advanced = decision.get("advanced_strata")
    if (
        decision.get("status")
        != "complete_nontraining_pleias_semantic_stratum_decision"
        or not isinstance(rows, list)
        or not isinstance(advanced, list)
        or len(advanced) != len(set(advanced))
    ):
        raise PleiasProductionDescriptorCensusError("semantic decision differs")
    replay = []
    expected = []
    quality: dict[str, tuple[int, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise PleiasProductionDescriptorCensusError("semantic row differs")
        unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
        if (
            row.get("row_sha256") != canonical_sha256(unsigned)
            or row.get("automatic_training_admission") is not False
        ):
            raise PleiasProductionDescriptorCensusError("semantic row differs")
        replay.append(row["row_sha256"])
        if row.get("decision") == "advance_to_full_candidate_decontamination":
            stratum = row.get("stratum")
            means = row.get("primary", {}).get("core_mean_scores_milli")
            if (
                not isinstance(stratum, str)
                or not stratum
                or stratum in quality
                or not isinstance(means, dict)
                or set(means) != set(CORE_SCORES)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 5_000
                    for value in means.values()
                )
            ):
                raise PleiasProductionDescriptorCensusError(
                    "advanced stratum quality differs"
                )
            values = [means[key] for key in CORE_SCORES]
            quality[stratum] = (min(values), sum(values) // len(values))
            expected.append(stratum)
    if (
        canonical_sha256(replay) != decision.get("ordered_decisions_sha256")
        or expected != advanced
        or any(not isinstance(value, str) or not value for value in advanced)
    ):
        raise PleiasProductionDescriptorCensusError("advanced strata differ")
    return quality


def advanced_strata(decision: dict[str, Any]) -> frozenset[str]:
    """Replay the exact advanced-stratum list from the semantic decision."""

    return frozenset(advanced_stratum_quality(decision))


def normalized_text(value: str) -> str:
    """Canonicalize surface-only differences for normalized exact deduplication."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def bottom_k_word_shingles(value: str, *, size: int = NEAR_BOTTOM_K) -> list[int]:
    """Return a stable KMV sketch for document-level near-duplicate discovery."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise PleiasProductionDescriptorCensusError("near-dedup sketch size differs")
    tokens = _WORD.findall(normalized_text(value))
    unique: set[int] = set()
    heap: list[int] = []
    for index in range(max(0, len(tokens) - NEAR_SHINGLE_WORDS + 1)):
        window = "\x1f".join(tokens[index : index + NEAR_SHINGLE_WORDS]).encode()
        digest = int.from_bytes(
            hashlib.blake2b(
                window,
                digest_size=8,
                person=b"sai-near-v1",
            ).digest(),
            "big",
        )
        if digest in unique:
            continue
        if len(heap) < size:
            heapq.heappush(heap, -digest)
            unique.add(digest)
            continue
        largest = -heap[0]
        if digest < largest:
            heapq.heapreplace(heap, -digest)
            unique.remove(largest)
            unique.add(digest)
    return sorted(unique)


def descriptor(
    row: dict[str, Any],
    parent: dict[str, Any],
    row_index: int,
    *,
    stratum_quality_floor_milli: int = 5_000,
    stratum_quality_mean_milli: int = 5_000,
) -> dict[str, Any]:
    """Build a source-safe descriptor for one mechanically eligible row."""

    if (
        isinstance(stratum_quality_floor_milli, bool)
        or not isinstance(stratum_quality_floor_milli, int)
        or isinstance(stratum_quality_mean_milli, bool)
        or not isinstance(stratum_quality_mean_milli, int)
        or not 0
        <= stratum_quality_floor_milli
        <= stratum_quality_mean_milli
        <= 5_000
    ):
        raise PleiasProductionDescriptorCensusError(
            "descriptor stratum quality differs"
        )
    text = row["text"]
    text_bytes = text.encode()
    content_sha256 = hashlib.sha256(text_bytes).hexdigest()
    identity = canonical_sha256(
        {
            "source_path": parent["source_path"],
            "row_index": row_index,
            "identifier": row["identifier"],
            "content_sha256": content_sha256,
        }
    )
    value = {
        "schema": DESCRIPTOR_SCHEMA,
        "source_path": parent["source_path"],
        "source_parent_sha256": parent["sha256"],
        "source_row_index": row_index,
        "source_row_identity_sha256": identity,
        "identifier_sha256": hashlib.sha256(row["identifier"].encode()).hexdigest(),
        "collection": row["collection"],
        "open_type": row["open_type"],
        "license": row["license"],
        "language": row["language"],
        "stratum": "::".join(
            (row["collection"], row["open_type"], _token_band(row["token_count"]))
        ),
        "stratum_quality_floor_milli": stratum_quality_floor_milli,
        "stratum_quality_mean_milli": stratum_quality_mean_milli,
        "word_count": row["word_count"],
        "token_count": row["token_count"],
        "text_utf8_bytes": len(text_bytes),
        "content_sha256": content_sha256,
        "normalized_content_sha256": hashlib.sha256(
            normalized_text(text).encode()
        ).hexdigest(),
        "near_dedup_bottom_k_u64": bottom_k_word_shingles(text),
        "training_ready": False,
    }
    value["descriptor_sha256"] = canonical_sha256(value)
    return value


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasProductionDescriptorCensusError("pyarrow is required") from error
    return pa.schema(
        [
            ("schema", pa.string()),
            ("source_path", pa.string()),
            ("source_parent_sha256", pa.string()),
            ("source_row_index", pa.int64()),
            ("source_row_identity_sha256", pa.string()),
            ("identifier_sha256", pa.string()),
            ("collection", pa.string()),
            ("open_type", pa.string()),
            ("license", pa.string()),
            ("language", pa.string()),
            ("stratum", pa.string()),
            ("stratum_quality_floor_milli", pa.int32()),
            ("stratum_quality_mean_milli", pa.int32()),
            ("word_count", pa.int64()),
            ("token_count", pa.int64()),
            ("text_utf8_bytes", pa.int64()),
            ("content_sha256", pa.string()),
            ("normalized_content_sha256", pa.string()),
            ("near_dedup_bottom_k_u64", pa.list_(pa.uint64())),
            ("training_ready", pa.bool_()),
            ("descriptor_sha256", pa.string()),
        ]
    )


def run_shard(
    manifest_path: Path,
    policy_path: Path,
    decision_path: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Scan every selected parent and emit only advanced candidate descriptors."""

    if (
        not token
        or not 0 <= shard_index < logical_shards
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise PleiasProductionDescriptorCensusError("descriptor arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasProductionDescriptorCensusError("pyarrow is required") from error
    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    if not parents:
        raise PleiasProductionDescriptorCensusError("descriptor shard is empty")
    policy = _load_signed(policy_path, POLICY_SCHEMA)
    routes = _routes(policy)
    decision = _signed(decision_path, DECISION_SCHEMA)
    allowed_strata = advanced_stratum_quality(decision)
    output_root.mkdir(parents=True)
    output_path = output_root / "candidate_descriptors.parquet"
    temporary = output_root / f".descriptors.partial.{uuid.uuid4().hex}.parquet"
    schema = _schema()
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    counts: Counter[str] = Counter()
    parent_receipts = []
    ordered_descriptors = hashlib.sha256()
    try:
        for parent_number, parent in enumerate(parents, start=1):
            with tempfile.TemporaryDirectory(
                prefix="sai-pleias-production-census-", dir=scratch_root
            ) as directory:
                source_path = _download(parent, token, Path(directory))
                parquet = pq.ParquetFile(source_path)
                if not REQUIRED_COLUMNS.issubset(parquet.schema_arrow.names):
                    raise PleiasProductionDescriptorCensusError(
                        "production parent schema differs"
                    )
                parent_rows = 0
                parent_candidates = 0
                batch_out = []
                for batch in parquet.iter_batches(
                    batch_size=16,
                    columns=sorted(REQUIRED_COLUMNS),
                    use_threads=False,
                ):
                    for row in batch.to_pylist():
                        row_index = parent_rows
                        parent_rows += 1
                        counts["source_rows"] += 1
                        route, _evidence = evaluate_row(row, routes)
                        counts[route] += 1
                        if route != "pass_mechanical_gate":
                            continue
                        stratum = "::".join(
                            (
                                row["collection"],
                                row["open_type"],
                                _token_band(row["token_count"]),
                            )
                        )
                        if stratum not in allowed_strata:
                            counts["hold_semantic_stratum"] += 1
                            continue
                        quality_floor, quality_mean = allowed_strata[stratum]
                        value = descriptor(
                            row,
                            parent,
                            row_index,
                            stratum_quality_floor_milli=quality_floor,
                            stratum_quality_mean_milli=quality_mean,
                        )
                        batch_out.append(value)
                        ordered_descriptors.update(
                            bytes.fromhex(value["descriptor_sha256"])
                        )
                        counts["production_candidate_descriptors"] += 1
                        counts["production_candidate_text_utf8_bytes"] += value[
                            "text_utf8_bytes"
                        ]
                        counts["production_candidate_source_tokens"] += value[
                            "token_count"
                        ]
                        parent_candidates += 1
                        if len(batch_out) >= 16:
                            writer.write_table(
                                pa.Table.from_pylist(batch_out, schema=schema)
                            )
                            batch_out.clear()
                if batch_out:
                    writer.write_table(pa.Table.from_pylist(batch_out, schema=schema))
                if parent_rows != parquet.metadata.num_rows:
                    raise PleiasProductionDescriptorCensusError(
                        "production parent row coverage differs"
                    )
                parent_receipts.append(
                    {
                        "source_path": parent["source_path"],
                        "source_sha256": parent["sha256"],
                        "rows": parent_rows,
                        "production_candidate_descriptors": parent_candidates,
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "pleias_production_descriptor_progress",
                        "shard_index": shard_index,
                        "complete_parents": parent_number,
                        "remaining_parents": len(parents) - parent_number,
                        "candidate_descriptors": counts[
                            "production_candidate_descriptors"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    os.replace(temporary, output_path)
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_production_descriptor_census_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "policy_file_sha256": sha256_file(policy_path),
            "policy_receipt_sha256": policy["receipt_sha256"],
            "semantic_decision_file_sha256": sha256_file(decision_path),
            "semantic_decision_receipt_sha256": decision["receipt_sha256"],
            "selected_parent_count": len(parents),
            "selected_paths_sha256": canonical_sha256(
                [row["source_path"] for row in parents]
            ),
            "ordered_parent_receipts_sha256": canonical_sha256(parent_receipts),
        },
        "policy": {
            "direct_group_route_only": True,
            "english_only": True,
            "explicit_rights_allowlist_only": True,
            "advanced_semantic_strata_only": True,
            "normalized_exact_signature": "NFKC_casefold_whitespace_collapse_sha256",
            "near_signature": {
                "method": "bottom_k_unique_blake2b_u64",
                "word_shingle_width": NEAR_SHINGLE_WORDS,
                "bottom_k": NEAR_BOTTOM_K,
                "hash_personalization": "sai-near-v1",
            },
        },
        "counts": dict(sorted(counts.items())),
        "ordered_descriptor_digests_sha256": ordered_descriptors.hexdigest(),
        "output": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "source_text_persisted": False,
        "complete_source_row_coverage": counts["source_rows"]
        == sum(row["rows"] for row in parent_receipts),
        "benchmark_decontamination_complete": False,
        "global_exact_deduplication_complete": False,
        "global_normalized_exact_deduplication_complete": False,
        "global_near_deduplication_complete": False,
        "production_selection_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate(
    manifest_path: Path,
    policy_path: Path,
    decision_path: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Seal parent-disjoint descriptor custody without admitting any content."""

    if output.exists() or output.is_symlink():
        raise PleiasProductionDescriptorCensusError("aggregate output exists")
    manifest = load_manifest(manifest_path)
    policy = _load_signed(policy_path, POLICY_SCHEMA)
    decision = _signed(decision_path, DECISION_SCHEMA)
    expected_paths = {row["source_path"] for row in manifest}
    seen_paths: set[str] = set()
    receipts = []
    totals: Counter[str] = Counter()
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _signed(root / "receipt.json", SHARD_SCHEMA)
        parents = select_shard(manifest, logical_shards, shard_index)
        paths = {row["source_path"] for row in parents}
        data_path = root / receipt.get("output", {}).get("path", "")
        if (
            receipt.get("status")
            != "complete_nontraining_pleias_production_descriptor_census_shard"
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("manifest_sha256")
            != sha256_file(manifest_path)
            or receipt.get("source", {}).get("policy_receipt_sha256")
            != policy["receipt_sha256"]
            or receipt.get("source", {}).get("semantic_decision_receipt_sha256")
            != decision["receipt_sha256"]
            or receipt.get("source", {}).get("selected_paths_sha256")
            != canonical_sha256([row["source_path"] for row in parents])
            or receipt.get("complete_source_row_coverage") is not True
            or receipt.get("source_text_persisted") is not False
            or not data_path.is_file()
            or data_path.is_symlink()
            or data_path.stat().st_nlink != 1
            or data_path.stat().st_size != receipt.get("output", {}).get("bytes")
            or sha256_file(data_path) != receipt.get("output", {}).get("sha256")
            or seen_paths.intersection(paths)
        ):
            raise PleiasProductionDescriptorCensusError("descriptor shard differs")
        seen_paths.update(paths)
        for key, value in receipt["counts"].items():
            totals[key] += value
        totals["descriptor_output_bytes"] += receipt["output"]["bytes"]
        receipts.append(receipt["receipt_sha256"])
    if seen_paths != expected_paths:
        raise PleiasProductionDescriptorCensusError(
            "descriptor parent coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_production_descriptor_census",
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "policy_receipt_sha256": policy["receipt_sha256"],
            "semantic_decision_receipt_sha256": decision["receipt_sha256"],
            "source_parent_count": len(manifest),
            "source_parent_bytes": sum(row["bytes"] for row in manifest),
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "complete_source_parent_coverage": True,
        "source_text_persisted": False,
        "benchmark_decontamination_complete": False,
        "global_exact_deduplication_complete": False,
        "global_normalized_exact_deduplication_complete": False,
        "global_near_deduplication_complete": False,
        "production_selection_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--policy", type=Path, required=True)
    shard.add_argument("--decision", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--manifest", type=Path, required=True)
    combine.add_argument("--policy", type=Path, required=True)
    combine.add_argument("--decision", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.policy,
            args.decision,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate(
            args.manifest,
            args.policy,
            args.decision,
            args.shards_root,
            args.logical_shards,
            args.output,
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
