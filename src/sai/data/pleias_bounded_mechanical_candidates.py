"""Build a bounded, fail-closed PleIAs row-level candidate population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_quality_core_policy import SCHEMA as POLICY_SCHEMA
from sai.data.source_quality_gate import POLICY_SHA256 as MECHANICAL_POLICY_SHA256
from sai.data.source_quality_gate import mechanical_quality_evidence
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-bounded-mechanical-candidates-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-bounded-mechanical-candidates-aggregate-v1"
CANDIDATE_SCHEMA = "sai-pleias-bounded-mechanical-candidate-v1"
DIRECT_ROUTE = "priority_direct_representation_verification"

# Exact, deliberately narrow labels. Unknown, composite, NC, and ND rights labels
# fail closed. Additional labels require a policy revision and test evidence.
EXACT_ALLOWED_LICENSES = frozenset(
    {
        "apache-2.0",
        "apache 2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0",
        "cc zero",
        "isc",
        "mit",
        "public domain",
        "u.s. federal public domain",
        "us federal public domain",
    }
)
_CC_BY = re.compile(r"cc[- ]by(?:[- ]sa)?(?:[- ]\d(?:\.\d)?)?", re.IGNORECASE)
REQUIRED_COLUMNS = frozenset(
    {
        "identifier",
        "collection",
        "open_type",
        "license",
        "language",
        "word_count",
        "token_count",
        "text",
    }
)
MINIMUM_TEXT_BYTES = 512
MINIMUM_WORD_COUNT = 64
MAXIMUM_TEXT_BYTES = 4 * 1024 * 1024


class PleiasBoundedMechanicalCandidatesError(RuntimeError):
    """A PleIAs parent, policy, row decision, or byte cap differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasBoundedMechanicalCandidatesError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasBoundedMechanicalCandidatesError(
            "signed input is invalid"
        ) from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise PleiasBoundedMechanicalCandidatesError("signed input differs")
    return payload


def _routes(policy: dict[str, Any]) -> dict[tuple[str, str], str]:
    if (
        policy.get("schema") != POLICY_SCHEMA
        or policy.get("status")
        != "complete_nontraining_pleias_quality_core_work_policy"
        or policy.get("automatic_training_admission") is not False
        or policy.get("training_ready") is not False
    ):
        raise PleiasBoundedMechanicalCandidatesError("PleIAs work policy differs")
    result = {}
    row_hashes = []
    for row in policy.get("groups", []):
        unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
        key = (row.get("collection"), row.get("language"))
        if (
            not all(isinstance(value, str) and value for value in key)
            or key in result
            or not isinstance(row.get("work_route"), str)
            or row.get("row_sha256") != canonical_sha256(unsigned)
            or row.get("automatic_training_admission") is not False
        ):
            raise PleiasBoundedMechanicalCandidatesError("PleIAs group route differs")
        result[key] = row.get("work_route")
        row_hashes.append(row["row_sha256"])
    if not result or canonical_sha256(row_hashes) != policy.get(
        "ordered_group_rows_sha256"
    ):
        raise PleiasBoundedMechanicalCandidatesError("PleIAs route coverage differs")
    return result


def license_allowed(value: Any) -> bool:
    """Return true only for a complete, explicitly reusable rights label."""

    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().casefold().split())
    return normalized in EXACT_ALLOWED_LICENSES or bool(_CC_BY.fullmatch(normalized))


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def evaluate_row(
    row: dict[str, Any], routes: dict[tuple[str, str], str]
) -> tuple[str, dict[str, Any] | None]:
    """Return a deterministic row route and quality evidence when scanned."""

    collection = _text(row.get("collection"))
    language = _text(row.get("language"))
    if collection is None or language is None or (collection, language) not in routes:
        raise PleiasBoundedMechanicalCandidatesError("PleIAs row group differs")
    if routes[(collection, language)] != DIRECT_ROUTE:
        return "hold_group_route", None
    if language.casefold() != "english":
        return "hold_nonenglish", None
    if not license_allowed(row.get("license")):
        return "hold_rights", None
    identifier = _text(row.get("identifier"))
    open_type = _text(row.get("open_type"))
    text = _text(row.get("text"))
    words = row.get("word_count")
    tokens = row.get("token_count")
    if identifier is None or open_type is None:
        return "hold_missing_identifier", None
    if text is None:
        return "hold_missing_text", None
    if (
        isinstance(words, bool)
        or not isinstance(words, int)
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or words < MINIMUM_WORD_COUNT
        or tokens <= 0
    ):
        return "hold_structural", None
    text_bytes = len(text.encode())
    if text_bytes < MINIMUM_TEXT_BYTES:
        return "hold_too_short", None
    if text_bytes > MAXIMUM_TEXT_BYTES:
        return "hold_too_large_for_row_candidate", None
    evidence = mechanical_quality_evidence(text)
    decision = evidence["decision"]
    if decision != "pass_mechanical_gate":
        return f"hold_{decision}", evidence
    return "pass_mechanical_gate", evidence


def content_selected(
    source_path: str,
    row_index: int,
    identifier: str,
    content_sha256: str,
    sample_ppm: int,
) -> bool:
    """Apply a stable source-row hash sample without depending on scan order."""

    if not 1 <= sample_ppm <= 1_000_000:
        raise PleiasBoundedMechanicalCandidatesError("sample fraction differs")
    identity = canonical_sha256(
        {
            "source_path": source_path,
            "row_index": row_index,
            "identifier": identifier,
            "content_sha256": content_sha256,
        }
    )
    return int(identity[:16], 16) % 1_000_000 < sample_ppm


def _download(row: dict[str, Any], token: str, scratch: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise PleiasBoundedMechanicalCandidatesError(
            "huggingface_hub is required"
        ) from error
    downloaded = Path(
        hf_hub_download(
            repo_id=row["source_repository"],
            filename=row["source_path"],
            repo_type="dataset",
            revision=row["source_revision"],
            token=token,
            cache_dir=scratch / "cache",
            local_dir=scratch / "local",
        )
    )
    if (
        not downloaded.is_file()
        or downloaded.stat().st_size != row["bytes"]
        or sha256_file(downloaded) != row["sha256"]
    ):
        raise PleiasBoundedMechanicalCandidatesError("PleIAs parent identity differs")
    return downloaded


def _schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasBoundedMechanicalCandidatesError("pyarrow is required") from error
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
            ("token_count", pa.int64()),
            ("content_sha256", pa.string()),
            ("text", pa.string()),
            ("training_ready", pa.bool_()),
        ]
    )


def run_shard(
    manifest_path: Path,
    policy_path: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    sample_ppm: int,
    maximum_text_bytes: int,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Rescan exact parents and retain a deterministic, byte-bounded population."""

    if (
        not token
        or not 0 <= shard_index < logical_shards
        or maximum_text_bytes <= 0
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise PleiasBoundedMechanicalCandidatesError("PleIAs shard arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasBoundedMechanicalCandidatesError("pyarrow is required") from error
    manifest = load_manifest(manifest_path)
    selected_parents = select_shard(manifest, logical_shards, shard_index)
    if not selected_parents:
        raise PleiasBoundedMechanicalCandidatesError("PleIAs shard is empty")
    policy = _load_signed(policy_path, POLICY_SCHEMA)
    routes = _routes(policy)
    output_root.mkdir(parents=True)
    output_path = output_root / "candidates.parquet"
    temporary = output_root / f".candidates.partial.{uuid.uuid4().hex}.parquet"
    schema = _schema()
    counts: Counter[str] = Counter()
    selected_text_bytes = 0
    selected_tokens = 0
    selected_hashes = []
    parent_receipts = []
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        for parent_number, parent in enumerate(selected_parents, start=1):
            with tempfile.TemporaryDirectory(
                prefix="sai-pleias-row-gate-", dir=scratch_root
            ) as directory:
                source_path = _download(parent, token, Path(directory))
                parquet = pq.ParquetFile(source_path)
                if not REQUIRED_COLUMNS.issubset(parquet.schema_arrow.names):
                    raise PleiasBoundedMechanicalCandidatesError(
                        "PleIAs parent columns differ"
                    )
                parent_rows = 0
                parent_selected = 0
                batch_out = []
                for batch in parquet.iter_batches(
                    batch_size=32,
                    columns=sorted(REQUIRED_COLUMNS),
                    use_threads=False,
                ):
                    for row in batch.to_pylist():
                        row_index = parent_rows
                        parent_rows += 1
                        counts["source_rows"] += 1
                        route, evidence = evaluate_row(row, routes)
                        counts[route] += 1
                        if route != "pass_mechanical_gate":
                            continue
                        content = row["text"].encode()
                        content_sha256 = hashlib.sha256(content).hexdigest()
                        if not content_selected(
                            parent["source_path"],
                            row_index,
                            row["identifier"],
                            content_sha256,
                            sample_ppm,
                        ):
                            counts["pass_not_hash_selected"] += 1
                            continue
                        if selected_text_bytes + len(content) > maximum_text_bytes:
                            counts["pass_over_shard_byte_cap"] += 1
                            continue
                        identity = canonical_sha256(
                            {
                                "source_path": parent["source_path"],
                                "row_index": row_index,
                                "identifier": row["identifier"],
                                "content_sha256": content_sha256,
                            }
                        )
                        candidate = {
                            "schema": CANDIDATE_SCHEMA,
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
                            "token_count": row["token_count"],
                            "content_sha256": content_sha256,
                            "text": row["text"],
                            "training_ready": False,
                        }
                        batch_out.append(candidate)
                        selected_hashes.append(identity)
                        selected_text_bytes += len(content)
                        selected_tokens += row["token_count"]
                        parent_selected += 1
                        counts["selected_candidates"] += 1
                        if len(batch_out) >= 32:
                            writer.write_table(
                                pa.Table.from_pylist(batch_out, schema=schema)
                            )
                            batch_out.clear()
                if batch_out:
                    writer.write_table(pa.Table.from_pylist(batch_out, schema=schema))
                if parent_rows != parquet.metadata.num_rows:
                    raise PleiasBoundedMechanicalCandidatesError(
                        "PleIAs parent row coverage differs"
                    )
                parent_receipts.append(
                    {
                        "source_path": parent["source_path"],
                        "source_sha256": parent["sha256"],
                        "rows": parent_rows,
                        "selected_candidates": parent_selected,
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "pleias_bounded_mechanical_progress",
                        "shard_index": shard_index,
                        "complete_parents": parent_number,
                        "remaining_parents": len(selected_parents) - parent_number,
                        "selected_candidates": counts["selected_candidates"],
                        "selected_text_bytes": selected_text_bytes,
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
        "status": "complete_nontraining_bounded_pleias_mechanical_candidates_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "policy_file_sha256": sha256_file(policy_path),
            "policy_receipt_sha256": policy["receipt_sha256"],
            "selected_parent_count": len(selected_parents),
            "selected_paths_sha256": canonical_sha256(
                [row["source_path"] for row in selected_parents]
            ),
            "ordered_parent_receipts_sha256": canonical_sha256(parent_receipts),
        },
        "policy": {
            "mechanical_policy_sha256": MECHANICAL_POLICY_SHA256,
            "direct_group_route_only": True,
            "english_only": True,
            "explicit_rights_allowlist_only": True,
            "minimum_text_bytes": MINIMUM_TEXT_BYTES,
            "minimum_word_count": MINIMUM_WORD_COUNT,
            "maximum_text_bytes_per_row": MAXIMUM_TEXT_BYTES,
            "sample_ppm": sample_ppm,
            "maximum_text_bytes_per_shard": maximum_text_bytes,
        },
        "counts": dict(sorted(counts.items())),
        "selected": {
            "rows": counts["selected_candidates"],
            "text_utf8_bytes": selected_text_bytes,
            "source_token_count": selected_tokens,
            "ordered_identity_sha256": canonical_sha256(selected_hashes),
        },
        "output": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "all_source_rows_accounted": counts["source_rows"]
        == sum(row["rows"] for row in parent_receipts),
        "byte_cap_respected": selected_text_bytes <= maximum_text_bytes,
        "global_exact_deduplication_complete": False,
        "global_near_deduplication_complete": False,
        "benchmark_decontamination_complete": False,
        "semantic_admission_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate(
    manifest_path: Path,
    policy_path: Path,
    shards_root: Path,
    logical_shards: int,
    maximum_text_bytes_per_shard: int,
    output: Path,
) -> dict[str, Any]:
    """Seal exact shard custody without declaring semantic admission."""

    if output.exists() or output.is_symlink():
        raise PleiasBoundedMechanicalCandidatesError("aggregate output exists")
    policy = _load_signed(policy_path, POLICY_SCHEMA)
    manifest = load_manifest(manifest_path)
    expected_paths = {row["source_path"] for row in manifest}
    seen_paths = set()
    totals: Counter[str] = Counter()
    receipts = []
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        selected = select_shard(manifest, logical_shards, shard_index)
        output_path = root / receipt.get("output", {}).get("path", "")
        paths = {row["source_path"] for row in selected}
        if (
            receipt.get("status")
            != "complete_nontraining_bounded_pleias_mechanical_candidates_shard"
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("manifest_sha256")
            != sha256_file(manifest_path)
            or receipt.get("source", {}).get("policy_receipt_sha256")
            != policy["receipt_sha256"]
            or receipt.get("source", {}).get("selected_paths_sha256")
            != canonical_sha256([row["source_path"] for row in selected])
            or receipt.get("policy", {}).get("maximum_text_bytes_per_shard")
            != maximum_text_bytes_per_shard
            or receipt.get("training_ready") is not False
            or receipt.get("byte_cap_respected") is not True
            or not output_path.is_file()
            or output_path.is_symlink()
            or output_path.stat().st_nlink != 1
            or output_path.stat().st_size != receipt["output"]["bytes"]
            or sha256_file(output_path) != receipt["output"]["sha256"]
            or seen_paths.intersection(paths)
        ):
            raise PleiasBoundedMechanicalCandidatesError("candidate shard differs")
        seen_paths.update(paths)
        for key, value in receipt["counts"].items():
            totals[f"route::{key}"] += value
        totals["selected_rows"] += receipt["selected"]["rows"]
        totals["selected_text_utf8_bytes"] += receipt["selected"]["text_utf8_bytes"]
        totals["selected_source_token_count"] += receipt["selected"][
            "source_token_count"
        ]
        totals["output_bytes"] += receipt["output"]["bytes"]
        receipts.append(receipt["receipt_sha256"])
    if seen_paths != expected_paths:
        raise PleiasBoundedMechanicalCandidatesError(
            "candidate parent coverage differs"
        )
    maximum_total = logical_shards * maximum_text_bytes_per_shard
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_bounded_pleias_mechanical_candidates",
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "policy_file_sha256": sha256_file(policy_path),
            "policy_receipt_sha256": policy["receipt_sha256"],
            "source_parent_count": len(manifest),
            "source_parent_bytes": sum(row["bytes"] for row in manifest),
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "bounds": {
            "maximum_text_bytes_per_shard": maximum_text_bytes_per_shard,
            "maximum_total_text_bytes": maximum_total,
            "maximum_total_text_bytes_respected": totals["selected_text_utf8_bytes"]
            <= maximum_total,
        },
        "totals": dict(sorted(totals.items())),
        "complete_source_parent_coverage": True,
        "mechanical_candidates_are_training_admission": False,
        "global_exact_deduplication_complete": False,
        "global_near_deduplication_complete": False,
        "benchmark_decontamination_complete": False,
        "semantic_admission_complete": False,
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
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--sample-ppm", type=int, required=True)
    shard.add_argument("--maximum-text-bytes", type=int, required=True)
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--manifest", type=Path, required=True)
    combine.add_argument("--policy", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--maximum-text-bytes-per-shard", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        token = os.environ.get(args.token_env, "")
        result = run_shard(
            args.manifest,
            args.policy,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            token,
            args.sample_ppm,
            args.maximum_text_bytes,
            args.scratch_root,
        )
    else:
        result = aggregate(
            args.manifest,
            args.policy,
            args.shards_root,
            args.logical_shards,
            args.maximum_text_bytes_per_shard,
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
