"""Replay, decontaminate, upload, and verify selected PleIAs candidate shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import binary_boundary_index
from sai.data.pleias_bounded_mechanical_candidates import (
    CANDIDATE_SCHEMA,
    REQUIRED_COLUMNS,
    _download,
    _schema,
)
from sai.data.pleias_full_candidate_decontamination import screen_text
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_production_byte_selection import SCHEMA as SELECTION_SCHEMA
from sai.data.pleias_semantic_sample import _token_band
from sai.data.pleias_semantic_stratum_decision import (
    SCHEMA as SEMANTIC_DECISION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-production-materialized-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-production-materialized-aggregate-v1"
DESTINATION_REPOSITORY = "Godlydonuts/Sai"
DESTINATION_PREFIX = "candidates/nontraining/pleias/20260826-r1"


class PleiasProductionMaterializerError(RuntimeError):
    """Selection, full-text replay, benchmark boundary, or remote custody differs."""


def _materialized_schema():
    try:
        import pyarrow as pa
    except ImportError as error:
        raise PleiasProductionMaterializerError("pyarrow is required") from error
    source = _schema()
    fields = [field for field in source if field.name != "training_ready"]
    token_index = next(
        index for index, field in enumerate(fields) if field.name == "token_count"
    ) + 1
    fields[token_index:token_index] = [
        pa.field("semantic_stratum", pa.string()),
        pa.field("semantic_quality_floor_milli", pa.int32()),
        pa.field("semantic_quality_mean_milli", pa.int32()),
        pa.field("semantic_difficulty_mean_milli", pa.int32()),
        pa.field("semantic_prerequisite_burden_mean_milli", pa.int32()),
        pa.field("semantic_curriculum_phase", pa.string()),
        pa.field("semantic_domains", pa.list_(pa.string())),
        pa.field("semantic_recurring_concepts", pa.list_(pa.string())),
        pa.field("semantic_recurring_prerequisites", pa.list_(pa.string())),
    ]
    return pa.schema(fields + [pa.field("training_ready", pa.bool_())])


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasProductionMaterializerError("signed input is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasProductionMaterializerError("signed input is invalid") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasProductionMaterializerError("signed input differs")
    return value


def _selection_database(root: Path) -> tuple[dict[str, Any], Path]:
    receipt = _load_signed(root / "receipt.json", SELECTION_SCHEMA)
    descriptor = receipt.get("selection_database")
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        receipt.get("status") != "complete_nontraining_pleias_production_byte_selection"
        or receipt.get("byte_ceiling_respected") is not True
        or receipt.get("selection_contains_source_text") is not False
        or receipt.get("benchmark_decontamination_complete") is not False
        or not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PleiasProductionMaterializerError("selection database differs")
    return receipt, path


def _semantic_metadata(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_signed(path, SEMANTIC_DECISION_SCHEMA)
    rows = receipt.get("decisions")
    metadata = {}
    if (
        receipt.get("status")
        != "complete_nontraining_pleias_semantic_stratum_decision"
        or not isinstance(rows, list)
    ):
        raise PleiasProductionMaterializerError("semantic decision differs")
    for row in rows:
        if not isinstance(row, dict):
            raise PleiasProductionMaterializerError("semantic stratum differs")
        stratum = row.get("stratum")
        primary = row.get("primary")
        if (
            row.get("row_sha256")
            != canonical_sha256(
                {key: value for key, value in row.items() if key != "row_sha256"}
            )
            or not isinstance(stratum, str)
            or stratum in metadata
            or not isinstance(primary, dict)
        ):
            raise PleiasProductionMaterializerError("semantic stratum differs")
        if row.get("decision") != "advance_to_full_candidate_decontamination":
            continue
        difficulty = primary.get("difficulty_mean_milli")
        burden = primary.get("prerequisite_burden_mean_milli")
        phase = primary.get("dominant_curriculum_phase")
        domains = primary.get("domain_counts")
        concepts = primary.get("recurring_concepts")
        prerequisites = primary.get("recurring_prerequisites")
        if (
            isinstance(difficulty, bool)
            or not isinstance(difficulty, int)
            or not 0 <= difficulty <= 4_000
            or isinstance(burden, bool)
            or not isinstance(burden, int)
            or not 0 <= burden <= 4_000
            or not isinstance(phase, str)
            or not phase
            or not isinstance(domains, dict)
            or not domains
            or not isinstance(concepts, list)
            or not isinstance(prerequisites, list)
            or any(
                not isinstance(value, dict)
                or not isinstance(value.get("concept"), str)
                or not value["concept"]
                or isinstance(value.get("votes"), bool)
                or not isinstance(value.get("votes"), int)
                or value["votes"] < 2
                for value in [*concepts, *prerequisites]
            )
        ):
            raise PleiasProductionMaterializerError(
                "semantic curriculum metadata differs"
            )
        metadata[stratum] = {
            "difficulty_mean_milli": difficulty,
            "prerequisite_burden_mean_milli": burden,
            "dominant_curriculum_phase": phase,
            "domains": sorted(domains),
            "recurring_concepts": [value["concept"] for value in concepts],
            "recurring_prerequisites": [
                value["concept"] for value in prerequisites
            ],
        }
    if set(metadata) != set(receipt.get("advanced_strata", [])):
        raise PleiasProductionMaterializerError("advanced semantic strata differ")
    return metadata, receipt


def replay_selected_row(
    row: dict[str, Any],
    parent: dict[str, Any],
    row_index: int,
    selected: tuple[Any, ...],
    semantic: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct and verify one exact selected candidate from its pinned parent."""

    (
        expected_identity,
        expected_parent_sha256,
        expected_content_sha256,
        expected_stratum,
        expected_bytes,
        expected_tokens,
        expected_quality_floor,
        expected_quality_mean,
    ) = selected
    text = row.get("text")
    identifier = row.get("identifier")
    if not isinstance(text, str) or not isinstance(identifier, str) or not identifier:
        raise PleiasProductionMaterializerError("selected source row differs")
    content = text.encode()
    content_sha256 = hashlib.sha256(content).hexdigest()
    identity = canonical_sha256(
        {
            "source_path": parent["source_path"],
            "row_index": row_index,
            "identifier": identifier,
            "content_sha256": content_sha256,
        }
    )
    stratum = "::".join(
        (row["collection"], row["open_type"], _token_band(row["token_count"]))
    )
    if (
        parent["sha256"] != expected_parent_sha256
        or identity != expected_identity
        or content_sha256 != expected_content_sha256
        or stratum != expected_stratum
        or len(content) != expected_bytes
        or row.get("token_count") != expected_tokens
        or isinstance(expected_quality_floor, bool)
        or not isinstance(expected_quality_floor, int)
        or not 0 <= expected_quality_floor <= 10_000
        or isinstance(expected_quality_mean, bool)
        or not isinstance(expected_quality_mean, int)
        or not expected_quality_floor <= expected_quality_mean <= 10_000
        or not isinstance(semantic, dict)
        or set(semantic)
        != {
            "difficulty_mean_milli",
            "prerequisite_burden_mean_milli",
            "dominant_curriculum_phase",
            "domains",
            "recurring_concepts",
            "recurring_prerequisites",
        }
    ):
        raise PleiasProductionMaterializerError("selected row identity differs")
    return {
        "schema": CANDIDATE_SCHEMA,
        "source_id": "pleias_common_corpus",
        "source_repository": parent["source_repository"],
        "source_revision": parent["source_revision"],
        "source_path": parent["source_path"],
        "source_parent_sha256": parent["sha256"],
        "source_row_index": row_index,
        "source_row_identity_sha256": identity,
        "identifier": identifier,
        "collection": row["collection"],
        "open_type": row["open_type"],
        "license": row["license"],
        "language": row["language"],
        "word_count": row["word_count"],
        "token_count": row["token_count"],
        "semantic_stratum": stratum,
        "semantic_quality_floor_milli": expected_quality_floor,
        "semantic_quality_mean_milli": expected_quality_mean,
        "semantic_difficulty_mean_milli": semantic["difficulty_mean_milli"],
        "semantic_prerequisite_burden_mean_milli": semantic[
            "prerequisite_burden_mean_milli"
        ],
        "semantic_curriculum_phase": semantic["dominant_curriculum_phase"],
        "semantic_domains": semantic["domains"],
        "semantic_recurring_concepts": semantic["recurring_concepts"],
        "semantic_recurring_prerequisites": semantic["recurring_prerequisites"],
        "content_sha256": content_sha256,
        "text": text,
        "training_ready": False,
    }


def _commit_oid(value: Any) -> str:
    for name in ("oid", "commit_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    raise PleiasProductionMaterializerError("Hugging Face commit identity differs")


def upload_verified(
    path: Path,
    remote_path: str,
    token: str,
    repository: str = DESTINATION_REPOSITORY,
    attempts: int = 5,
) -> dict[str, Any]:
    """Upload one LFS payload and replay its exact remote size and SHA-256."""

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise PleiasProductionMaterializerError(
            "huggingface_hub is required"
        ) from error
    if not token or not path.is_file() or attempts <= 0:
        raise PleiasProductionMaterializerError("upload arguments differ")
    size = path.stat().st_size
    digest = sha256_file(path)
    api = HfApi(token=token)
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            commit = api.upload_file(
                path_or_fileobj=path,
                path_in_repo=remote_path,
                repo_id=repository,
                repo_type="dataset",
                commit_message=f"Add verified Sai candidate {Path(remote_path).name}",
            )
            oid = _commit_oid(commit)
            info = api.dataset_info(repository, revision=oid, files_metadata=True)
            sibling = next(
                (item for item in info.siblings or [] if item.rfilename == remote_path),
                None,
            )
            lfs = None if sibling is None else sibling.lfs
            if (
                info.sha != oid
                or sibling is None
                or lfs is None
                or sibling.size != size
                or lfs.size != size
                or lfs.sha256 != digest
            ):
                raise PleiasProductionMaterializerError("remote LFS identity differs")
            return {
                "repository": repository,
                "commit": oid,
                "path": remote_path,
                "bytes": size,
                "sha256": digest,
            }
        except BaseException as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(60, 2**attempt * 5))
    raise PleiasProductionMaterializerError("verified upload failed") from last_error


def run_shard(
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    boundary_roots: list[Path],
    output_root: Path,
    logical_shards: int,
    shard_index: int,
    token: str,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Replay one parent-disjoint selection shard and retain only remote payload."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or not 0 <= shard_index < logical_shards
        or not boundary_roots
    ):
        raise PleiasProductionMaterializerError("materializer arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasProductionMaterializerError("pyarrow is required") from error
    manifest = load_manifest(manifest_path)
    parents = select_shard(manifest, logical_shards, shard_index)
    selection, selection_path = _selection_database(selection_root)
    semantic_by_stratum, semantic_receipt = _semantic_metadata(
        semantic_decision_path
    )
    connection = sqlite3.connect(f"file:{selection_path.resolve()}?mode=ro", uri=True)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    word_boundary = words[0] if len(words) == 1 else _Union(words)
    code_boundary = code[0] if len(code) == 1 else _Union(code)
    output_root.mkdir(parents=True)
    local_path = output_root / "benchmark_disjoint_candidates.parquet"
    temporary = output_root / f".candidates.partial.{uuid.uuid4().hex}.parquet"
    output_schema = _materialized_schema()
    writer = pq.ParquetWriter(temporary, output_schema, compression="zstd")
    counts: Counter[str] = Counter()
    retained_identities = []
    selected_parent_receipts = []
    try:
        for parent_number, parent in enumerate(parents, start=1):
            selected_rows = connection.execute(
                "SELECT source_row_index, source_row_identity_sha256, "
                "source_parent_sha256, content_sha256, stratum, "
                "text_utf8_bytes, token_count, stratum_quality_floor_milli, "
                "stratum_quality_mean_milli FROM selected WHERE source_path=? "
                "ORDER BY source_row_index",
                (parent["source_path"],),
            ).fetchall()
            if not selected_rows:
                counts["parents_without_selected_rows"] += 1
                continue
            by_index = {row[0]: row[1:] for row in selected_rows}
            if len(by_index) != len(selected_rows):
                raise PleiasProductionMaterializerError(
                    "selected parent indices overlap"
                )
            with tempfile.TemporaryDirectory(
                prefix="sai-pleias-production-materializer-", dir=scratch_root
            ) as directory:
                source_path = _download(parent, token, Path(directory))
                parquet = pq.ParquetFile(source_path)
                if not REQUIRED_COLUMNS.issubset(parquet.schema_arrow.names):
                    raise PleiasProductionMaterializerError(
                        "selected parent schema differs"
                    )
                seen = set()
                row_offset = 0
                for batch in parquet.iter_batches(
                    batch_size=32,
                    columns=sorted(REQUIRED_COLUMNS),
                    use_threads=False,
                ):
                    output_rows = []
                    for relative, row in enumerate(batch.to_pylist()):
                        row_index = row_offset + relative
                        expected = by_index.get(row_index)
                        if expected is None:
                            continue
                        candidate = replay_selected_row(
                            row,
                            parent,
                            row_index,
                            expected,
                            semantic_by_stratum.get(expected[3], {}),
                        )
                        seen.add(row_index)
                        counts["selected_rows_replayed"] += 1
                        counts["selected_text_utf8_bytes_replayed"] += len(
                            candidate["text"].encode()
                        )
                        counts["selected_tokens_replayed"] += candidate["token_count"]
                        word_overlap, code_overlap = screen_text(
                            candidate["text"], word_boundary, code_boundary
                        )
                        counts["word_overlap_shingles"] += word_overlap
                        counts["code_overlap_shingles"] += code_overlap
                        if word_overlap or code_overlap:
                            counts["benchmark_contaminated_rows"] += 1
                            continue
                        output_rows.append(candidate)
                        retained_identities.append(
                            candidate["source_row_identity_sha256"]
                        )
                        counts["retained_rows"] += 1
                        counts["retained_text_utf8_bytes"] += len(
                            candidate["text"].encode()
                        )
                        counts["retained_tokens"] += candidate["token_count"]
                    if output_rows:
                        writer.write_table(
                            pa.Table.from_pylist(output_rows, schema=output_schema)
                        )
                    row_offset += batch.num_rows
                if seen != set(by_index):
                    raise PleiasProductionMaterializerError(
                        "selected parent coverage differs"
                    )
            selected_parent_receipts.append(
                {
                    "source_path": parent["source_path"],
                    "source_sha256": parent["sha256"],
                    "selected_rows": len(selected_rows),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "pleias_production_materializer_progress",
                        "shard_index": shard_index,
                        "complete_parents": parent_number,
                        "remaining_parents": len(parents) - parent_number,
                        "retained_rows": counts["retained_rows"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except BaseException:
        writer.close()
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        for boundary in [*words, *code]:
            boundary.close()
    writer.close()
    connection.close()
    os.replace(temporary, local_path)
    remote_path = (
        f"{DESTINATION_PREFIX}/shard-{shard_index:05d}-of-{logical_shards:05d}.parquet"
    )
    remote = upload_verified(local_path, remote_path, token)
    local_path.unlink()
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_production_materialized_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "selection_receipt_sha256": selection["receipt_sha256"],
            "semantic_decision_receipt_sha256": semantic_receipt[
                "receipt_sha256"
            ],
            "selected_parent_count": len(selected_parent_receipts),
            "ordered_selected_parent_receipts_sha256": canonical_sha256(
                selected_parent_receipts
            ),
            "boundary_receipts_sha256": canonical_sha256(
                [row["receipt_sha256"] for row in boundary_receipts]
            ),
        },
        "counts": dict(sorted(counts.items())),
        "ordered_retained_identities_sha256": canonical_sha256(retained_identities),
        "remote_output": remote,
        "local_payload_removed_after_remote_verification": True,
        "full_document_benchmark_decontamination_complete": True,
        "cross_source_near_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


class _Union:
    def __init__(self, members: list[Any]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def aggregate(
    manifest_path: Path,
    selection_root: Path,
    semantic_decision_path: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
    token: str,
) -> dict[str, Any]:
    """Replay every selection identity and remote LFS payload exactly once."""

    if output.exists() or output.is_symlink() or not token or logical_shards <= 0:
        raise PleiasProductionMaterializerError("aggregate arguments differ")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise PleiasProductionMaterializerError(
            "huggingface_hub is required"
        ) from error
    manifest = load_manifest(manifest_path)
    selection, selection_path = _selection_database(selection_root)
    _semantic_by_stratum, semantic_receipt = _semantic_metadata(
        semantic_decision_path
    )
    connection = sqlite3.connect(f"file:{selection_path.resolve()}?mode=ro", uri=True)
    totals: Counter[str] = Counter()
    receipts = []
    remote_by_path = {}
    seen_parents = set()
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        parents = select_shard(manifest, logical_shards, shard_index)
        paths = {row["source_path"] for row in parents}
        expected_rows = 0
        for path in paths:
            expected_rows += connection.execute(
                "SELECT COUNT(*) FROM selected WHERE source_path=?", (path,)
            ).fetchone()[0]
        remote = receipt.get("remote_output")
        if (
            receipt.get("status")
            != "complete_nontraining_pleias_production_materialized_shard"
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("manifest_sha256")
            != sha256_file(manifest_path)
            or receipt.get("source", {}).get("selection_receipt_sha256")
            != selection["receipt_sha256"]
            or receipt.get("source", {}).get("semantic_decision_receipt_sha256")
            != semantic_receipt["receipt_sha256"]
            or receipt.get("counts", {}).get("selected_rows_replayed") != expected_rows
            or receipt.get("counts", {}).get("retained_rows", 0)
            + receipt.get("counts", {}).get("benchmark_contaminated_rows", 0)
            != expected_rows
            or receipt.get("local_payload_removed_after_remote_verification")
            is not True
            or receipt.get("full_document_benchmark_decontamination_complete")
            is not True
            or not isinstance(remote, dict)
            or remote.get("repository") != DESTINATION_REPOSITORY
            or not isinstance(remote.get("path"), str)
            or not remote["path"].startswith(f"{DESTINATION_PREFIX}/")
            or remote["path"] in remote_by_path
            or not isinstance(remote.get("bytes"), int)
            or remote["bytes"] <= 0
            or not isinstance(remote.get("sha256"), str)
            or len(remote["sha256"]) != 64
            or seen_parents.intersection(paths)
        ):
            connection.close()
            raise PleiasProductionMaterializerError("materialized shard differs")
        seen_parents.update(paths)
        remote_by_path[remote["path"]] = remote
        for key, value in receipt["counts"].items():
            totals[key] += value
        totals["remote_output_bytes"] += remote["bytes"]
        receipts.append(receipt["receipt_sha256"])
    selected_rows, selected_bytes, selected_tokens = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(text_utf8_bytes), 0), "
        "COALESCE(SUM(token_count), 0) FROM selected"
    ).fetchone()
    connection.close()
    if (
        seen_parents != {row["source_path"] for row in manifest}
        or totals["selected_rows_replayed"] != selected_rows
        or totals["selected_text_utf8_bytes_replayed"] != selected_bytes
        or totals["selected_tokens_replayed"] != selected_tokens
        or selected_rows != selection.get("counts", {}).get("selected_rows")
        or selected_bytes != selection.get("counts", {}).get("selected_text_utf8_bytes")
        or selected_tokens != selection.get("counts", {}).get("selected_tokens")
    ):
        raise PleiasProductionMaterializerError(
            "materialized aggregate accounting differs"
        )
    api = HfApi(token=token)
    info = api.dataset_info(DESTINATION_REPOSITORY, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings or []}
    for path, expected in remote_by_path.items():
        sibling = siblings.get(path)
        lfs = None if sibling is None else sibling.lfs
        if (
            sibling is None
            or lfs is None
            or sibling.size != expected["bytes"]
            or lfs.size != expected["bytes"]
            or lfs.sha256 != expected["sha256"]
        ):
            raise PleiasProductionMaterializerError(
                "aggregate remote LFS identity differs"
            )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_production_materialized",
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "selection_receipt_sha256": selection["receipt_sha256"],
            "semantic_decision_receipt_sha256": semantic_receipt[
                "receipt_sha256"
            ],
            "ordered_shard_receipts_sha256": canonical_sha256(receipts),
        },
        "shards": {
            "logical_shards": logical_shards,
            "remote_repository": DESTINATION_REPOSITORY,
            "remote_revision_verified": info.sha,
            "remote_prefix": DESTINATION_PREFIX,
        },
        "totals": dict(sorted(totals.items())),
        "complete_selection_identity_coverage": True,
        "complete_source_parent_coverage": True,
        "all_remote_lfs_identities_verified": True,
        "full_document_benchmark_decontamination_complete": True,
        "cross_source_near_deduplication_complete": False,
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
    shard.add_argument("--selection-root", type=Path, required=True)
    shard.add_argument("--semantic-decision", type=Path, required=True)
    shard.add_argument("--boundary-index", type=Path, action="append", required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    shard.add_argument("--scratch-root", type=Path)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--manifest", type=Path, required=True)
    combine.add_argument("--selection-root", type=Path, required=True)
    combine.add_argument("--semantic-decision", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.selection_root,
            args.semantic_decision,
            args.boundary_index,
            args.output_root,
            args.logical_shards,
            args.shard_index,
            os.environ.get(args.token_env, ""),
            args.scratch_root,
        )
    else:
        result = aggregate(
            args.manifest,
            args.selection_root,
            args.semantic_decision,
            args.shards_root,
            args.logical_shards,
            args.output,
            os.environ.get(args.token_env, ""),
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
