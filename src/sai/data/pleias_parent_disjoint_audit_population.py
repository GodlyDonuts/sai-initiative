"""Acquire a larger parent-disjoint, partition-stratified PleIAs audit screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import AgentLabelingError
from sai.data.frontier_source_audit_expansion import _as_py, acquire_metadata_row
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.reservoir_audit_population import (
    SCHEMA,
    ReservoirAuditError,
    _candidate_and_lineage,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SEED = 20260826
SOURCE_ID = "pleias_common_corpus"
PARTITION_QUOTAS = {
    partition: 103 if partition <= 4 else 102 for partition in range(1, 11)
}
EXPECTED_ROWS = sum(PARTITION_QUOTAS.values())
ACQUISITION_MODES = {"range", "full_verified_parent"}


class PleiasParentDisjointAuditError(RuntimeError):
    """The PleIAs population identity, disjointness, or acquisition differs."""


def prior_parent_identities(path: Path) -> frozenset[tuple[str, str, str]]:
    """Load exact parents used by an earlier source-safe audit population."""

    if not path.is_file() or path.is_symlink():
        raise PleiasParentDisjointAuditError("prior lineage boundary differs")
    identities: set[tuple[str, str, str]] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("source_id") != SOURCE_ID:
                    continue
                identity = (row["repository"], row["revision"], row["path"])
            except Exception as error:
                raise PleiasParentDisjointAuditError(
                    f"prior lineage row {line_number} differs"
                ) from error
            identities.add(identity)
    if not identities:
        raise PleiasParentDisjointAuditError("prior PleIAs lineage is empty")
    return frozenset(identities)


def _partition(path: str) -> int | None:
    for partition in PARTITION_QUOTAS:
        if path.startswith(f"common_corpus_{partition}/"):
            return partition
    return None


def _rank(row: dict[str, Any], partition: int) -> str:
    return hashlib.sha256(
        (
            f"{SEED}:{SOURCE_ID}:{partition}:{row['repository']}:"
            f"{row['revision']}:{row['path']}:{row['sha256']}"
        ).encode()
    ).hexdigest()


def build_plan(
    rows: list[dict[str, Any]],
    excluded_parents: frozenset[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Hash-rank exact parents within every corpus partition."""

    plan: list[dict[str, Any]] = []
    for partition, quota in PARTITION_QUOTAS.items():
        candidates = []
        for row in rows:
            identity = (row["repository"], row["revision"], row["path"])
            if (
                row["source_id"] == SOURCE_ID
                and _partition(row["path"]) == partition
                and identity not in excluded_parents
            ):
                candidates.append((_rank(row, partition), row))
        ranked = sorted(candidates, key=lambda item: (item[0], item[1]["path"]))
        if len(ranked) < quota:
            raise PleiasParentDisjointAuditError(
                f"PleIAs partition {partition} is underfilled"
            )
        for selection_key, row in ranked[:quota]:
            plan.append(
                {
                    "ordinal": len(plan),
                    "source_id": SOURCE_ID,
                    "stratum": f"open_corpus_partition:{partition}",
                    "source_type": "reference",
                    "repository": row["repository"],
                    "revision": row["revision"],
                    "license": row["license"],
                    "access": row["access"],
                    "path": row["path"],
                    "parent_file_bytes": row["physical_bytes"],
                    "parent_file_sha256": row["sha256"],
                    "text_column": row["text_column"],
                    "selection_key": selection_key,
                }
            )
    identities = [(row["repository"], row["revision"], row["path"]) for row in plan]
    if (
        len(plan) != EXPECTED_ROWS
        or len(identities) != len(set(identities))
        or any(identity in excluded_parents for identity in identities)
    ):
        raise PleiasParentDisjointAuditError("PleIAs plan custody differs")
    return plan


def shard_plan(
    plan: list[dict[str, Any]], logical_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    """Return one exact identity-disjoint shard of the frozen global plan."""

    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 32
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < logical_shards
        or len(plan) != EXPECTED_ROWS
    ):
        raise PleiasParentDisjointAuditError("PleIAs shard geometry differs")
    selected = [row for row in plan if row["ordinal"] % logical_shards == shard_index]
    expected = (len(plan) + logical_shards - 1 - shard_index) // logical_shards
    if len(selected) != expected:
        raise PleiasParentDisjointAuditError("PleIAs shard custody differs")
    return selected


def acquire_full_verified_metadata_row(
    plan: dict[str, Any], token: str
) -> dict[str, Any]:
    """Download, hash-verify, read, and remove exactly one parent at a time."""

    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise PleiasParentDisjointAuditError(
            "pyarrow and huggingface_hub are required"
        ) from error
    try:
        with tempfile.TemporaryDirectory(prefix="sai-pleias-parent-") as temporary:
            temporary_path = Path(temporary)
            downloaded = Path(
                hf_hub_download(
                    repo_id=plan["repository"],
                    filename=plan["path"],
                    repo_type="dataset",
                    revision=plan["revision"],
                    token=token,
                    cache_dir=temporary_path / "cache",
                    local_dir=temporary_path / "local",
                )
            )
            if (
                not downloaded.is_file()
                or downloaded.stat().st_size != plan["parent_file_bytes"]
                or sha256_file(downloaded) != plan["parent_file_sha256"]
            ):
                raise PleiasParentDisjointAuditError(
                    "PleIAs downloaded parent identity differs"
                )
            source = parquet.ParquetFile(downloaded)
            available = set(source.schema_arrow.names)
            if "text" not in available or source.metadata.num_row_groups <= 0:
                raise PleiasParentDisjointAuditError(
                    "PleIAs downloaded parent schema differs"
                )
            metadata_columns = (
                "identifier",
                "collection",
                "open_type",
                "license",
                "language",
            )
            columns = [
                "text",
                *(name for name in metadata_columns if name in available),
            ]
            group_index = (
                int(
                    hashlib.sha256(
                        f"{plan['selection_key']}:row-group".encode()
                    ).hexdigest(),
                    16,
                )
                % source.metadata.num_row_groups
            )
            row_group_rows = source.metadata.row_group(group_index).num_rows
            if row_group_rows <= 0:
                raise PleiasParentDisjointAuditError(
                    "PleIAs downloaded parent row group is empty"
                )
            start = (
                int(
                    hashlib.sha256(
                        f"{plan['selection_key']}:row".encode()
                    ).hexdigest(),
                    16,
                )
                % row_group_rows
            )
            selected = None
            selected_distance = row_group_rows + 1
            cursor = 0
            for batch in source.iter_batches(
                batch_size=16,
                row_groups=[group_index],
                columns=columns,
                use_threads=False,
            ):
                for row_in_batch in range(batch.num_rows):
                    row_index = cursor + row_in_batch
                    text = _as_py(batch, "text", row_in_batch)
                    if not isinstance(text, str) or len(text.strip().encode()) < 200:
                        continue
                    distance = (row_index - start) % row_group_rows
                    if distance >= selected_distance:
                        continue
                    license_name = _as_py(batch, "license", row_in_batch)
                    if not isinstance(license_name, str) or not license_name.strip():
                        license_name = plan["license"]
                    native_id = _as_py(batch, "identifier", row_in_batch)
                    if native_id is not None and not isinstance(native_id, str):
                        native_id = canonical_sha256(native_id)
                    selected = {
                        "text": text.strip(),
                        "locator": {
                            "format": "parquet",
                            "row_group": group_index,
                            "row_in_group": row_index,
                            "row_index": sum(
                                source.metadata.row_group(index).num_rows
                                for index in range(group_index)
                            )
                            + row_index,
                            "native_id": native_id,
                            "language": _as_py(batch, "language", row_in_batch),
                            "collection": _as_py(
                                batch, "collection", row_in_batch
                            ),
                            "open_type": _as_py(batch, "open_type", row_in_batch),
                            "metadata_sha256": canonical_sha256({}),
                        },
                        "declared_license": license_name.strip(),
                        "full_file_content_verified": True,
                    }
                    selected_distance = distance
                cursor += batch.num_rows
            if cursor != row_group_rows:
                raise PleiasParentDisjointAuditError(
                    "PleIAs row-group batch custody differs"
                )
            if selected is not None:
                return selected
    except PleiasParentDisjointAuditError:
        raise
    except Exception as error:
        raise PleiasParentDisjointAuditError(
            "PleIAs full-parent acquisition failed"
        ) from error
    raise PleiasParentDisjointAuditError(
        "PleIAs downloaded parent has no usable text"
    )


def build_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    prior_lineage_path: Path,
    output_root: Path,
    *,
    token: str,
    acquisition_mode: str = "range",
    logical_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Range-read one deterministic usable row from every selected parent."""

    if (
        not token
        or acquisition_mode not in ACQUISITION_MODES
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise PleiasParentDisjointAuditError("credential or output boundary differs")
    rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    excluded = prior_parent_identities(prior_lineage_path)
    global_plan = build_plan(rows, excluded)
    plan = shard_plan(global_plan, logical_shards, shard_index)
    candidates = []
    lineage = []
    for index, item in enumerate(plan, start=1):
        acquired = (
            acquire_metadata_row(item, token)
            if acquisition_mode == "range"
            else acquire_full_verified_metadata_row(item, token)
        )
        row_plan = {**item, "license": acquired["declared_license"]}
        try:
            candidate, source_lineage = _candidate_and_lineage(row_plan, acquired)
        except (ReservoirAuditError, AgentLabelingError) as error:
            raise PleiasParentDisjointAuditError("PleIAs candidate differs") from error
        source_lineage["manifest_license"] = item["license"]
        source_lineage["declared_license"] = acquired["declared_license"]
        source_lineage.pop("lineage_sha256")
        source_lineage["lineage_sha256"] = canonical_sha256(source_lineage)
        candidates.append(candidate)
        lineage.append(source_lineage)
        if index % 16 == 0 or index == len(plan):
            print(
                json.dumps(
                    {
                        "event": "pleias_parent_disjoint_audit_progress",
                        "logical_shards": logical_shards,
                        "shard_index": shard_index,
                        "acquired": index,
                        "remaining": len(plan) - index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(identities) != len(plan) or len(identities) != len(set(identities)):
        raise PleiasParentDisjointAuditError("PleIAs candidate identities differ")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise PleiasParentDisjointAuditError("temporary output boundary differs")
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        by_stratum = Counter(row["stratum"] for row in lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": "sha256_ranked_exact_parents_within_partition",
            "acquisition_mode": acquisition_mode,
            "global_plan_rows": len(global_plan),
            "logical_shards": logical_shards,
            "shard_index": shard_index,
            "maximum_simultaneous_parent_files": 1,
            "temporary_parent_removed_after_each_row": True,
            "screen_only": True,
            "statistically_representative": False,
            "source_id": SOURCE_ID,
            "prior_population": {
                "lineage_sha256": sha256_file(prior_lineage_path),
                "excluded_parent_files": len(excluded),
                "source_parent_disjoint": True,
            },
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_file_sha256": sha256_file(reservoir_receipt_path),
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
            "by_stratum": dict(sorted(by_stratum.items())),
            "range_read_parent_files": len(plan),
            "fully_verified_parent_files": (
                len(plan) if acquisition_mode == "full_verified_parent" else 0
            ),
            "benchmark_decontamination_complete": False,
            "hermes_judgments_complete": False,
            "source_wide_yield_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_jsonl(receipt_path, [receipt])
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservoir-receipt", type=Path, required=True)
    parser.add_argument("--prior-lineage", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument(
        "--acquisition-mode", choices=sorted(ACQUISITION_MODES), default="range"
    )
    parser.add_argument("--logical-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    result = build_population(
        args.manifest,
        args.reservoir_receipt,
        args.prior_lineage,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
        acquisition_mode=args.acquisition_mode,
        logical_shards=args.logical_shards,
        shard_index=args.shard_index,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
