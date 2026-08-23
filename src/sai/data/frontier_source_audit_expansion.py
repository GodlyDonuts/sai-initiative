"""Acquire a metadata-aware screen of newly added frontier source families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import AgentLabelingError
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.reservoir_audit_population import (
    SCHEMA,
    ReservoirAuditError,
    _candidate_and_lineage,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SEED = 20260824


@dataclass(frozen=True)
class AuditStratum:
    source_id: str
    stratum: str
    path_prefix: str
    quota: int
    source_type: str


AUDIT_STRATA = (
    *(
        AuditStratum(
            "pleias_common_corpus",
            f"open_corpus_partition:{partition}",
            f"common_corpus_{partition}/",
            4,
            "reference",
        )
        for partition in range(1, 11)
    ),
    AuditStratum(
        "nemotron_specialized_v1_2",
        "fact_seeking",
        "Nemotron-Pretraining-Fact-Seeking/",
        20,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_specialized_v1_2",
        "generative",
        "Nemotron-Pretraining-Generative/",
        1,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_specialized_v1_2",
        "moral_scenarios",
        "Nemotron-Pretraining-Moral-Scenarios/",
        1,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_specialized_v1_2",
        "multiple_choice",
        "Nemotron-Pretraining-Multiple-Choice/",
        8,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_legal_v1",
        "legal_all_published_parents",
        "Nemotron-Pretraining-Legal-",
        21,
        "reference",
    ),
)

EXPECTED_ROWS = sum(stratum.quota for stratum in AUDIT_STRATA)
if EXPECTED_ROWS != 91:  # pragma: no cover - frozen screen geometry
    raise RuntimeError("frontier expansion screen geometry differs")


class FrontierSourceAuditExpansionError(RuntimeError):
    """The frontier expansion identity, metadata, or screen differs."""


def _selection_key(stratum: AuditStratum, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{SEED}:{stratum.source_id}:{stratum.stratum}:"
            f"{row['repository']}:{row['path']}:{row['sha256']}"
        ).encode()
    ).hexdigest()


def build_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the smallest immutable parents in every new-source stratum."""

    plan = []
    for stratum in AUDIT_STRATA:
        matches = [
            row
            for row in rows
            if row["source_id"] == stratum.source_id
            and row["path"].startswith(stratum.path_prefix)
        ]
        ranked = sorted(
            matches,
            key=lambda row: (
                row["physical_bytes"],
                row["path"],
                row["sha256"],
            ),
        )
        if len(ranked) < stratum.quota:
            raise FrontierSourceAuditExpansionError(
                f"frontier expansion stratum is underfilled: {stratum.stratum}"
            )
        for row in ranked[: stratum.quota]:
            plan.append(
                {
                    "ordinal": len(plan),
                    "source_id": row["source_id"],
                    "stratum": stratum.stratum,
                    "source_type": stratum.source_type,
                    "repository": row["repository"],
                    "revision": row["revision"],
                    "license": row["license"],
                    "access": row["access"],
                    "path": row["path"],
                    "parent_file_bytes": row["physical_bytes"],
                    "parent_file_sha256": row["sha256"],
                    "text_column": row["text_column"],
                    "selection_key": _selection_key(stratum, row),
                }
            )
    parents = [(row["repository"], row["path"]) for row in plan]
    if len(plan) != EXPECTED_ROWS or len(parents) != len(set(parents)):
        raise FrontierSourceAuditExpansionError(
            "frontier expansion plan identity differs"
        )
    return plan


def _as_py(table: Any, column: str, row_index: int) -> Any:
    if column not in table.column_names:
        return None
    return table[column][row_index].as_py()


def acquire_metadata_row(plan: dict[str, Any], token: str) -> dict[str, Any]:
    """Range-read one source row and bind rights/language metadata when present."""

    try:
        import fsspec
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_url
    except ImportError as error:
        raise FrontierSourceAuditExpansionError(
            "fsspec, pyarrow, and huggingface_hub are required"
        ) from error
    url = hf_hub_url(
        plan["repository"],
        plan["path"],
        repo_type="dataset",
        revision=plan["revision"],
    )
    filesystem = fsspec.filesystem("http", headers={"Authorization": f"Bearer {token}"})
    try:
        with filesystem.open(url, "rb", block_size=8 << 20) as handle:
            if handle.size != plan["parent_file_bytes"]:
                raise FrontierSourceAuditExpansionError(
                    "frontier expansion parent size differs"
                )
            source = parquet.ParquetFile(handle)
            available = set(source.schema_arrow.names)
            if "text" not in available or source.metadata.num_row_groups <= 0:
                raise FrontierSourceAuditExpansionError(
                    "frontier expansion Parquet schema differs"
                )
            metadata_columns = (
                ("identifier", "collection", "open_type", "license", "language")
                if plan["source_id"] == "pleias_common_corpus"
                else ("license", "metadata", "uuid")
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
            table = source.read_row_group(
                group_index, columns=columns, use_threads=False
            )
            if table.num_rows <= 0:
                raise FrontierSourceAuditExpansionError(
                    "frontier expansion row group is empty"
                )
            start = (
                int(
                    hashlib.sha256(f"{plan['selection_key']}:row".encode()).hexdigest(),
                    16,
                )
                % table.num_rows
            )
            for offset in range(table.num_rows):
                row_index = (start + offset) % table.num_rows
                text = _as_py(table, "text", row_index)
                if not isinstance(text, str) or len(text.strip().encode()) < 200:
                    continue
                license_name = _as_py(table, "license", row_index)
                if not isinstance(license_name, str) or not license_name.strip():
                    license_name = plan["license"]
                metadata = _as_py(table, "metadata", row_index)
                native_id = _as_py(table, "identifier", row_index)
                if native_id is None:
                    native_id = _as_py(table, "uuid", row_index)
                if native_id is not None and not isinstance(native_id, str):
                    native_id = canonical_sha256(native_id)
                locator = {
                    "format": "parquet",
                    "row_group": group_index,
                    "row_in_group": row_index,
                    "row_index": sum(
                        source.metadata.row_group(index).num_rows
                        for index in range(group_index)
                    )
                    + row_index,
                    "native_id": native_id,
                    "language": _as_py(table, "language", row_index),
                    "collection": _as_py(table, "collection", row_index),
                    "open_type": _as_py(table, "open_type", row_index),
                    "metadata_sha256": canonical_sha256(
                        metadata if isinstance(metadata, dict) else {}
                    ),
                }
                return {
                    "text": text.strip(),
                    "locator": locator,
                    "declared_license": license_name.strip(),
                    "full_file_content_verified": False,
                }
    except FrontierSourceAuditExpansionError:
        raise
    except Exception as error:
        raise FrontierSourceAuditExpansionError(
            "frontier expansion acquisition failed"
        ) from error
    raise FrontierSourceAuditExpansionError(
        "frontier expansion parent has no usable text"
    )


def build_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    output_root: Path,
    *,
    token: str,
) -> dict[str, Any]:
    """Acquire and seal the source-discovery screen without admitting any row."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise FrontierSourceAuditExpansionError(
            "frontier expansion credential or output boundary differs"
        )
    rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    plan = build_plan(rows)
    candidates = []
    lineage = []
    for index, item in enumerate(plan, start=1):
        acquired = acquire_metadata_row(item, token)
        row_plan = {**item, "license": acquired["declared_license"]}
        try:
            candidate, source_lineage = _candidate_and_lineage(row_plan, acquired)
        except (ReservoirAuditError, AgentLabelingError) as error:
            raise FrontierSourceAuditExpansionError(
                "frontier expansion candidate differs"
            ) from error
        source_lineage["manifest_license"] = item["license"]
        source_lineage["declared_license"] = acquired["declared_license"]
        source_lineage.pop("lineage_sha256")
        source_lineage["lineage_sha256"] = canonical_sha256(source_lineage)
        candidates.append(candidate)
        lineage.append(source_lineage)
        if index % 8 == 0 or index == len(plan):
            print(
                json.dumps(
                    {
                        "event": "frontier_expansion_audit_progress",
                        "acquired": index,
                        "remaining": len(plan) - index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(candidates) != EXPECTED_ROWS or len(identities) != len(set(identities)):
        raise FrontierSourceAuditExpansionError(
            "frontier expansion candidate identities differ"
        )
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise FrontierSourceAuditExpansionError(
            "frontier expansion temporary output exists"
        )
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        by_source = Counter(row["source_id"] for row in lineage)
        by_stratum = Counter(f"{row['source_id']}::{row['stratum']}" for row in lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "smallest_exact_parents_per_new_source_stratum_then_"
                "deterministic_usable_row"
            ),
            "screen_only": True,
            "statistically_representative": False,
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_sha256": sha256_file(reservoir_receipt_path),
                "selected_files": len(rows),
                "selected_bytes": sum(row["physical_bytes"] for row in rows),
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
            "by_source": dict(sorted(by_source.items())),
            "by_stratum": dict(sorted(by_stratum.items())),
            "range_read_parent_files": len(plan),
            "fully_verified_parent_files": 0,
            "hermes_judgments_complete": False,
            "training_ready": False,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    payload = build_population(
        args.manifest,
        args.reservoir_receipt,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
