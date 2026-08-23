"""Build a larger identity-disjoint confirmation of selected Common Pile lanes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.common_pile_audit_population import (
    COMMON_PILE_SOURCE_TYPES,
    CommonPileAuditError,
    sample_verified_gzip_parent,
)
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.reservoir_audit_confirmation_plan import (
    METHOD,
)
from sai.data.reservoir_audit_confirmation_plan import (
    SCHEMA as PLAN_SCHEMA,
)
from sai.data.reservoir_audit_population import (
    SCHEMA,
    ReservoirAuditError,
    _candidate_and_lineage,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SEED = 20260825
ROWS_PER_SOURCE = METHOD["confirmation_rows_per_source"]


class CommonPileConfirmationError(RuntimeError):
    """The plan, source-disjoint geometry, bytes, or population differs."""


def _load_plan(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 8 << 20
    ):
        raise CommonPileConfirmationError("confirmation plan is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CommonPileConfirmationError(
            "confirmation plan cannot be decoded"
        ) from error
    if not isinstance(payload, dict):
        raise CommonPileConfirmationError("confirmation plan differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    selected = payload.get("selected_source_ids")
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("method") != METHOD
        or payload.get("method_sha256") != canonical_sha256(METHOD)
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or not isinstance(selected, list)
        or not selected
        or selected != sorted(set(selected))
        or any(
            not isinstance(source_id, str) or not source_id.startswith("common_pile_")
            for source_id in selected
        )
        or payload.get("selected_sources") != len(selected)
        or payload.get("confirmation_rows_per_source") != ROWS_PER_SOURCE
        or payload.get("target_confirmation_rows") != len(selected) * ROWS_PER_SOURCE
        or payload.get("bulk_training_admission") is not False
        or payload.get("training_ready") is not False
    ):
        raise CommonPileConfirmationError("confirmation plan differs")
    return payload


def _parent_selection_key(row: dict[str, Any], plan_sha256: str) -> str:
    return canonical_sha256(
        {
            "seed": SEED,
            "plan_sha256": plan_sha256,
            "source_id": row["source_id"],
            "repository": row["repository"],
            "path": row["path"],
            "sha256": row["sha256"],
        }
    )


def build_confirmation_parent_plan(
    rows: list[dict[str, Any]],
    selected_source_ids: list[str],
    discovery_parent_paths: dict[str, set[str]],
    *,
    plan_sha256: str,
) -> list[dict[str, Any]]:
    """Prefer a new exact parent, otherwise reuse one with row exclusions."""

    plan = []
    for source_id in selected_source_ids:
        component = source_id.removeprefix("common_pile_")
        if component not in COMMON_PILE_SOURCE_TYPES:
            raise CommonPileConfirmationError("selected Common Pile source differs")
        matches = sorted(
            (row for row in rows if row["source_id"] == source_id),
            key=lambda row: (row["physical_bytes"], row["path"], row["sha256"]),
        )
        prior_paths = discovery_parent_paths.get(source_id)
        if not matches or not prior_paths or len(prior_paths) != 1:
            raise CommonPileConfirmationError("discovery parent geometry differs")
        alternatives = [row for row in matches if row["path"] not in prior_paths]
        selected = alternatives[0] if alternatives else matches[0]
        parent_disjoint = selected["path"] not in prior_paths
        plan.append(
            {
                "source_id": selected["source_id"],
                "stratum": selected["epistemic_function"],
                "source_type": COMMON_PILE_SOURCE_TYPES[component],
                "repository": selected["repository"],
                "revision": selected["revision"],
                "license": selected["license"],
                "access": selected["access"],
                "path": selected["path"],
                "parent_file_bytes": selected["physical_bytes"],
                "parent_file_sha256": selected["sha256"],
                "text_column": selected["text_column"],
                "parent_selection_key": _parent_selection_key(selected, plan_sha256),
                "parent_disjoint_from_discovery": parent_disjoint,
            }
        )
    if len(plan) != len(selected_source_ids) or len(
        {(row["repository"], row["path"]) for row in plan}
    ) != len(plan):
        raise CommonPileConfirmationError("confirmation parent geometry differs")
    return plan


def download_and_sample_parent(
    parent: dict[str, Any],
    token: str,
    excluded_line_numbers: frozenset[int],
    excluded_text_sha256s: frozenset[str],
) -> list[dict[str, Any]]:
    """Verify one parent and select exact rows outside the discovery identities."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise CommonPileConfirmationError("huggingface_hub is required") from error
    with tempfile.TemporaryDirectory(prefix="sai-common-confirmation-") as temporary:
        try:
            path = Path(
                hf_hub_download(
                    parent["repository"],
                    parent["path"],
                    repo_type="dataset",
                    revision=parent["revision"],
                    token=token,
                    local_dir=temporary,
                )
            )
            return sample_verified_gzip_parent(
                path,
                parent,
                rows_per_source=ROWS_PER_SOURCE,
                excluded_line_numbers=excluded_line_numbers,
                excluded_text_sha256s=excluded_text_sha256s,
            )
        except CommonPileAuditError as error:
            raise CommonPileConfirmationError(
                "Common Pile confirmation source differs"
            ) from error
        except Exception as error:
            raise CommonPileConfirmationError(
                "Common Pile confirmation download failed"
            ) from error


def build_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    discovery_root: Path,
    plan_path: Path,
    output_root: Path,
    *,
    token: str,
    acquire_function: Callable[
        [dict[str, Any], str, frozenset[int], frozenset[str]],
        list[dict[str, Any]],
    ] = download_and_sample_parent,
) -> dict[str, Any]:
    """Acquire and seal a larger, exact-identity-disjoint confirmation."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise CommonPileConfirmationError(
            "confirmation credential or output boundary differs"
        )
    plan_receipt = _load_plan(plan_path)
    discovery_candidates, discovery_lineage, discovery_receipt = load_population(
        discovery_root
    )
    selected = plan_receipt["selected_source_ids"]
    discovery_parent_paths: dict[str, set[str]] = defaultdict(set)
    discovery_line_numbers: dict[str, set[int]] = defaultdict(set)
    discovery_text_sha256s: dict[str, set[str]] = defaultdict(set)
    for candidate, lineage in zip(discovery_candidates, discovery_lineage, strict=True):
        source_id = lineage["source_id"]
        if source_id not in selected:
            continue
        discovery_parent_paths[source_id].add(lineage["path"])
        line_number = lineage.get("locator", {}).get("line_number")
        if isinstance(line_number, int) and not isinstance(line_number, bool):
            discovery_line_numbers[source_id].add(line_number)
        discovery_text_sha256s[source_id].add(candidate["source_content_sha256"])
    rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    parent_plan = build_confirmation_parent_plan(
        rows,
        selected,
        dict(discovery_parent_paths),
        plan_sha256=plan_receipt["receipt_sha256"],
    )
    candidates = []
    lineage_rows = []
    for parent_index, parent in enumerate(parent_plan, start=1):
        source_id = parent["source_id"]
        excluded_lines = (
            frozenset()
            if parent["parent_disjoint_from_discovery"]
            else frozenset(discovery_line_numbers[source_id])
        )
        acquired_rows = acquire_function(
            parent,
            token,
            excluded_lines,
            frozenset(discovery_text_sha256s[source_id]),
        )
        if len(acquired_rows) != ROWS_PER_SOURCE:
            raise CommonPileConfirmationError("confirmation row count differs")
        for acquired in acquired_rows:
            row_plan = {
                **parent,
                "ordinal": len(candidates),
                "license": acquired["declared_license"],
                "selection_key": acquired["row_selection_key"],
            }
            try:
                candidate, source_lineage = _candidate_and_lineage(row_plan, acquired)
            except ReservoirAuditError as error:
                raise CommonPileConfirmationError(
                    "confirmation candidate differs"
                ) from error
            source_lineage["manifest_license"] = parent["license"]
            source_lineage["declared_license"] = acquired["declared_license"]
            source_lineage["confirmation_plan_receipt_sha256"] = plan_receipt[
                "receipt_sha256"
            ]
            source_lineage["identity_disjoint_from_discovery"] = True
            source_lineage["parent_disjoint_from_discovery"] = parent[
                "parent_disjoint_from_discovery"
            ]
            source_lineage.pop("lineage_sha256")
            source_lineage["lineage_sha256"] = canonical_sha256(source_lineage)
            candidates.append(candidate)
            lineage_rows.append(source_lineage)
        print(
            json.dumps(
                {
                    "event": "common_pile_confirmation_acquisition_progress",
                    "parents_acquired": parent_index,
                    "parents_remaining": len(parent_plan) - parent_index,
                    "rows_acquired": len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    identities = [row["candidate_identity_sha256"] for row in candidates]
    content = [row["source_content_sha256"] for row in candidates]
    discovery_identities = {
        row["candidate_identity_sha256"] for row in discovery_candidates
    }
    discovery_content = {row["source_content_sha256"] for row in discovery_candidates}
    if (
        len(candidates) != plan_receipt["target_confirmation_rows"]
        or len(identities) != len(set(identities))
        or len(content) != len(set(content))
        or set(identities).intersection(discovery_identities)
        or set(content).intersection(discovery_content)
    ):
        raise CommonPileConfirmationError("confirmation identity custody differs")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise CommonPileConfirmationError("confirmation temporary output exists")
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage_rows)
        by_source = Counter(row["source_id"] for row in lineage_rows)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "confirmation_plan_then_different_parent_when_available_else_"
                "exact_discovery_row_and_content_exclusion_then_bottom_k"
            ),
            "statistically_representative": False,
            "plan": {
                "path": plan_path.name,
                "bytes": plan_path.stat().st_size,
                "sha256": sha256_file(plan_path),
                "receipt_sha256": plan_receipt["receipt_sha256"],
            },
            "discovery": {
                "root_name": discovery_root.name,
                "receipt_sha256": discovery_receipt["receipt_sha256"],
                "population_file_sha256": sha256_file(
                    discovery_root / "candidates.jsonl"
                ),
                "lineage_file_sha256": sha256_file(discovery_root / "lineage.jsonl"),
            },
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
                "rows": len(lineage_rows),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage_rows),
            },
            "by_source": dict(sorted(by_source.items())),
            "selected_sources": len(parent_plan),
            "rows_per_source": ROWS_PER_SOURCE,
            "parent_disjoint_sources": sum(
                row["parent_disjoint_from_discovery"] for row in parent_plan
            ),
            "parent_reused_with_row_exclusion_sources": sum(
                not row["parent_disjoint_from_discovery"] for row in parent_plan
            ),
            "identity_disjoint_from_discovery": True,
            "exact_content_disjoint_from_discovery": True,
            "fully_verified_parent_files": len(parent_plan),
            "fully_verified_compressed_parent_bytes": sum(
                row["parent_file_bytes"] for row in parent_plan
            ),
            "maximum_simultaneous_parent_files": 1,
            "raw_source_rows_are_training_ready": False,
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
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = build_population(
        args.manifest,
        args.reservoir_receipt,
        args.discovery_root,
        args.plan,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
