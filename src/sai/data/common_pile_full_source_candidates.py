"""Materialize one promoted Common Pile parent into filtered full-source candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.attribution_manifest import build_manifest as build_attribution_manifest
from sai.data.common_pile_full_source_promotion import SCHEMA as PROMOTION_SCHEMA
from sai.data.common_pile_streaming_pilot import (
    audit_exclusions,
    download_parent,
    select_bottom_k,
    write_raw_population,
)
from sai.data.contextless_answer_key_filter import build as build_answer_key_filter
from sai.data.data_yield_ledger import _load_receipt
from sai.data.decontamination import build as build_decontaminated
from sai.data.external_exact_deduplication import build_exact_deduplication
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-full-source-candidates-v1"
MAXIMUM_SOURCE_ROWS = 100_000


class CommonPileFullSourceCandidatesError(RuntimeError):
    """The promotion, full-parent coverage, filter, or output differs."""


def load_promotion(path: Path, source_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require one exact per-source candidate-materialization authorization."""

    try:
        payload = _load_receipt(path)
    except Exception as error:  # noqa: BLE001 - normalize receipt loader failures
        raise CommonPileFullSourceCandidatesError(
            "promotion receipt differs"
        ) from error
    sources = payload.get("sources")
    matches = (
        [row for row in sources if row.get("source_id") == source_id]
        if isinstance(sources, list)
        else []
    )
    if (
        payload.get("schema") != PROMOTION_SCHEMA
        or payload.get("status") != "complete_candidate_only_source_decision"
        or source_id not in payload.get("authorized_source_ids", [])
        or len(matches) != 1
        or matches[0].get("full_source_candidate_materialization_authorized")
        is not True
        or matches[0].get("bulk_training_admission") is not False
        or matches[0].get("training_ready") is not False
        or payload.get("full_source_materialization_is_training_admission")
        is not False
        or payload.get("rights_provenance_verified") is not False
        or payload.get("legal_clearance_established") is not False
        or payload.get("training_ready") is not False
        or payload.get("four_b_training_authorized") is not False
    ):
        raise CommonPileFullSourceCandidatesError("source is not promoted")
    parent = matches[0].get("parent")
    if (
        not isinstance(parent, dict)
        or parent.get("source_id") != source_id
        or not isinstance(parent.get("repository"), str)
        or not isinstance(parent.get("revision"), str)
        or not isinstance(parent.get("path"), str)
        or isinstance(parent.get("bytes"), bool)
        or not isinstance(parent.get("bytes"), int)
        or parent["bytes"] <= 0
        or not isinstance(parent.get("sha256"), str)
        or len(parent["sha256"]) != 64
        or not isinstance(parent.get("manifest_license"), str)
        or not isinstance(parent.get("domain"), str)
    ):
        raise CommonPileFullSourceCandidatesError("promoted parent differs")
    return payload, matches[0]


def build_candidates(
    promotion_path: Path,
    audit_roots: list[Path],
    boundary_roots: list[Path],
    output_root: Path,
    *,
    source_id: str,
    token: str,
    decontamination_workers: int = 1,
) -> dict[str, Any]:
    """Replay every eligible parent row through deterministic candidate filters."""

    if (
        not token
        or output_root.exists()
        or output_root.is_symlink()
        or isinstance(decontamination_workers, bool)
        or not 1 <= decontamination_workers <= 64
    ):
        raise CommonPileFullSourceCandidatesError("candidate output boundary differs")
    promotion, decision = load_promotion(promotion_path, source_id)
    parent = decision["parent"]
    excluded_lines, excluded_content, audit_receipts = audit_exclusions(
        audit_roots, source_id
    )
    output_root.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-common-pile-full-source-"
        ) as temporary:
            compressed = download_parent(parent, token, Path(temporary))
            selected, scan = select_bottom_k(
                compressed,
                parent,
                maximum_rows=MAXIMUM_SOURCE_ROWS,
                excluded_lines=excluded_lines.get(
                    (parent["repository"], parent["path"]), frozenset()
                ),
                excluded_content_sha256s=excluded_content,
            )
            if scan.get("eligible_rows") != scan.get("selected_rows"):
                raise CommonPileFullSourceCandidatesError(
                    "promoted parent exceeds the full-source row bound"
                )
            raw_path = output_root / "raw_candidates.jsonl"
            raw = write_raw_population(compressed, parent, selected, raw_path)

        decontaminated_path = output_root / "benchmark_disjoint_candidates.jsonl"
        decontamination_receipt_path = output_root / "decontamination_receipt.json"
        decontamination = build_decontaminated(
            raw_path,
            [],
            decontaminated_path,
            decontamination_receipt_path,
            boundary_indexes=boundary_roots,
            workers=decontamination_workers,
        )
        answer_key_path = output_root / "answer_key_filtered_candidates.jsonl"
        answer_key_receipt_path = output_root / "answer_key_filter_receipt.json"
        answer_key = build_answer_key_filter(
            decontaminated_path,
            answer_key_path,
            answer_key_receipt_path,
        )
        candidates_path = output_root / "normalized_exact_deduplicated_candidates.jsonl"
        duplicate_manifest_path = output_root / "normalized_exact_duplicate_drops.jsonl"
        exact_receipt_path = output_root / "normalized_exact_deduplication_receipt.json"
        exact = build_exact_deduplication(
            [answer_key_path],
            candidates_path,
            duplicate_manifest_path,
            exact_receipt_path,
            temporary_root=output_root,
        )
        attribution_path = output_root / "attribution_manifest.jsonl"
        attribution_receipt_path = output_root / "attribution_receipt.json"
        attribution = build_attribution_manifest(
            raw_path,
            candidates_path,
            attribution_path,
            attribution_receipt_path,
        )
        final_rows = exact["counts"]["survivors"]
        if (
            raw["rows"] != scan["eligible_rows"]
            or decontamination["scanned"] != raw["rows"]
            or answer_key["scanned"] != decontamination["accepted"]
            or exact["counts"]["documents"] != answer_key["accepted"]
            or attribution["output"]["records"] != final_rows
        ):
            raise CommonPileFullSourceCandidatesError("candidate row custody differs")
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_full_source_candidates",
            "source_id": source_id,
            "promotion": {
                "path": str(promotion_path),
                "file_sha256": sha256_file(promotion_path),
                "receipt_sha256": promotion["receipt_sha256"],
                "source_checks": decision["checks"],
            },
            "parent": parent,
            "audit_populations": audit_receipts,
            "audit_excluded_content_identities": len(excluded_content),
            "scan": scan,
            "raw_population": raw,
            "benchmark_decontamination": {
                "receipt_path": decontamination_receipt_path.name,
                "receipt_file_sha256": sha256_file(decontamination_receipt_path),
                "receipt_sha256": decontamination["receipt_sha256"],
                "scanned": decontamination["scanned"],
                "accepted": decontamination["accepted"],
                "dropped": decontamination["dropped"],
            },
            "contextless_answer_key_filter": {
                "receipt_path": answer_key_receipt_path.name,
                "receipt_file_sha256": sha256_file(answer_key_receipt_path),
                "receipt_sha256": answer_key["receipt_sha256"],
                "scanned": answer_key["scanned"],
                "accepted": answer_key["accepted"],
                "dropped": answer_key["dropped_contextless_answer_keys"],
            },
            "normalized_exact_deduplication": {
                "receipt_path": exact_receipt_path.name,
                "receipt_file_sha256": sha256_file(exact_receipt_path),
                "receipt_sha256": exact["receipt_sha256"],
                "scanned": exact["counts"]["documents"],
                "survivors": final_rows,
                "duplicates_dropped": exact["counts"]["duplicates_dropped"],
                "duplicate_manifest_path": duplicate_manifest_path.name,
                "duplicate_manifest_sha256": sha256_file(duplicate_manifest_path),
            },
            "candidates": {
                "path": candidates_path.name,
                "rows": final_rows,
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
            },
            "attribution_manifest": {
                "path": attribution_path.name,
                "rows": attribution["output"]["records"],
                "bytes": attribution_path.stat().st_size,
                "sha256": sha256_file(attribution_path),
                "receipt_path": attribution_receipt_path.name,
                "receipt_sha256": attribution["receipt_sha256"],
            },
            "rejected_rows_excluded_from_candidates": (
                decontamination["dropped"]
                + answer_key["dropped_contextless_answer_keys"]
                + exact["counts"]["duplicates_dropped"]
            ),
            "complete_parent_row_coverage": True,
            "maximum_source_rows": MAXIMUM_SOURCE_ROWS,
            "maximum_simultaneous_parent_files": 1,
            "parent_removed_after_materialization": True,
            "benchmark_decontamination_complete": True,
            "contextless_answer_key_filter_complete": True,
            "normalized_exact_deduplication_complete": True,
            "global_near_duplicate_filtering_complete": False,
            "global_semantic_deduplication_complete": False,
            "representation_verification_complete": False,
            "rights_provenance_verified": False,
            "legal_clearance_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, action="append", required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--decontamination-workers", type=int, default=1)
    args = parser.parse_args()
    result = build_candidates(
        args.promotion,
        args.audit_root,
        args.boundary_index,
        args.output_root,
        source_id=args.source_id,
        token=os.environ.get(args.token_env, ""),
        decontamination_workers=args.decontamination_workers,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
