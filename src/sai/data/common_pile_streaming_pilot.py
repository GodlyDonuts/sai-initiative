"""Build one storage-bounded, audit-disjoint Common Pile source pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_near_duplicate_filter import build_filter as build_near_duplicate
from sai.data.common_pile_audit_population import (
    _declared_license,
    _native_id,
)
from sai.data.confirmation_promotion import SCHEMA as PROMOTION_SCHEMA
from sai.data.decontamination import RAW_SCHEMA
from sai.data.decontamination import build as build_decontaminated
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.license_policy import classify_declared_license
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-bounded-streaming-pilot-v1"
SEED = 20260825
MINIMUM_TEXT_BYTES = 200
MAXIMUM_TEXT_BYTES = 128 * 1024
SOURCE_DOMAINS = {
    "common_pile_arxiv_abstracts": "science",
    "common_pile_github_archive": "code",
    "common_pile_libretexts": "technical",
    "common_pile_pressbooks": "english",
    "common_pile_public_domain_review": "english",
    "common_pile_python_enhancement_proposals": "code",
    "common_pile_stackexchange": "technical",
}


class CommonPileStreamingPilotError(RuntimeError):
    """A promotion, source parent, bounded sample, or output differs."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise CommonPileStreamingPilotError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CommonPileStreamingPilotError(f"{label} cannot be decoded") from error
    if not isinstance(payload, dict):
        raise CommonPileStreamingPilotError(f"{label} differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise CommonPileStreamingPilotError(f"{label} receipt differs")
    return payload


def load_promotion(path: Path, source_id: str) -> dict[str, Any]:
    """Require exact bounded-pilot authorization for one named source."""

    payload = _load_json(path, "promotion decision")
    selected = payload.get("selected_source_ids")
    source_rows = payload.get("sources")
    matches = (
        [row for row in source_rows if row.get("source_id") == source_id]
        if isinstance(source_rows, list)
        else []
    )
    if (
        payload.get("schema") != PROMOTION_SCHEMA
        or payload.get("status") != "complete"
        or not isinstance(selected, list)
        or source_id not in selected
        or len(matches) != 1
        or matches[0].get("bounded_streaming_source_pilot_authorized") is not True
        or matches[0].get("bulk_training_admission") is not False
        or matches[0].get("training_ready") is not False
        or payload.get("bulk_training_admission") is not False
        or payload.get("full_source_ingestion_authorized") is not False
        or payload.get("training_ready") is not False
    ):
        raise CommonPileStreamingPilotError("source pilot is not authorized")
    return payload


def audit_exclusions(
    population_roots: list[Path], source_id: str
) -> tuple[dict[tuple[str, str], frozenset[int]], frozenset[str], list[dict[str, Any]]]:
    """Bind all audit row identities so pilot data cannot recycle its evidence."""

    if len(population_roots) != len(set(population_roots)) or not population_roots:
        raise CommonPileStreamingPilotError("audit population roots differ")
    lines: dict[tuple[str, str], set[int]] = defaultdict(set)
    content: set[str] = set()
    receipts = []
    for root in population_roots:
        candidates, lineage, receipt = load_population(root)
        for candidate, source in zip(candidates, lineage, strict=True):
            if source["source_id"] != source_id:
                continue
            line_number = source.get("locator", {}).get("line_number")
            if not isinstance(line_number, int) or isinstance(line_number, bool):
                raise CommonPileStreamingPilotError("audit row locator differs")
            lines[(source["repository"], source["path"])].add(line_number)
            content.add(candidate["source_content_sha256"])
        receipts.append(
            {
                "root_name": root.name,
                "receipt_sha256": receipt["receipt_sha256"],
                "population_sha256": sha256_file(root / "candidates.jsonl"),
                "lineage_sha256": sha256_file(root / "lineage.jsonl"),
            }
        )
    return (
        {key: frozenset(value) for key, value in lines.items()},
        frozenset(content),
        receipts,
    )


def select_parent(
    rows: list[dict[str, Any]],
    source_id: str,
    audit_parent_paths: set[tuple[str, str]],
) -> dict[str, Any]:
    """Prefer the smallest parent not used by either audit population."""

    matches = [row for row in rows if row["source_id"] == source_id]
    if not matches or source_id not in SOURCE_DOMAINS:
        raise CommonPileStreamingPilotError("pilot source is absent")
    alternatives = [
        row
        for row in matches
        if (row["repository"], row["path"]) not in audit_parent_paths
    ]
    selected = min(
        alternatives or matches,
        key=lambda row: (row["physical_bytes"], row["path"], row["sha256"]),
    )
    return {
        "source_id": source_id,
        "repository": selected["repository"],
        "revision": selected["revision"],
        "path": selected["path"],
        "bytes": selected["physical_bytes"],
        "sha256": selected["sha256"],
        "manifest_license": selected["license"],
        "domain": SOURCE_DOMAINS[source_id],
        "parent_disjoint_from_audits": (
            selected["repository"], selected["path"]
        )
        not in audit_parent_paths,
    }


def _selection_key(
    parent: dict[str, Any], line_number: int, row: dict[str, Any]
) -> str:
    text = row["text"].strip()
    return canonical_sha256(
        {
            "seed": SEED,
            "source_id": parent["source_id"],
            "parent_sha256": parent["sha256"],
            "line_number": line_number,
            "native_id": _native_id(row),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )


def select_bottom_k(
    compressed_path: Path,
    parent: dict[str, Any],
    *,
    maximum_rows: int,
    excluded_lines: frozenset[int],
    excluded_content_sha256s: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Scan a verified parent fully while retaining text-free bottom-k metadata."""

    if (
        isinstance(maximum_rows, bool)
        or not 1 <= maximum_rows <= 100_000
        or not compressed_path.is_file()
        or compressed_path.is_symlink()
        or compressed_path.stat().st_size != parent["bytes"]
        or sha256_file(compressed_path) != parent["sha256"]
    ):
        raise CommonPileStreamingPilotError("pilot parent identity differs")
    heap: list[tuple[int, str, int, str, str, str, str | None]] = []
    counters: Counter[str] = Counter()
    try:
        with gzip.open(compressed_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                counters["scanned_rows"] += 1
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    counters["non_text_rows"] += 1
                    continue
                text = row["text"].strip()
                text_bytes = len(text.encode())
                if text_bytes < MINIMUM_TEXT_BYTES:
                    counters["short_rows"] += 1
                    continue
                if text_bytes > MAXIMUM_TEXT_BYTES:
                    counters["oversized_rows"] += 1
                    continue
                text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                if (
                    line_number in excluded_lines
                    or text_sha256 in excluded_content_sha256s
                ):
                    counters["audit_excluded_rows"] += 1
                    continue
                license_name = _declared_license(row, parent["manifest_license"])
                if not license_name or len(license_name) > 256:
                    counters["invalid_license_rows"] += 1
                    continue
                license_classification = classify_declared_license(license_name)
                if license_classification["rights_hold"]:
                    counters["rights_hold_rows"] += 1
                    continue
                key = _selection_key(parent, line_number, {**row, "text": text})
                item = (
                    -int(key, 16),
                    key,
                    line_number,
                    text_sha256,
                    license_name,
                    license_classification["canonical_license"],
                    _native_id(row),
                )
                counters["eligible_rows"] += 1
                if len(heap) < maximum_rows:
                    heapq.heappush(heap, item)
                elif int(key, 16) < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CommonPileStreamingPilotError("pilot parent content differs") from error
    if not heap:
        raise CommonPileStreamingPilotError("pilot parent has no eligible rows")
    selected = [
        {
            "selection_key": key,
            "line_number": line_number,
            "text_sha256": text_sha256,
            "declared_license": license_name,
            "canonical_license": canonical_license,
            "native_id": native_id,
        }
        for (
            _,
            key,
            line_number,
            text_sha256,
            license_name,
            canonical_license,
            native_id,
        ) in sorted(
            heap, key=lambda item: item[2]
        )
    ]
    counters["selected_rows"] = len(selected)
    return selected, dict(sorted(counters.items()))


def write_raw_population(
    compressed_path: Path,
    parent: dict[str, Any],
    selected: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Replay selected line identities and write exact source-order raw rows."""

    if output_path.exists() or output_path.is_symlink() or not selected:
        raise CommonPileStreamingPilotError("pilot raw output boundary differs")
    by_line = {row["line_number"]: row for row in selected}
    if len(by_line) != len(selected):
        raise CommonPileStreamingPilotError("pilot selected lines are duplicated")
    written = 0
    ordered_identity = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with gzip.open(compressed_path, "rt", encoding="utf-8") as source:
            with temporary.open("w") as target:
                for line_number, line in enumerate(source, start=1):
                    descriptor = by_line.get(line_number)
                    if descriptor is None:
                        continue
                    row = json.loads(line)
                    text = row["text"].strip()
                    if (
                        hashlib.sha256(text.encode()).hexdigest()
                        != descriptor["text_sha256"]
                        or _native_id(row) != descriptor["native_id"]
                        or _declared_license(row, parent["manifest_license"])
                        != descriptor["declared_license"]
                        or _selection_key(parent, line_number, {**row, "text": text})
                        != descriptor["selection_key"]
                    ):
                        raise CommonPileStreamingPilotError(
                            "pilot selected row changed during replay"
                        )
                    raw = {
                        "schema": RAW_SCHEMA,
                        "text": text,
                        "source": {
                            "dataset": parent["repository"],
                            "revision": parent["revision"],
                            "source_file": parent["path"],
                            "row_index": line_number - 1,
                        "license": descriptor["canonical_license"],
                        "declared_license": descriptor["declared_license"],
                        "domain": parent["domain"],
                    },
                    }
                    target.write(
                        json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    ordered_identity.append(
                        {
                            "line_number": line_number,
                            "selection_key": descriptor["selection_key"],
                            "text_sha256": descriptor["text_sha256"],
                        }
                    )
                    written += 1
        if written != len(selected):
            raise CommonPileStreamingPilotError("pilot selected row coverage differs")
        if sha256_file(compressed_path) != parent["sha256"]:
            raise CommonPileStreamingPilotError("pilot parent changed during replay")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": output_path.name,
        "rows": written,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "ordered_selection_sha256": canonical_sha256(ordered_identity),
    }


def download_parent(parent: dict[str, Any], token: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise CommonPileStreamingPilotError("huggingface_hub is required") from error
    try:
        return Path(
            hf_hub_download(
                parent["repository"],
                parent["path"],
                repo_type="dataset",
                revision=parent["revision"],
                token=token,
                local_dir=local_dir,
            )
        )
    except Exception as error:
        raise CommonPileStreamingPilotError("pilot parent download failed") from error


def build_pilot(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    promotion_path: Path,
    audit_roots: list[Path],
    boundary_roots: list[Path],
    output_root: Path,
    *,
    source_id: str,
    maximum_rows: int,
    token: str,
) -> dict[str, Any]:
    """Acquire, audit-exclude, decontaminate, and seal one bounded pilot."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise CommonPileStreamingPilotError("pilot credential or output differs")
    promotion = load_promotion(promotion_path, source_id)
    excluded_lines, excluded_content, audit_receipts = audit_exclusions(
        audit_roots, source_id
    )
    rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    parent = select_parent(rows, source_id, set(excluded_lines))
    output_root.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="sai-common-pile-pilot-") as temp:
            compressed = download_parent(parent, token, Path(temp))
            selected, scan = select_bottom_k(
                compressed,
                parent,
                maximum_rows=maximum_rows,
                excluded_lines=excluded_lines.get(
                    (parent["repository"], parent["path"]), frozenset()
                ),
                excluded_content_sha256s=excluded_content,
            )
            raw_path = output_root / "raw_candidates.jsonl"
            raw = write_raw_population(compressed, parent, selected, raw_path)
        admitted_path = output_root / "benchmark_disjoint_candidates.jsonl"
        decontamination_receipt_path = output_root / "decontamination_receipt.json"
        decontamination = build_decontaminated(
            raw_path,
            [],
            admitted_path,
            decontamination_receipt_path,
            boundary_indexes=boundary_roots,
            workers=1,
        )
        near_duplicate_path = output_root / "bounded_near_deduplicated_candidates.jsonl"
        near_duplicate_receipt_path = output_root / "near_duplicate_receipt.json"
        near_duplicate = build_near_duplicate(
            admitted_path,
            near_duplicate_path,
            near_duplicate_receipt_path,
        )
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_pilot",
            "source_id": source_id,
            "seed": SEED,
            "maximum_rows": maximum_rows,
            "promotion": {
                "path": promotion_path.name,
                "bytes": promotion_path.stat().st_size,
                "sha256": sha256_file(promotion_path),
                "receipt_sha256": promotion["receipt_sha256"],
            },
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_sha256": sha256_file(reservoir_receipt_path),
            },
            "parent": parent,
            "audit_populations": audit_receipts,
            "audit_excluded_content_identities": len(excluded_content),
            "scan": scan,
            "raw_population": raw,
            "decontamination": {
                "receipt_path": decontamination_receipt_path.name,
                "receipt_file_sha256": sha256_file(
                    decontamination_receipt_path
                ),
                "receipt_sha256": decontamination["receipt_sha256"],
                "scanned": decontamination["scanned"],
                "accepted": decontamination["accepted"],
                "dropped": decontamination["dropped"],
                "output_path": admitted_path.name,
                "output_bytes": admitted_path.stat().st_size,
                "output_sha256": sha256_file(admitted_path),
            },
            "near_duplicate_filter": {
                "receipt_path": near_duplicate_receipt_path.name,
                "receipt_file_sha256": sha256_file(near_duplicate_receipt_path),
                "receipt_sha256": near_duplicate["receipt_sha256"],
                "input_documents": near_duplicate["input"]["documents"],
                "output_documents": near_duplicate["output"]["documents"],
                "documents_dropped": near_duplicate["evidence"][
                    "documents_dropped"
                ],
                "duplicate_groups": near_duplicate["evidence"][
                    "duplicate_groups"
                ],
                "output_path": near_duplicate_path.name,
                "output_bytes": near_duplicate_path.stat().st_size,
                "output_sha256": sha256_file(near_duplicate_path),
            },
            "parent_removed_after_pilot": True,
            "maximum_simultaneous_parent_files": 1,
            "full_source_ingestion_authorized": False,
            "bounded_pilot_near_duplicate_filter_complete": True,
            "global_cross_source_near_duplicate_filter_complete": False,
            "rights_verification_complete": False,
            "representation_verification_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        receipt_path = output_root / "receipt.json"
        _atomic_create(receipt_path, payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservoir-receipt", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, action="append", required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--maximum-rows", type=int, default=10_000)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = build_pilot(
        args.manifest,
        args.reservoir_receipt,
        args.promotion,
        args.audit_root,
        args.boundary_index,
        args.output_root,
        source_id=args.source_id,
        maximum_rows=args.maximum_rows,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
