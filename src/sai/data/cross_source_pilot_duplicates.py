"""Build an exhaustive bounded cross-source duplicate sample from source pilots."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_near_duplicate_filter import build_filter
from sai.data.common_pile_streaming_pilot import SCHEMA as COMMON_PILE_PILOT_SCHEMA
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-cross-source-pilot-duplicate-sample-v1"
SELECTION_SEED = "sai-cross-source-pilot-duplicates-v1"
MAXIMUM_ROWS = 10_000


class CrossSourcePilotDuplicateError(RuntimeError):
    """Pilot evidence, deterministic selection, or output custody differs."""


def _pilot_binding(root: Path) -> tuple[dict[str, Any], Path]:
    receipt_path = root / "receipt.json"
    try:
        receipt = _load_receipt(receipt_path)
    except Exception as error:
        raise CrossSourcePilotDuplicateError(
            f"cross-source pilot receipt differs: {receipt_path}"
        ) from error
    near_duplicate = receipt.get("near_duplicate_filter")
    if (
        receipt.get("schema") != COMMON_PILE_PILOT_SCHEMA
        or receipt.get("training_ready") is not False
        or receipt.get("bounded_pilot_near_duplicate_filter_complete") is not True
        or receipt.get("global_cross_source_near_duplicate_filter_complete")
        is not False
        or not isinstance(receipt.get("source_id"), str)
        or not isinstance(near_duplicate, dict)
    ):
        raise CrossSourcePilotDuplicateError("cross-source pilot state differs")
    descriptor = {
        "path": near_duplicate.get("output_path"),
        "bytes": near_duplicate.get("output_bytes"),
        "sha256": near_duplicate.get("output_sha256"),
    }
    try:
        document_path = _bound_file(root, descriptor)
    except Exception as error:
        raise CrossSourcePilotDuplicateError(
            "cross-source pilot document binding differs"
        ) from error
    return receipt, document_path


def _documents(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield normalize_document(json.loads(line))
            except Exception as error:
                raise CrossSourcePilotDuplicateError(
                    f"cross-source pilot row {line_number} differs"
                ) from error


def _selection_key(source_id: str, identity: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}\0{source_id}\0{identity}".encode()
    ).hexdigest()


def select_bottom_k(
    pilot_roots: list[Path], *, maximum_rows: int
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[dict[str, Any]]]:
    """Stream every pilot and retain deterministic bottom-k identities."""

    if (
        isinstance(maximum_rows, bool)
        or not isinstance(maximum_rows, int)
        or not 2 <= maximum_rows <= MAXIMUM_ROWS
        or len(pilot_roots) < 2
    ):
        raise CrossSourcePilotDuplicateError("cross-source sample geometry differs")
    identities: set[str] = set()
    bindings = []
    sources: dict[str, tuple[Path, dict[str, Any], Path]] = {}
    for root in pilot_roots:
        receipt, document_path = _pilot_binding(root)
        source_id = receipt["source_id"]
        if source_id in sources:
            raise CrossSourcePilotDuplicateError("cross-source sample repeats a source")
        sources[source_id] = (root, receipt, document_path)
    if maximum_rows < len(sources):
        raise CrossSourcePilotDuplicateError(
            "cross-source sample cannot cover every source"
        )
    base_quota, remainder = divmod(maximum_rows, len(sources))
    selected_items: list[tuple[int, str, str, str, dict[str, Any]]] = []
    global_heap: list[tuple[int, str, str, str, dict[str, Any]]] = []
    for source_index, source_id in enumerate(sorted(sources)):
        root, receipt, document_path = sources[source_id]
        quota = base_quota + (source_index < remainder)
        heap: list[tuple[int, str, str, str, dict[str, Any]]] = []
        observed = 0
        for document in _documents(document_path):
            identity = document["identity_sha256"]
            if identity in identities:
                raise CrossSourcePilotDuplicateError(
                    "cross-source sample repeats a document identity"
                )
            identities.add(identity)
            key = _selection_key(source_id, identity)
            item = (-int(key, 16), identity, key, source_id, document)
            if len(global_heap) < maximum_rows:
                heapq.heappush(global_heap, item)
            elif item[0] > global_heap[0][0]:
                heapq.heapreplace(global_heap, item)
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
            observed += 1
        selected_items.extend(heap)
        expected = receipt["near_duplicate_filter"].get("output_documents")
        if observed != expected:
            raise CrossSourcePilotDuplicateError(
                "cross-source pilot document coverage differs"
            )
        bindings.append(
            {
                "root": str(root.resolve()),
                "source_id": source_id,
                "receipt_sha256": receipt["receipt_sha256"],
                "near_deduplicated_documents": observed,
                "near_deduplicated_file_sha256": sha256_file(document_path),
            }
        )
    selected_identities = {item[1] for item in selected_items}
    for item in sorted(global_heap, reverse=True):
        if len(selected_items) >= maximum_rows:
            break
        if item[1] not in selected_identities:
            selected_items.append(item)
            selected_identities.add(item[1])
    selected = sorted(
        ((item[2], item[3], item[4]) for item in selected_items),
        key=lambda item: (item[0], item[2]["identity_sha256"]),
    )
    return selected, sorted(bindings, key=lambda row: row["source_id"])


def _write_population(
    selected: list[tuple[str, str, dict[str, Any]]], output_path: Path
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    ordered_identity = hashlib.sha256()
    by_source: Counter[str] = Counter()
    try:
        with temporary.open("w") as handle:
            for _, source_id, document in selected:
                handle.write(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                ordered_identity.update(bytes.fromhex(document["identity_sha256"]))
                by_source[source_id] += 1
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": output_path.name,
        "rows": len(selected),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "ordered_identity_sha256": ordered_identity.hexdigest(),
        "by_source": dict(sorted(by_source.items())),
    }


def build_sample(
    pilot_roots: list[Path], output_root: Path, *, maximum_rows: int
) -> dict[str, Any]:
    """Seal one bounded cross-source duplicate sample and filtered output."""

    if output_root.exists() or output_root.is_symlink():
        raise CrossSourcePilotDuplicateError("cross-source output already exists")
    output_root.mkdir(parents=True)
    try:
        selected, bindings = select_bottom_k(pilot_roots, maximum_rows=maximum_rows)
        if len({source_id for _, source_id, _ in selected}) < 2:
            raise CrossSourcePilotDuplicateError(
                "cross-source selection does not cover two sources"
            )
        input_path = output_root / "selected_candidates.jsonl"
        population = _write_population(selected, input_path)
        filtered_path = output_root / "deduplicated_candidates.jsonl"
        duplicate_receipt_path = output_root / "duplicate_receipt.json"
        duplicate = build_filter(input_path, filtered_path, duplicate_receipt_path)
        source_by_identity = {
            document["identity_sha256"]: source_id
            for _, source_id, document in selected
        }
        cross_source_groups = sum(
            len(
                {
                    source_by_identity[identity]
                    for identity in group["member_identity_sha256s"]
                }
            )
            > 1
            for group in duplicate["groups"]
        )
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_cross_source_sample",
            "selection": {
                "method": ("deterministic_source_stratified_bottom_k_then_global_fill"),
                "seed": SELECTION_SEED,
                "maximum_rows": maximum_rows,
                "input_documents": sum(
                    row["near_deduplicated_documents"] for row in bindings
                ),
                "input_sources": len(bindings),
            },
            "pilot_bindings": bindings,
            "population": population,
            "duplicate_filter": {
                "receipt_path": duplicate_receipt_path.name,
                "receipt_file_sha256": sha256_file(duplicate_receipt_path),
                "receipt_sha256": duplicate["receipt_sha256"],
                "output_path": filtered_path.name,
                "output_bytes": filtered_path.stat().st_size,
                "output_sha256": sha256_file(filtered_path),
                "input_documents": duplicate["input"]["documents"],
                "output_documents": duplicate["output"]["documents"],
                "documents_dropped": duplicate["evidence"]["documents_dropped"],
                "duplicate_groups": duplicate["evidence"]["duplicate_groups"],
                "cross_source_duplicate_groups": cross_source_groups,
            },
            "all_selected_unordered_pairs_logically_covered": True,
            "bounded_cross_source_pilot_sample_complete": True,
            "full_pilot_population_cross_source_deduplication_complete": (
                len(selected)
                == sum(row["near_deduplicated_documents"] for row in bindings)
            ),
            "full_reservoir_cross_source_deduplication_complete": False,
            "source_text_persisted_in_receipt": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-rows", type=int, default=MAXIMUM_ROWS)
    args = parser.parse_args()
    result = build_sample(
        args.pilot_root, args.output_root, maximum_rows=args.maximum_rows
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
