"""Screen independently retained grounded bridges against benchmark indexes."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Container
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_overlap_count,
    _normalize,
    _overlap_count,
    binary_boundary_index,
)
from sai.data.nemotron_grounded_bridge_verification_aggregate import (
    RETAINED_SCHEMA,
)
from sai.data.nemotron_grounded_bridge_verification_aggregate import (
    SCHEMA as INDEPENDENT_AGGREGATE_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-bridge-decontamination-v1"
DECISION_SCHEMA = "sai-grounded-bridge-decontamination-record-v1"
CLEAN_SCHEMA = "sai-benchmark-disjoint-grounded-bridge-candidate-v1"
TEXT_FIELDS = (
    "bridge_thesis",
    "shared_structure",
    "claims",
    "representations",
    "prerequisite_map",
    "analogy_failure_modes",
    "verification_questions",
)


class GroundedBridgeDecontaminationError(RuntimeError):
    """The retained bridge, benchmark boundary, or screen differs."""


def _bridge_text(candidate: dict[str, Any]) -> str:
    values = {key: candidate.get(key) for key in TEXT_FIELDS}
    if any(value is None for value in values.values()):
        raise GroundedBridgeDecontaminationError("bridge generated fields differ")
    return json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def load_retained_bridges(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact independently retained bridge population."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("retained")
    if (
        receipt.get("schema") != INDEPENDENT_AGGREGATE_SCHEMA
        or receipt.get("status")
        != "complete_independent_model_family_bridge_verification_routes"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("same_model_family_as_generator") is not False
        or receipt.get("independent_model_family_verification_complete") is not True
        or receipt.get("benchmark_decontamination_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedBridgeDecontaminationError(
            "independent bridge aggregate differs"
        )
    path = _bound_file(root, descriptor)
    rows = []
    identities = set()
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("record_sha256")
                unsigned_row = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                if (
                    row.get("schema") != RETAINED_SCHEMA
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in identities
                    or identity != canonical_sha256(unsigned_row)
                    or row.get("source_text_persisted") is not False
                    or row.get("independent_family_retention_passed") is not True
                    or row.get("benchmark_decontamination_complete") is not False
                    or row.get("bridge_verified") is not False
                    or row.get("training_ready") is not False
                ):
                    raise GroundedBridgeDecontaminationError(
                        "retained bridge candidate differs"
                    )
                _bridge_text(row)
                identities.add(identity)
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedBridgeDecontaminationError(
            "retained bridge candidate differs"
        ) from error
    if (
        len(rows) != descriptor.get("rows")
        or descriptor.get("ordered_records_sha256")
        != canonical_sha256([row["record_sha256"] for row in rows])
    ):
        raise GroundedBridgeDecontaminationError(
            "retained bridge coverage differs"
        )
    return rows, receipt


def screen_bridge(
    candidate: dict[str, Any],
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> dict[str, Any]:
    """Create one text-free exact-shingle bridge decision."""

    identity = candidate.get("record_sha256")
    if candidate.get("schema") != RETAINED_SCHEMA or not isinstance(identity, str):
        raise GroundedBridgeDecontaminationError("bridge screen candidate differs")
    normalized = _normalize(_bridge_text(candidate))
    word_overlaps = _overlap_count(
        _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
    )
    code_overlaps = _code_overlap_count(_CODE.findall(normalized), code_boundary)
    decision = {
        "schema": DECISION_SCHEMA,
        "bridge_record_sha256": identity,
        "verification_candidate_identity_sha256": candidate[
            "verification_candidate_identity_sha256"
        ],
        "word_overlap_count": word_overlaps,
        "code_overlap_count": code_overlaps,
        "contaminated": bool(word_overlaps or code_overlaps),
        "bridge_text_persisted": False,
        "training_ready": False,
    }
    decision["record_sha256"] = canonical_sha256(decision)
    return decision


def promote_clean(candidate: dict[str, Any]) -> dict[str, Any]:
    """Advance one bridge only through the benchmark-disjoint transition."""

    if (
        candidate.get("schema") != RETAINED_SCHEMA
        or candidate.get("benchmark_decontamination_complete") is not False
        or candidate.get("bridge_verified") is not False
        or candidate.get("training_ready") is not False
    ):
        raise GroundedBridgeDecontaminationError("clean bridge candidate differs")
    row = {
        **{
            key: value
            for key, value in candidate.items()
            if key not in {"schema", "record_sha256"}
        },
        "schema": CLEAN_SCHEMA,
        "pre_decontamination_record_sha256": candidate["record_sha256"],
        "benchmark_decontamination_complete": True,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


class _Union:
    def __init__(self, members: list[Container[bytes]]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def build_screen(
    aggregate_root: Path, boundary_roots: list[Path], output_root: Path
) -> dict[str, Any]:
    """Screen every retained bridge and emit benchmark-disjoint candidates."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not boundary_roots
        or len(boundary_roots) != len(set(boundary_roots))
    ):
        raise GroundedBridgeDecontaminationError(
            "bridge decontamination output differs"
        )
    candidates, aggregate = load_retained_bridges(aggregate_root)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        decisions = [
            screen_bridge(candidate, word_boundary, code_boundary)
            for candidate in candidates
        ]
    finally:
        for member in [*words, *code]:
            member.close()
    clean = [
        promote_clean(candidate)
        for candidate, decision in zip(candidates, decisions, strict=True)
        if not decision["contaminated"]
    ]
    output_root.mkdir(parents=True)
    try:
        decision_path = output_root / "decisions.jsonl"
        clean_path = output_root / "benchmark_disjoint_bridges.jsonl"
        _atomic_jsonl(decision_path, decisions)
        _atomic_jsonl(clean_path, clean)
        totals: Counter[str] = Counter()
        for decision in decisions:
            totals["contaminated_rows"] += decision["contaminated"]
            totals["word_overlap_shingles"] += decision["word_overlap_count"]
            totals["code_overlap_shingles"] += decision["code_overlap_count"]
        payload = {
            "schema": SCHEMA,
            "status": "complete_post_generation_bridge_benchmark_screen",
            "aggregate": {
                "root_name": aggregate_root.name,
                "receipt_file_sha256": sha256_file(aggregate_root / "receipt.json"),
                "receipt_sha256": aggregate["receipt_sha256"],
                "retained_file_sha256": aggregate["retained"]["sha256"],
            },
            "boundary_indexes": boundary_receipts,
            "boundary_indexes_sha256": canonical_sha256(boundary_receipts),
            "policy": POLICY,
            "policy_sha256": canonical_sha256(POLICY),
            "input_rows": len(candidates),
            "clean_rows": len(clean),
            "contaminated_rows": totals["contaminated_rows"],
            "word_overlap_shingles": totals["word_overlap_shingles"],
            "code_overlap_shingles": totals["code_overlap_shingles"],
            "decisions": {
                "path": decision_path.name,
                "rows": len(decisions),
                "bytes": decision_path.stat().st_size,
                "sha256": sha256_file(decision_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in decisions]
                ),
                "bridge_text_persisted": False,
            },
            "benchmark_disjoint_bridges": {
                "path": clean_path.name,
                "rows": len(clean),
                "bytes": clean_path.stat().st_size,
                "sha256": sha256_file(clean_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in clean]
                ),
                "text_bytes": sum(
                    len(_bridge_text(row).encode()) for row in clean
                ),
            },
            "post_generation_benchmark_screen_complete": True,
            "independent_model_family_verification_complete": True,
            "global_deduplication_complete": False,
            "transfer_ablation_complete": False,
            "bridge_verified": False,
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
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_screen(args.aggregate_root, args.boundary_index, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
