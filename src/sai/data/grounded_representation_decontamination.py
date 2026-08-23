"""Screen generated grounded representations against official benchmark indexes."""

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
from sai.data.grounded_representation_aggregate import (
    REPRESENTATION_SCHEMA,
)
from sai.data.grounded_representation_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-representation-decontamination-v1"
DECISION_SCHEMA = "sai-grounded-representation-decontamination-record-v1"
CLEAN_SCHEMA = "sai-benchmark-disjoint-grounded-representation-candidate-v1"


class GroundedRepresentationDecontaminationError(RuntimeError):
    """The generated candidate, benchmark boundary, or screen differs."""


def load_representations(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the complete generated representation candidate population."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("representations")
    if (
        receipt.get("schema") != AGGREGATE_SCHEMA
        or receipt.get("status")
        != "complete_unverified_grounded_representation_candidates"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_text_persisted_in_candidate_outputs") is not False
        or receipt.get("evidence_quote_text_persisted_in_candidate_outputs")
        is not False
        or receipt.get("source_license") != "CC-BY-SA-4.0"
        or receipt.get("attribution_required") is not True
        or receipt.get("share_alike_required") is not True
        or receipt.get("benchmark_decontamination_complete") is not False
        or receipt.get("representation_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedRepresentationDecontaminationError(
            "grounded representation aggregate differs"
        )
    path = _bound_file(root, descriptor)
    rows = []
    identities = set()
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("record_sha256")
                text = row.get("text")
                unsigned_row = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                if (
                    row.get("schema") != REPRESENTATION_SCHEMA
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in identities
                    or not isinstance(text, str)
                    or not text
                    or row.get("record_sha256") != canonical_sha256(unsigned_row)
                    or row.get("benchmark_decontamination_complete") is not False
                    or row.get("representation_verified") is not False
                    or row.get("training_ready") is not False
                ):
                    raise GroundedRepresentationDecontaminationError(
                        "grounded representation candidate differs"
                    )
                identities.add(identity)
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedRepresentationDecontaminationError(
            "grounded representation candidate differs"
        ) from error
    if (
        not rows
        or len(rows) != descriptor.get("rows")
        or descriptor.get("ordered_records_sha256")
        != canonical_sha256([row["record_sha256"] for row in rows])
    ):
        raise GroundedRepresentationDecontaminationError(
            "grounded representation coverage differs"
        )
    return rows, receipt


def screen_representation(
    candidate: dict[str, Any],
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> dict[str, Any]:
    """Create one text-free exact-shingle decision."""

    identity = candidate.get("record_sha256")
    text = candidate.get("text")
    if (
        candidate.get("schema") != REPRESENTATION_SCHEMA
        or not isinstance(identity, str)
        or len(identity) != 64
        or not isinstance(text, str)
        or not text
    ):
        raise GroundedRepresentationDecontaminationError(
            "representation screen candidate differs"
        )
    normalized = _normalize(text)
    word_overlaps = _overlap_count(
        _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
    )
    code_overlaps = _code_overlap_count(_CODE.findall(normalized), code_boundary)
    decision = {
        "schema": DECISION_SCHEMA,
        "representation_record_sha256": identity,
        "source_candidate_identity_sha256": candidate[
            "source_candidate_identity_sha256"
        ],
        "word_overlap_count": word_overlaps,
        "code_overlap_count": code_overlaps,
        "contaminated": bool(word_overlaps or code_overlaps),
        "representation_text_persisted": False,
        "training_ready": False,
    }
    decision["record_sha256"] = canonical_sha256(decision)
    return decision


def promote_clean(candidate: dict[str, Any]) -> dict[str, Any]:
    """Advance one clean row only through the contamination state transition."""

    if (
        candidate.get("schema") != REPRESENTATION_SCHEMA
        or candidate.get("benchmark_decontamination_complete") is not False
        or candidate.get("representation_verified") is not False
        or candidate.get("training_ready") is not False
    ):
        raise GroundedRepresentationDecontaminationError(
            "clean representation candidate differs"
        )
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
    """Screen every generated text and emit a benchmark-disjoint candidate set."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not boundary_roots
        or len(boundary_roots) != len(set(boundary_roots))
    ):
        raise GroundedRepresentationDecontaminationError(
            "representation decontamination output differs"
        )
    candidates, aggregate = load_representations(aggregate_root)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        decisions = [
            screen_representation(candidate, word_boundary, code_boundary)
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
    if any(
        candidate["record_sha256"] != decision["representation_record_sha256"]
        for candidate, decision in zip(candidates, decisions, strict=True)
    ):
        raise GroundedRepresentationDecontaminationError(
            "representation decision coverage differs"
        )
    output_root.mkdir(parents=True)
    try:
        decision_path = output_root / "decisions.jsonl"
        clean_path = output_root / "benchmark_disjoint_representations.jsonl"
        _atomic_jsonl(decision_path, decisions)
        _atomic_jsonl(clean_path, clean)
        totals = Counter()
        for decision in decisions:
            totals["contaminated_rows"] += decision["contaminated"]
            totals["word_overlap_shingles"] += decision["word_overlap_count"]
            totals["code_overlap_shingles"] += decision["code_overlap_count"]
        payload = {
            "schema": SCHEMA,
            "status": "complete_post_generation_benchmark_screen",
            "aggregate": {
                "root_name": aggregate_root.name,
                "receipt_file_sha256": sha256_file(aggregate_root / "receipt.json"),
                "receipt_sha256": aggregate["receipt_sha256"],
                "representation_file_sha256": aggregate["representations"]["sha256"],
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
                "representation_text_persisted": False,
            },
            "benchmark_disjoint_representations": {
                "path": clean_path.name,
                "rows": len(clean),
                "bytes": clean_path.stat().st_size,
                "sha256": sha256_file(clean_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in clean]
                ),
                "text_bytes": sum(len(row["text"].encode()) for row in clean),
            },
            "post_generation_benchmark_screen_complete": True,
            "source_claims_independently_verified": False,
            "external_bridge_anchors_verified": False,
            "global_deduplication_complete": False,
            "representation_verification_complete": False,
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
