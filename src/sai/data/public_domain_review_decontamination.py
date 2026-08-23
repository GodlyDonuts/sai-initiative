"""Re-screen materialized Public Domain Review text after quote removal."""

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
from sai.data.public_domain_review_scoped_candidates import (
    CANDIDATE_SCHEMA,
)
from sai.data.public_domain_review_scoped_candidates import (
    SCHEMA as MATERIALIZATION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-public-domain-review-post-scope-decontamination-v1"
DECISION_SCHEMA = "sai-public-domain-review-post-scope-decontamination-record-v1"


class PublicDomainReviewDecontaminationError(RuntimeError):
    """The materialized candidate, benchmark boundary, or decision differs."""


def _load_candidates(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("scoped_candidates")
    if (
        receipt.get("schema") != MATERIALIZATION_SCHEMA
        or receipt.get("status") != "complete_nontraining_scoped_candidates"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_page_html_persisted") is not False
        or receipt.get("source_text_persisted_in_candidate_file") is not True
        or receipt.get("content_quality_verified") is not False
        or receipt.get("legal_clearance_established") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise PublicDomainReviewDecontaminationError("materialization receipt differs")
    path = _bound_file(root, descriptor)
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            unsigned_row = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or row.get("record_sha256") != canonical_sha256(unsigned_row)
                or len(row.get("text", "").encode()) != row.get("scoped_text_bytes")
                or row.get("source_page_replay_complete") is not True
                or row.get("rights_scope_evidence_observed") is not True
                or row.get("content_quality_verified") is not False
                or row.get("legal_clearance_established") is not False
                or row.get("training_ready") is not False
            ):
                raise PublicDomainReviewDecontaminationError(
                    "materialized candidate differs"
                )
            rows.append(row)
    if len(rows) != descriptor.get("rows") or descriptor.get(
        "ordered_records_sha256"
    ) != canonical_sha256([row["record_sha256"] for row in rows]):
        raise PublicDomainReviewDecontaminationError(
            "materialized candidate coverage differs"
        )
    return rows, receipt


def screen_candidate(
    candidate: dict[str, Any],
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> dict[str, Any]:
    """Return a text-free exact-shingle decision for one transformed row."""

    identity = candidate.get("original_candidate_identity_sha256")
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or not isinstance(identity, str)
        or len(identity) != 64
        or not isinstance(candidate.get("text"), str)
    ):
        raise PublicDomainReviewDecontaminationError("screen candidate differs")
    normalized = _normalize(candidate["text"])
    word_overlaps = _overlap_count(
        _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
    )
    code_overlaps = _code_overlap_count(_CODE.findall(normalized), code_boundary)
    decision = {
        "schema": DECISION_SCHEMA,
        "original_candidate_identity_sha256": identity,
        "scoped_candidate_record_sha256": candidate["record_sha256"],
        "word_overlap_count": word_overlaps,
        "code_overlap_count": code_overlaps,
        "contaminated": bool(word_overlaps or code_overlaps),
        "source_text_persisted": False,
        "training_ready": False,
    }
    decision["record_sha256"] = canonical_sha256(decision)
    return decision


class _Union:
    def __init__(self, members: list[Container[bytes]]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def build_screen(
    materialization_root: Path, boundary_roots: list[Path], output_root: Path
) -> dict[str, Any]:
    """Create a post-transformation benchmark-disjoint candidate population."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not boundary_roots
        or len(set(boundary_roots)) != len(boundary_roots)
    ):
        raise PublicDomainReviewDecontaminationError(
            "decontamination output or boundary differs"
        )
    candidates, materialization = _load_candidates(materialization_root)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        decisions = [
            screen_candidate(candidate, word_boundary, code_boundary)
            for candidate in candidates
        ]
    finally:
        for member in [*words, *code]:
            member.close()
    clean = [
        candidate
        for candidate, decision in zip(candidates, decisions, strict=True)
        if not decision["contaminated"]
    ]
    if (
        len(decisions) != len(candidates)
        or len({row["original_candidate_identity_sha256"] for row in decisions})
        != len(candidates)
        or any(
            decision["original_candidate_identity_sha256"]
            != candidate["original_candidate_identity_sha256"]
            for candidate, decision in zip(candidates, decisions, strict=True)
        )
    ):
        raise PublicDomainReviewDecontaminationError(
            "decontamination decision coverage differs"
        )
    output_root.mkdir(parents=True)
    try:
        decisions_path = output_root / "decisions.jsonl"
        clean_path = output_root / "benchmark_disjoint_candidates.jsonl"
        _atomic_jsonl(decisions_path, decisions)
        _atomic_jsonl(clean_path, clean)
        totals = Counter()
        for decision in decisions:
            totals["contaminated_rows"] += decision["contaminated"]
            totals["word_overlap_shingles"] += decision["word_overlap_count"]
            totals["code_overlap_shingles"] += decision["code_overlap_count"]
        payload = {
            "schema": SCHEMA,
            "status": "complete_post_scope_benchmark_screen",
            "materialization": {
                "root_name": materialization_root.name,
                "receipt_file_sha256": sha256_file(
                    materialization_root / "receipt.json"
                ),
                "receipt_sha256": materialization["receipt_sha256"],
                "candidate_file_sha256": materialization["scoped_candidates"]["sha256"],
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
                "path": decisions_path.name,
                "rows": len(decisions),
                "bytes": decisions_path.stat().st_size,
                "sha256": sha256_file(decisions_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in decisions]
                ),
                "source_text_persisted": False,
            },
            "benchmark_disjoint_candidates": {
                "path": clean_path.name,
                "rows": len(clean),
                "bytes": clean_path.stat().st_size,
                "sha256": sha256_file(clean_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in clean]
                ),
                "text_bytes": sum(row["scoped_text_bytes"] for row in clean),
            },
            "post_transformation_benchmark_screen_complete": True,
            "full_source_population_decontaminated": False,
            "content_quality_verified": False,
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
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_screen(
        args.materialization_root, args.boundary_index, args.output_root
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
