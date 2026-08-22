"""Freeze a blinded FineMath language-score quality-review ladder."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.finemath_audit import DATASET, validate_audit
from sai.data.finemath_filter import LICENSE, _signals
from sai.data.finemath_filter import _policy as _base_policy
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-finemath-language-ladder-receipt-v1"
CANDIDATE_SCHEMA = "sai-finemath-language-ladder-candidate-v1"
REVIEW_SCHEMA = "sai-finemath-language-ladder-blind-review-v1"
KEY_SCHEMA = "sai-finemath-language-ladder-key-v1"
SELECTION_SALT = b"sai-finemath-language-ladder-v1"
STRATA = ("below_0p90", "0p90_to_0p95", "at_least_0p95")
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "audit",
    "source",
    "policy",
    "policy_sha256",
    "summary",
    "candidate_output",
    "blind_review_output",
    "review_key_output",
    "limitations",
    "receipt_sha256",
}


class FineMathLadderError(RuntimeError):
    """The audited source, ladder policy, or blinded review packet differs."""


def _policy(*, review_per_stratum: int) -> dict[str, Any]:
    return {
        "base_filter": "finemath_filter_v1_all_non_language_rules_unchanged",
        "base_filter_policy_sha256": canonical_sha256(_base_policy()),
        "language_score_strata": [
            {"stratum": "below_0p90", "minimum_ppm": 0, "maximum_ppm": 899_999},
            {
                "stratum": "0p90_to_0p95",
                "minimum_ppm": 900_000,
                "maximum_ppm": 949_999,
            },
            {
                "stratum": "at_least_0p95",
                "minimum_ppm": 950_000,
                "maximum_ppm": 1_000_000,
            },
        ],
        "review_per_stratum": review_per_stratum,
        "selection": "lowest_sha256_within_nonoverlapping_stratum",
        "review_packet_hides_stratum_and_language_score": True,
        "key_is_separate_from_blind_review_packet": True,
        "human_review_required_before_threshold_selection": True,
        "candidate_authorizes_no_training": True,
    }


def _stratum(language_score_ppm: int) -> str:
    if language_score_ppm < 900_000:
        return "below_0p90"
    if language_score_ppm < 950_000:
        return "0p90_to_0p95"
    return "at_least_0p95"


def _import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FineMathLadderError("PyArrow is required for FineMath ladder") from error
    return pq


def _scan(
    source: Path,
    *,
    revision: str,
    source_file: str,
    review_per_stratum: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    if (
        isinstance(review_per_stratum, bool)
        or not isinstance(review_per_stratum, int)
        or review_per_stratum <= 0
    ):
        raise FineMathLadderError("FineMath ladder review geometry differs")
    pq = _import_parquet()
    parquet = pq.ParquetFile(source)
    columns = [
        "crawl",
        "int_score",
        "language_score",
        "metadata",
        "text",
        "url",
        "warc_filename",
        "warc_record_length",
        "warc_record_offset",
    ]
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise FineMathLadderError("FineMath ladder columns differ")
    policy = _policy(review_per_stratum=review_per_stratum)
    policy_sha256 = canonical_sha256(policy)
    candidates: list[dict[str, Any]] = []
    heaps: dict[str, list[tuple[int, int, dict[str, Any], dict[str, Any]]]] = {
        stratum: [] for stratum in STRATA
    }
    rows = 0
    base_rejection_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    seen_texts: set[str] = set()
    for batch in parquet.iter_batches(batch_size=2_048, columns=columns):
        values = batch.to_pydict()
        for raw in zip(*(values[column] for column in columns), strict=True):
            row = dict(zip(columns, raw, strict=True))
            row_index = rows
            rows += 1
            if not isinstance(row["text"], str) or not row["text"]:
                raise FineMathLadderError("FineMath ladder row differs")
            text_sha256 = hashlib.sha256(row["text"].encode()).hexdigest()
            duplicate = text_sha256 in seen_texts
            seen_texts.add(text_sha256)
            reasons, signals = _signals(row)
            base_reasons = [
                reason
                for reason in reasons
                if reason != "language_confidence_below_0p98"
            ]
            if duplicate:
                base_reasons.append("exact_duplicate_text")
                base_reasons.sort()
            if base_reasons:
                base_rejection_counts.update(base_reasons)
                continue
            language_score_ppm = signals["language_score_ppm"]
            stratum = _stratum(language_score_ppm)
            candidate_counts[stratum] += 1
            identity = canonical_sha256(
                {
                    "dataset": DATASET,
                    "revision": revision,
                    "source_file": source_file,
                    "row_index": row_index,
                    "text_sha256": text_sha256,
                    "policy_sha256": policy_sha256,
                }
            )
            candidate = {
                "schema": CANDIDATE_SCHEMA,
                "identity_sha256": identity,
                "stratum": stratum,
                "language_score_ppm": language_score_ppm,
                "source": {
                    "dataset": DATASET,
                    "revision": revision,
                    "source_file": source_file,
                    "row_index": row_index,
                    "url": row["url"],
                    "warc_filename": row["warc_filename"],
                    "warc_record_offset": row["warc_record_offset"],
                    "warc_record_length": row["warc_record_length"],
                    "crawl": row["crawl"],
                    "license": LICENSE,
                },
                "text_sha256": text_sha256,
                "text": row["text"],
                "limitations": [
                    "candidate_not_human_quality_approved",
                    "candidate_not_near_deduplicated",
                    "candidate_not_benchmark_decontaminated",
                ],
            }
            candidates.append(candidate)
            blind = {
                "schema": REVIEW_SCHEMA,
                "review_identity_sha256": identity,
                "source_url": row["url"],
                "text_sha256": text_sha256,
                "text": row["text"],
            }
            rank = hashlib.sha256(SELECTION_SALT + bytes.fromhex(identity)).hexdigest()
            key = {
                "schema": KEY_SCHEMA,
                "review_identity_sha256": identity,
                "selection_rank_sha256": rank,
                "stratum": stratum,
                "language_score_ppm": language_score_ppm,
            }
            item = (-int(rank, 16), -row_index, blind, key)
            heap = heaps[stratum]
            if len(heap) < review_per_stratum:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    blind_rows = []
    key_rows = []
    for stratum in STRATA:
        if len(heaps[stratum]) != review_per_stratum:
            raise FineMathLadderError("FineMath ladder stratum is incomplete")
        selected = sorted(heaps[stratum], key=lambda item: (-item[0], -item[1]))
        blind_rows.extend(item[2] for item in selected)
        key_rows.extend(item[3] for item in selected)
    order = sorted(
        range(len(blind_rows)),
        key=lambda index: hashlib.sha256(
            b"sai-finemath-blind-order-v1"
            + bytes.fromhex(blind_rows[index]["review_identity_sha256"])
        ).digest(),
    )
    blind_rows = [blind_rows[index] for index in order]
    key_by_identity = {row["review_identity_sha256"]: row for row in key_rows}
    key_rows = [key_by_identity[row["review_identity_sha256"]] for row in blind_rows]
    summary = {
        "source_rows": rows,
        "unique_text_sha256": len(seen_texts),
        "base_candidate_rows": len(candidates),
        "candidate_rows_by_stratum": {
            stratum: candidate_counts[stratum] for stratum in STRATA
        },
        "base_rejection_reason_counts": dict(sorted(base_rejection_counts.items())),
        "blind_review_rows": len(blind_rows),
        "blind_review_rows_by_stratum": dict.fromkeys(STRATA, review_per_stratum),
    }
    return candidates, blind_rows, key_rows, summary


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in rows
    )


def _descriptor(
    path: Path, rows: list[dict[str, Any]], encoded: bytes
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "ordered_rows_sha256": canonical_sha256(rows),
    }


def build_ladder(
    audit_receipt: Path,
    candidate_output: Path,
    blind_review_output: Path,
    review_key_output: Path,
    receipt_output: Path,
    *,
    review_per_stratum: int = 64,
) -> dict[str, Any]:
    """Build one blinded, non-overlapping language-score quality review."""

    outputs = (
        candidate_output,
        blind_review_output,
        review_key_output,
        receipt_output,
    )
    if len({path.parent.resolve() for path in outputs}) != 1:
        raise FineMathLadderError("FineMath ladder outputs must share one parent")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FineMathLadderError("FineMath ladder output already exists")
    try:
        audit = validate_audit(audit_receipt)
    except Exception as error:
        raise FineMathLadderError("FineMath audit validation failed") from error
    source_row = audit["source"]
    candidates, blind_rows, key_rows, summary = _scan(
        Path(source_row["path"]),
        revision=source_row["revision"],
        source_file=source_row["source_file"],
        review_per_stratum=review_per_stratum,
    )
    encoded_rows = [_jsonl(rows) for rows in (candidates, blind_rows, key_rows)]
    if any(not encoded for encoded in encoded_rows):
        raise FineMathLadderError("FineMath ladder produced an empty artifact")
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    stages = [path.with_name(f".{path.name}.partial.{os.getpid()}") for path in outputs]
    try:
        for stage, encoded in zip(stages[:3], encoded_rows, strict=True):
            with stage.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        policy = _policy(review_per_stratum=review_per_stratum)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "selected_for_blind_quality_review",
            "training_authorized": False,
            "four_b_training_authorized": False,
            "audit": {
                "path": str(audit_receipt.resolve()),
                "file_sha256": sha256_file(audit_receipt),
                "receipt_sha256": audit["receipt_sha256"],
            },
            "source": source_row,
            "policy": policy,
            "policy_sha256": canonical_sha256(policy),
            "summary": summary,
            "candidate_output": _descriptor(
                candidate_output, candidates, encoded_rows[0]
            ),
            "blind_review_output": _descriptor(
                blind_review_output, blind_rows, encoded_rows[1]
            ),
            "review_key_output": _descriptor(
                review_key_output, key_rows, encoded_rows[2]
            ),
            "limitations": [
                "reviewers_must_not_open_review_key_before_labeling",
                "human_review_required_before_threshold_selection",
                "selected_candidates_require_global_near_deduplication",
                "selected_candidates_require_benchmark_decontamination",
                "receipt_authorizes_no_training",
            ],
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        with stages[3].open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for stage, output in zip(stages, outputs, strict=True):
            os.replace(stage, output)
        return payload
    except BaseException:
        for stage in stages:
            stage.unlink(missing_ok=True)
        raise


def validate_ladder(receipt: Path) -> dict[str, Any]:
    """Recompute the candidate population, blind packet, and hidden key."""

    if not receipt.is_file() or receipt.is_symlink() or receipt.stat().st_nlink != 1:
        raise FineMathLadderError("FineMath ladder receipt is missing or unsafe")
    try:
        payload = json.loads(receipt.read_text())
    except json.JSONDecodeError as error:
        raise FineMathLadderError("FineMath ladder receipt JSON differs") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_KEYS
        or payload.get("schema") != SCHEMA
        or payload.get("receipt_sha256")
        != canonical_sha256({k: v for k, v in payload.items() if k != "receipt_sha256"})
    ):
        raise FineMathLadderError("FineMath ladder receipt differs")
    audit_row = payload["audit"]
    audit_receipt = Path(audit_row.get("path", ""))
    try:
        audit = validate_audit(audit_receipt)
    except Exception as error:
        raise FineMathLadderError("FineMath audit validation failed") from error
    if audit_row != {
        "path": str(audit_receipt.resolve()),
        "file_sha256": sha256_file(audit_receipt),
        "receipt_sha256": audit["receipt_sha256"],
    }:
        raise FineMathLadderError("FineMath ladder audit lineage differs")
    policy = payload["policy"]
    review_per_stratum = policy.get("review_per_stratum")
    if policy != _policy(review_per_stratum=review_per_stratum):
        raise FineMathLadderError("FineMath ladder policy differs")
    source_row = audit["source"]
    candidates, blind_rows, key_rows, summary = _scan(
        Path(source_row["path"]),
        revision=source_row["revision"],
        source_file=source_row["source_file"],
        review_per_stratum=review_per_stratum,
    )
    rows_by_field = {
        "candidate_output": candidates,
        "blind_review_output": blind_rows,
        "review_key_output": key_rows,
    }
    for field, rows in rows_by_field.items():
        descriptor = payload[field]
        path = Path(descriptor.get("path", ""))
        encoded = _jsonl(rows)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.read_bytes() != encoded
            or descriptor != _descriptor(path, rows, encoded)
        ):
            raise FineMathLadderError(f"FineMath ladder {field} differs")
    if (
        payload["status"] != "selected_for_blind_quality_review"
        or payload["training_authorized"] is not False
        or payload["four_b_training_authorized"] is not False
        or payload["source"] != source_row
        or payload["policy_sha256"] != canonical_sha256(policy)
        or payload["summary"] != summary
        or payload["limitations"]
        != [
            "reviewers_must_not_open_review_key_before_labeling",
            "human_review_required_before_threshold_selection",
            "selected_candidates_require_global_near_deduplication",
            "selected_candidates_require_benchmark_decontamination",
            "receipt_authorizes_no_training",
        ]
    ):
        raise FineMathLadderError("FineMath ladder replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--audit-receipt", type=Path, required=True)
    build.add_argument("--candidate-output", type=Path, required=True)
    build.add_argument("--blind-review-output", type=Path, required=True)
    build.add_argument("--review-key-output", type=Path, required=True)
    build.add_argument("--receipt-output", type=Path, required=True)
    build.add_argument("--review-per-stratum", type=int, default=64)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_ladder(
            args.audit_receipt,
            args.candidate_output,
            args.blind_review_output,
            args.review_key_output,
            args.receipt_output,
            review_per_stratum=args.review_per_stratum,
        )
    else:
        payload = validate_ladder(args.receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_rows": payload["summary"]["base_candidate_rows"],
                "blind_review_rows": payload["summary"]["blind_review_rows"],
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
