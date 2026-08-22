"""Build a conservative, review-only FineMath candidate from an audited shard."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.finemath_audit import DATASET, RISK_PATTERNS, validate_audit
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-finemath-filtered-candidate-receipt-v1"
ROW_SCHEMA = "sai-finemath-filtered-candidate-v1"
REVIEW_SCHEMA = "sai-finemath-filter-review-row-v1"
SELECTION_SALT = b"sai-finemath-filter-review-v1"
LICENSE = "ODC-By-1.0"
DENIED_HOST_SUFFIXES = {
    "chegg.com",
    "coursehero.com",
    "gradesaver.com",
    "jiskha.com",
    "masterpapers.com",
}
MATH_PATTERNS = (
    re.compile(r"\\(?:frac|sum|int|sqrt|begin|end|mathbf|mathbb)\b"),
    re.compile(r"[$][^$\n]{1,200}[$]"),
    re.compile(r"\b(?:theorem|lemma|corollary|proof|equation|inequality)\b", re.I),
    re.compile(r"(?:^|\s)[A-Za-z0-9)]\s*[=<>±×÷]\s*[A-Za-z0-9(]"),
)
EXPLANATION_PATTERNS = (
    re.compile(r"\b(?:because|therefore|hence|thus|so that)\b", re.I),
    re.compile(r"\b(?:let|suppose|assume|given|define)\b", re.I),
    re.compile(r"\b(?:example|solution|derivation|step|explanation)\b", re.I),
    re.compile(r"\b(?:we can|we have|it follows|this means)\b", re.I),
)
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
    "accepted_output",
    "review_output",
    "limitations",
    "receipt_sha256",
}


class FineMathFilterError(RuntimeError):
    """The FineMath audit, filter policy, or candidate output differs."""


def _policy() -> dict[str, Any]:
    return {
        "required_upstream_integer_score": 5,
        "required_found_math": True,
        "minimum_language_score_ppm": 980_000,
        "minimum_words": 160,
        "maximum_characters": 200_000,
        "minimum_distinct_math_signal_classes": 2,
        "minimum_distinct_explanation_signal_classes": 2,
        "maximum_url_like_strings": 8,
        "reject_exact_duplicate_text_after_first_occurrence": True,
        "risk_patterns": {
            name: pattern.pattern for name, pattern in sorted(RISK_PATTERNS.items())
        },
        "denied_host_suffixes": sorted(DENIED_HOST_SUFFIXES),
        "accepted_license": LICENSE,
        "review_selection": "lowest_sha256_per_decision",
        "accepted_is_candidate_not_training_admission": True,
    }


def _import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FineMathFilterError(
            "PyArrow is required for FineMath filtering"
        ) from error
    return pq


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").casefold()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _host_denied(host: str) -> bool:
    return any(
        host == suffix or host.endswith(f".{suffix}") for suffix in DENIED_HOST_SUFFIXES
    )


def _signals(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    text = row["text"]
    metadata = row["metadata"]
    try:
        metadata_payload = json.loads(metadata)
    except (TypeError, json.JSONDecodeError) as error:
        raise FineMathFilterError("FineMath metadata differs") from error
    if not isinstance(metadata_payload, dict):
        raise FineMathFilterError("FineMath metadata differs")
    host = _host(row["url"])
    words = len(text.split())
    math_classes = sum(bool(pattern.search(text)) for pattern in MATH_PATTERNS)
    explanation_classes = sum(
        bool(pattern.search(text)) for pattern in EXPLANATION_PATTERNS
    )
    url_like_strings = len(re.findall(r"https?://|www\.", text, re.I))
    reasons = []
    if row["int_score"] != 5:
        reasons.append("upstream_score_below_5")
    if metadata_payload.get("found_math") is not True:
        reasons.append("found_math_absent")
    language_score = row["language_score"]
    if (
        isinstance(language_score, bool)
        or not isinstance(language_score, (int, float))
        or not math.isfinite(language_score)
        or language_score < 0.98
    ):
        reasons.append("language_confidence_below_0p98")
    if not host:
        reasons.append("url_host_absent_or_malformed")
    elif _host_denied(host):
        reasons.append("denied_answer_farm_host")
    if words < 160:
        reasons.append("under_160_words")
    if len(text) > 200_000:
        reasons.append("over_200k_characters")
    for name, pattern in sorted(RISK_PATTERNS.items()):
        if pattern.search(text):
            reasons.append(f"risk_pattern:{name}")
    if math_classes < 2:
        reasons.append("insufficient_distinct_math_signals")
    if explanation_classes < 2:
        reasons.append("insufficient_explanatory_structure")
    if url_like_strings > 8:
        reasons.append("excessive_embedded_urls")
    signals = {
        "url_host": host,
        "words": words,
        "characters": len(text),
        "math_signal_classes": math_classes,
        "explanation_signal_classes": explanation_classes,
        "url_like_strings": url_like_strings,
        "found_math": metadata_payload.get("found_math") is True,
        "language_score_ppm": int(language_score * 1_000_000),
    }
    return sorted(reasons), signals


def _scan(
    source: Path, *, revision: str, source_file: str, review_per_decision: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if (
        isinstance(review_per_decision, bool)
        or not isinstance(review_per_decision, int)
        or review_per_decision <= 0
    ):
        raise FineMathFilterError("review geometry differs")
    pq = _import_parquet()
    parquet = pq.ParquetFile(source)
    required = {
        "url",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
        "text",
        "int_score",
        "metadata",
        "language",
        "language_score",
        "crawl",
    }
    if not required.issubset(parquet.schema_arrow.names):
        raise FineMathFilterError("FineMath filter columns differ")
    accepted: list[dict[str, Any]] = []
    review_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        "accepted": [],
        "rejected": [],
    }
    reason_counts: Counter[str] = Counter()
    rows = 0
    exact_texts: set[str] = set()
    accepted_texts: set[str] = set()
    for batch in parquet.iter_batches(batch_size=2_048, columns=sorted(required)):
        values = batch.to_pydict()
        for raw in zip(*(values[column] for column in sorted(required)), strict=True):
            row = dict(zip(sorted(required), raw, strict=True))
            row_index = rows
            rows += 1
            if (
                not isinstance(row["text"], str)
                or not row["text"]
                or not isinstance(row["url"], str)
                or row["language"] != "en"
                or isinstance(row["int_score"], bool)
                or not isinstance(row["int_score"], int)
            ):
                raise FineMathFilterError("FineMath row differs")
            text_sha256 = hashlib.sha256(row["text"].encode()).hexdigest()
            is_exact_duplicate = text_sha256 in exact_texts
            exact_texts.add(text_sha256)
            reasons, signals = _signals(row)
            if is_exact_duplicate:
                reasons.append("exact_duplicate_text")
                reasons.sort()
            decision = "accepted" if not reasons else "rejected"
            for reason in reasons:
                reason_counts[reason] += 1
            identity_payload = {
                "dataset": DATASET,
                "revision": revision,
                "source_file": source_file,
                "row_index": row_index,
                "text_sha256": text_sha256,
                "policy_sha256": canonical_sha256(_policy()),
            }
            identity = canonical_sha256(identity_payload)
            if decision == "accepted":
                if text_sha256 in accepted_texts:
                    raise FineMathFilterError("accepted exact duplicate differs")
                accepted_texts.add(text_sha256)
                candidate = {
                    "schema": ROW_SCHEMA,
                    "identity_sha256": identity,
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
                    "upstream": {
                        "int_score": row["int_score"],
                        "metadata": json.loads(row["metadata"]),
                    },
                    "selection_signals": signals,
                    "text": row["text"],
                    "limitations": [
                        "candidate_not_benchmark_decontaminated",
                        "candidate_not_near_deduplicated",
                        "candidate_not_human_quality_approved",
                    ],
                }
                accepted.append(candidate)
            review = {
                "schema": REVIEW_SCHEMA,
                "decision": decision,
                "identity_sha256": identity,
                "row_index": row_index,
                "text_sha256": text_sha256,
                "url": row["url"],
                "rejection_reasons": reasons,
                "signals": signals,
                "text": row["text"],
            }
            rank = int.from_bytes(
                hashlib.sha256(SELECTION_SALT + bytes.fromhex(identity)).digest(),
                "big",
            )
            item = (-rank, -row_index, review)
            heap = review_heaps[decision]
            if len(heap) < review_per_decision:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    review_rows = []
    review_rows_by_decision = {}
    for decision in ("accepted", "rejected"):
        selected = [
            item[2]
            for item in sorted(
                review_heaps[decision], key=lambda item: (-item[0], -item[1])
            )
        ]
        review_rows_by_decision[decision] = len(selected)
        review_rows.extend(selected)
    summary = {
        "rows": rows,
        "unique_text_sha256": len(exact_texts),
        "accepted_rows": len(accepted),
        "rejected_rows": rows - len(accepted),
        "accepted_fraction_ppm": len(accepted) * 1_000_000 // rows,
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "review_rows_by_decision": review_rows_by_decision,
    }
    return accepted, review_rows, summary


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in rows
    )


def build_filtered_candidate(
    audit_receipt: Path,
    accepted_output: Path,
    review_output: Path,
    receipt_output: Path,
    *,
    review_per_decision: int = 64,
) -> dict[str, Any]:
    """Publish a deterministic candidate and balanced human-review packet."""

    outputs = (accepted_output, review_output, receipt_output)
    if len({path.parent.resolve() for path in outputs}) != 1:
        raise FineMathFilterError("FineMath filter outputs must share one parent")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FineMathFilterError("FineMath filter output already exists")
    try:
        audit = validate_audit(audit_receipt)
    except Exception as error:
        raise FineMathFilterError("FineMath audit validation failed") from error
    source_row = audit["source"]
    source = Path(source_row["path"])
    accepted, review_rows, summary = _scan(
        source,
        revision=source_row["revision"],
        source_file=source_row["source_file"],
        review_per_decision=review_per_decision,
    )
    accepted_bytes = _jsonl_bytes(accepted)
    review_bytes = _jsonl_bytes(review_rows)
    if not review_bytes:
        raise FineMathFilterError("FineMath filter produced no review evidence")
    accepted_output.parent.mkdir(parents=True, exist_ok=True)
    stages = [path.with_name(f".{path.name}.partial.{os.getpid()}") for path in outputs]
    try:
        for stage, encoded in zip(
            stages[:2], (accepted_bytes, review_bytes), strict=True
        ):
            with stage.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        policy = _policy()
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": (
                "filtered_candidate_not_admitted"
                if accepted
                else "filter_empty_no_candidate"
            ),
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
            "accepted_output": {
                "path": str(accepted_output.resolve()),
                "bytes": len(accepted_bytes),
                "sha256": hashlib.sha256(accepted_bytes).hexdigest(),
                "ordered_rows_sha256": canonical_sha256(accepted),
            },
            "review_output": {
                "path": str(review_output.resolve()),
                "bytes": len(review_bytes),
                "sha256": hashlib.sha256(review_bytes).hexdigest(),
                "ordered_rows_sha256": canonical_sha256(review_rows),
                "rows_per_decision": review_per_decision,
                "rows_by_decision": summary["review_rows_by_decision"],
            },
            "limitations": [
                "accepted_rows_require_independent_human_review",
                "accepted_rows_require_global_near_deduplication",
                "accepted_rows_require_benchmark_decontamination",
                "receipt_authorizes_no_training",
            ],
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        with stages[2].open("x") as handle:
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


def validate_filtered_candidate(receipt: Path) -> dict[str, Any]:
    """Recompute the exact candidate and review packet from the audited source."""

    if not receipt.is_file() or receipt.is_symlink() or receipt.stat().st_nlink != 1:
        raise FineMathFilterError("FineMath filter receipt is missing or unsafe")
    try:
        payload = json.loads(receipt.read_text())
    except json.JSONDecodeError as error:
        raise FineMathFilterError("FineMath filter receipt JSON differs") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_KEYS
        or payload.get("schema") != SCHEMA
        or payload.get("receipt_sha256")
        != canonical_sha256({k: v for k, v in payload.items() if k != "receipt_sha256"})
    ):
        raise FineMathFilterError("FineMath filter receipt differs")
    audit_row = payload["audit"]
    audit_receipt = Path(audit_row.get("path", ""))
    try:
        audit = validate_audit(audit_receipt)
    except Exception as error:
        raise FineMathFilterError("FineMath audit validation failed") from error
    if audit_row != {
        "path": str(audit_receipt.resolve()),
        "file_sha256": sha256_file(audit_receipt),
        "receipt_sha256": audit["receipt_sha256"],
    }:
        raise FineMathFilterError("FineMath audit lineage differs")
    source_row = audit["source"]
    review_descriptor = payload["review_output"]
    review_per_decision = review_descriptor.get("rows_per_decision")
    accepted, review_rows, summary = _scan(
        Path(source_row["path"]),
        revision=source_row["revision"],
        source_file=source_row["source_file"],
        review_per_decision=review_per_decision,
    )
    accepted_bytes = _jsonl_bytes(accepted)
    review_bytes = _jsonl_bytes(review_rows)
    accepted_path = Path(payload["accepted_output"].get("path", ""))
    review_path = Path(review_descriptor.get("path", ""))
    if (
        payload["status"]
        != (
            "filtered_candidate_not_admitted"
            if accepted
            else "filter_empty_no_candidate"
        )
        or payload["training_authorized"] is not False
        or payload["four_b_training_authorized"] is not False
        or payload["source"] != source_row
        or payload["policy"] != _policy()
        or payload["policy_sha256"] != canonical_sha256(_policy())
        or payload["summary"] != summary
        or not accepted_path.is_file()
        or accepted_path.is_symlink()
        or accepted_path.stat().st_nlink != 1
        or accepted_path.read_bytes() != accepted_bytes
        or payload["accepted_output"]
        != {
            "path": str(accepted_path.resolve()),
            "bytes": len(accepted_bytes),
            "sha256": hashlib.sha256(accepted_bytes).hexdigest(),
            "ordered_rows_sha256": canonical_sha256(accepted),
        }
        or not review_path.is_file()
        or review_path.is_symlink()
        or review_path.stat().st_nlink != 1
        or review_path.read_bytes() != review_bytes
        or review_descriptor
        != {
            "path": str(review_path.resolve()),
            "bytes": len(review_bytes),
            "sha256": hashlib.sha256(review_bytes).hexdigest(),
            "ordered_rows_sha256": canonical_sha256(review_rows),
            "rows_per_decision": review_per_decision,
            "rows_by_decision": summary["review_rows_by_decision"],
        }
        or payload["limitations"]
        != [
            "accepted_rows_require_independent_human_review",
            "accepted_rows_require_global_near_deduplication",
            "accepted_rows_require_benchmark_decontamination",
            "receipt_authorizes_no_training",
        ]
    ):
        raise FineMathFilterError("FineMath filter replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--audit-receipt", type=Path, required=True)
    build.add_argument("--accepted-output", type=Path, required=True)
    build.add_argument("--review-output", type=Path, required=True)
    build.add_argument("--receipt-output", type=Path, required=True)
    build.add_argument("--review-per-decision", type=int, default=64)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_filtered_candidate(
            args.audit_receipt,
            args.accepted_output,
            args.review_output,
            args.receipt_output,
            review_per_decision=args.review_per_decision,
        )
    else:
        payload = validate_filtered_candidate(args.receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "accepted_rows": payload["summary"]["accepted_rows"],
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
