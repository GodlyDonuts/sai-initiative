"""Audit exact FineMath Parquet shards before any Sai admission decision."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-finemath-shard-audit-v1"
SAMPLE_SCHEMA = "sai-finemath-review-sample-v1"
DATASET = "HuggingFaceTB/finemath"
SELECTION_SALT = b"sai-finemath-review-v1"
EXPECTED_COLUMNS = {
    "url",
    "fetch_time",
    "content_mime_type",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "text",
    "token_count",
    "char_count",
    "metadata",
    "score",
    "int_score",
    "crawl",
    "snapshot_type",
    "language",
    "language_score",
}
RISK_PATTERNS = {
    "answer_key_or_homework_site": re.compile(
        r"answer key|homework answer|chegg|course hero", re.IGNORECASE
    ),
    "casino_or_betting": re.compile(r"casino|sportsbook|betting|poker", re.IGNORECASE),
    "essay_service": re.compile(
        r"write (?:my |a )?paper|essay (?:writer|writing|service)|masterpapers",
        re.IGNORECASE,
    ),
    "seo_or_marketing": re.compile(
        r"click here|buy now|free shipping|contact us today", re.IGNORECASE
    ),
}
ADDITIONAL_RISK_SIGNALS = {"malformed_url", "under_80_words"}
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "source",
    "policy",
    "policy_sha256",
    "summary",
    "review_sample",
    "claims",
    "receipt_sha256",
}


class FineMathAuditError(RuntimeError):
    """The source shard, audit evidence, or review sample differs."""


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FineMathAuditError(f"{label} differs")
    if not any(bytes.fromhex(value)):
        raise FineMathAuditError(f"{label} is a placeholder")
    return value


def _revision(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise FineMathAuditError("FineMath revision differs")
    if not any(bytes.fromhex(value)):
        raise FineMathAuditError("FineMath revision is a placeholder")
    return value


def _source_file(value: str) -> str:
    path = Path(value)
    if (
        not isinstance(value, str)
        or path.is_absolute()
        or ".." in path.parts
        or not re.fullmatch(
            r"finemath-4plus/train-[0-9]{5}-of-[0-9]{5}\.parquet", value
        )
    ):
        raise FineMathAuditError("FineMath source file differs")
    return value


def _safe_regular(path: Path, label: str) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise FineMathAuditError(f"{label} is missing or unsafe") from error
    try:
        value = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or value.st_size <= 0:
        raise FineMathAuditError(f"{label} is missing or unsafe")
    return value


def _import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FineMathAuditError("PyArrow is required for FineMath audit") from error
    return pq


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        raise FineMathAuditError("FineMath shard contains no rows")
    values.sort()
    return {
        name: values[int((len(values) - 1) * fraction)]
        for name, fraction in (
            ("p10", 0.10),
            ("p50", 0.50),
            ("p90", 0.90),
            ("p99", 0.99),
        )
    }


def _policy() -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "source_subset": "finemath-4plus",
        "exact_parquet_columns": sorted(EXPECTED_COLUMNS),
        "accepted_upstream_integer_scores": [4, 5],
        "required_language": "en",
        "risk_patterns": {
            name: pattern.pattern for name, pattern in sorted(RISK_PATTERNS.items())
        },
        "risk_patterns_case_insensitive": True,
        "under_word_count_risk_threshold": 80,
        "top_url_host_count": 25,
        "review_rank": "sha256_sai_finemath_review_v1_plus_text_sha256",
        "risk_counts_are_nonexclusive_literal_lower_bounds": True,
    }


def _scan(
    path: Path, *, sample_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size <= 0
    ):
        raise FineMathAuditError("FineMath sample size differs")
    pq = _import_parquet()
    parquet = pq.ParquetFile(path)
    if set(parquet.schema_arrow.names) != EXPECTED_COLUMNS:
        raise FineMathAuditError("FineMath Parquet columns differ")

    rows = 0
    seen: set[bytes] = set()
    exact_duplicate_texts = 0
    score_counts: Counter[int] = Counter()
    found_math = 0
    empty_host = 0
    host_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    character_counts: list[int] = []
    samples: list[tuple[int, int, dict[str, Any]]] = []
    columns = [
        "url",
        "text",
        "int_score",
        "metadata",
        "token_count",
        "char_count",
        "language",
        "language_score",
    ]
    for batch in parquet.iter_batches(batch_size=2_048, columns=columns):
        values = batch.to_pydict()
        for row_values in zip(*(values[column] for column in columns), strict=True):
            row = dict(zip(columns, row_values, strict=True))
            row_index = rows
            rows += 1
            text = row["text"]
            url = row["url"]
            int_score = row["int_score"]
            metadata = row["metadata"]
            if (
                not isinstance(text, str)
                or not text
                or not isinstance(url, str)
                or isinstance(int_score, bool)
                or not isinstance(int_score, int)
                or int_score not in {4, 5}
                or isinstance(row["token_count"], bool)
                or not isinstance(row["token_count"], int)
                or row["token_count"] <= 0
                or row["char_count"] != len(text)
                or row["language"] != "en"
                or isinstance(row["language_score"], bool)
                or not isinstance(row["language_score"], (int, float))
                or not math.isfinite(row["language_score"])
                or not 0.0 <= row["language_score"] <= 1.0
            ):
                raise FineMathAuditError("FineMath row fields differ")
            try:
                metadata_payload = json.loads(metadata)
            except (TypeError, json.JSONDecodeError) as error:
                raise FineMathAuditError("FineMath metadata differs") from error
            if not isinstance(metadata_payload, dict):
                raise FineMathAuditError("FineMath metadata differs")

            text_sha256 = hashlib.sha256(text.encode()).digest()
            if text_sha256 in seen:
                exact_duplicate_texts += 1
            seen.add(text_sha256)
            score_counts[int_score] += 1
            found_math += metadata_payload.get("found_math") is True
            character_counts.append(len(text))
            try:
                host = (urlsplit(url).hostname or "").casefold()
            except ValueError:
                host = ""
                risk_counts["malformed_url"] += 1
                risk_matches = ["malformed_url"]
            else:
                risk_matches = []
            if host.startswith("www."):
                host = host[4:]
            if not host:
                empty_host += 1
            host_counts[host] += 1
            for name, pattern in RISK_PATTERNS.items():
                if pattern.search(text):
                    risk_counts[name] += 1
                    risk_matches.append(name)
            if len(text.split()) < 80:
                risk_counts["under_80_words"] += 1
                risk_matches.append("under_80_words")

            rank = int.from_bytes(
                hashlib.sha256(SELECTION_SALT + text_sha256).digest(), "big"
            )
            sample = {
                "schema": SAMPLE_SCHEMA,
                "row_index": row_index,
                "url": url,
                "url_host": host,
                "int_score": int_score,
                "found_math": metadata_payload.get("found_math") is True,
                "text_sha256": text_sha256.hex(),
                "risk_signals": sorted(risk_matches),
                "text": text,
            }
            item = (-rank, row_index, sample)
            if len(samples) < sample_size:
                heapq.heappush(samples, item)
            elif item > samples[0]:
                heapq.heapreplace(samples, item)

    selected = [item[2] for item in sorted(samples, key=lambda item: -item[0])]
    summary = {
        "rows": rows,
        "unique_text_sha256": len(seen),
        "exact_duplicate_texts": exact_duplicate_texts,
        "score_counts": {str(score): score_counts[score] for score in (4, 5)},
        "found_math_rows": found_math,
        "found_math_fraction_ppm": found_math * 1_000_000 // rows,
        "empty_url_host_rows": empty_host,
        "character_count_quantiles": _quantiles(character_counts),
        "risk_signal_lower_bounds": {
            key: risk_counts[key]
            for key in sorted(set(RISK_PATTERNS) | ADDITIONAL_RISK_SIGNALS)
        },
        "top_url_hosts": [
            {"host": host, "rows": count} for host, count in host_counts.most_common(25)
        ],
    }
    return summary, selected


def audit_shard(
    source: Path,
    *,
    revision: str,
    source_file: str,
    expected_bytes: int,
    expected_sha256: str,
    sample_size: int,
    sample_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Audit one exact shard and atomically publish evidence and review rows."""

    revision = _revision(revision)
    source_file = _source_file(source_file)
    expected_sha256 = _sha256(expected_sha256, "FineMath source SHA-256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise FineMathAuditError("FineMath source bytes differ")
    source_stat = _safe_regular(source, "FineMath source")
    if source_stat.st_size != expected_bytes or sha256_file(source) != expected_sha256:
        raise FineMathAuditError("FineMath source content differs")
    if any(
        path.exists() or path.is_symlink() for path in (sample_output, receipt_output)
    ):
        raise FineMathAuditError("FineMath audit output already exists")
    if sample_output.parent != receipt_output.parent:
        raise FineMathAuditError("FineMath audit outputs must share one parent")

    summary, samples = _scan(source, sample_size=sample_size)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"partial.{os.getpid()}"
    sample_stage = sample_output.with_name(f".{sample_output.name}.{suffix}")
    receipt_stage = receipt_output.with_name(f".{receipt_output.name}.{suffix}")
    try:
        with sample_stage.open("x") as handle:
            for row in samples:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        policy = _policy()
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "audited_candidate_not_admitted",
            "training_authorized": False,
            "four_b_training_authorized": False,
            "source": {
                "dataset": DATASET,
                "revision": revision,
                "source_file": source_file,
                "path": str(source.resolve()),
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            },
            "policy": policy,
            "policy_sha256": canonical_sha256(policy),
            "summary": summary,
            "review_sample": {
                "schema": SAMPLE_SCHEMA,
                "selection": "lowest_sha256_sai_finemath_review_v1_plus_text_sha256",
                "requested_rows": sample_size,
                "rows": len(samples),
                "path": str(sample_output.resolve()),
                "bytes": sample_stage.stat().st_size,
                "sha256": sha256_file(sample_stage),
            },
            "claims": {
                "risk_counts_are_literal_lower_bounds_not_prevalence": True,
                "upstream_score_is_not_sai_admission": True,
                "audit_authorizes_filter_design_only": True,
            },
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        with receipt_stage.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(sample_stage, sample_output)
        os.replace(receipt_stage, receipt_output)
        return payload
    except BaseException:
        sample_stage.unlink(missing_ok=True)
        receipt_stage.unlink(missing_ok=True)
        raise


def validate_audit(receipt: Path) -> dict[str, Any]:
    """Reopen an audit receipt, source shard, and deterministic review sample."""

    _safe_regular(receipt, "FineMath audit receipt")
    try:
        payload = json.loads(receipt.read_text())
    except json.JSONDecodeError as error:
        raise FineMathAuditError("FineMath audit receipt differs") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_KEYS
        or payload.get("schema") != SCHEMA
    ):
        raise FineMathAuditError("FineMath audit receipt differs")
    receipt_sha256 = _sha256(payload.get("receipt_sha256"), "audit receipt SHA-256")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if receipt_sha256 != canonical_sha256(unsigned):
        raise FineMathAuditError("FineMath audit receipt hash differs")
    policy = _policy()
    if payload.get("policy") != policy or payload.get(
        "policy_sha256"
    ) != canonical_sha256(policy):
        raise FineMathAuditError("FineMath audit policy differs")
    source_row = payload.get("source")
    if not isinstance(source_row, dict) or set(source_row) != {
        "dataset",
        "revision",
        "source_file",
        "path",
        "bytes",
        "sha256",
    }:
        raise FineMathAuditError("FineMath audit source differs")
    if source_row["dataset"] != DATASET:
        raise FineMathAuditError("FineMath audit source differs")
    _revision(source_row["revision"])
    _source_file(source_row["source_file"])
    source = Path(source_row["path"])
    source_stat = _safe_regular(source, "FineMath source")
    if source_stat.st_size != source_row["bytes"] or sha256_file(source) != _sha256(
        source_row["sha256"], "source SHA-256"
    ):
        raise FineMathAuditError("FineMath source content differs")
    review = payload.get("review_sample")
    if not isinstance(review, dict) or set(review) != {
        "schema",
        "selection",
        "requested_rows",
        "rows",
        "path",
        "bytes",
        "sha256",
    }:
        raise FineMathAuditError("FineMath review sample differs")
    sample = Path(review["path"])
    sample_stat = _safe_regular(sample, "FineMath review sample")
    if (
        review["schema"] != SAMPLE_SCHEMA
        or review["selection"]
        != "lowest_sha256_sai_finemath_review_v1_plus_text_sha256"
        or sample_stat.st_size != review["bytes"]
        or sha256_file(sample) != _sha256(review["sha256"], "sample SHA-256")
    ):
        raise FineMathAuditError("FineMath review sample differs")
    summary, samples = _scan(source, sample_size=review["requested_rows"])
    expected_sample = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in samples
    ).encode()
    if (
        payload.get("status") != "audited_candidate_not_admitted"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("summary") != summary
        or review["rows"] != len(samples)
        or sample.read_bytes() != expected_sample
        or payload.get("claims")
        != {
            "risk_counts_are_literal_lower_bounds_not_prevalence": True,
            "upstream_score_is_not_sai_admission": True,
            "audit_authorizes_filter_design_only": True,
        }
    ):
        raise FineMathAuditError("FineMath audit replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--source", type=Path, required=True)
    audit.add_argument("--revision", required=True)
    audit.add_argument("--source-file", required=True)
    audit.add_argument("--expected-bytes", type=int, required=True)
    audit.add_argument("--expected-sha256", required=True)
    audit.add_argument("--sample-size", type=int, default=20)
    audit.add_argument("--sample-output", type=Path, required=True)
    audit.add_argument("--receipt-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        payload = audit_shard(
            args.source,
            revision=args.revision,
            source_file=args.source_file,
            expected_bytes=args.expected_bytes,
            expected_sha256=args.expected_sha256,
            sample_size=args.sample_size,
            sample_output=args.sample_output,
            receipt_output=args.receipt_output,
        )
    else:
        payload = validate_audit(args.receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
