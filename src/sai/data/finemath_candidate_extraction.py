"""Extract a conservative, source-bound FineMath candidate before semantic review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.decontamination import RAW_SCHEMA
from sai.data.finemath_full_census import (
    EXPECTED_SHARDS,
    REPOSITORY,
    REVISION,
    SUBSET,
    FineMathFullCensusError,
    _domain,
    _load_source_manifest,
)
from sai.data.source_lineage import parent_row_id
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-finemath-conservative-candidate-shard-v1"
AGGREGATE_SCHEMA = "sai-finemath-conservative-candidate-aggregate-v1"
LICENSE = "ODC-By-1.0"
POLICY = {
    "required_language": "en",
    "minimum_language_score_ppm": 900_000,
    "required_upstream_integer_score": 5,
    "required_found_math": True,
    "minimum_upstream_tokens": 128,
    "maximum_upstream_tokens_exclusive": 32_768,
    "minimum_text_utf8_bytes": 512,
    "required_url_host": True,
    "decision_scope": "mechanical_candidate_only_not_training_admission",
}
POLICY_SHA256 = canonical_sha256(POLICY)
RAW_FILE = "raw-candidates.jsonl"
LINEAGE_FILE = "lineage.jsonl"
RECEIPT_FILE = "receipt.json"


class FineMathCandidateError(RuntimeError):
    """A source shard, policy decision, or aggregate differs."""


def candidate_reasons(record: dict[str, Any]) -> tuple[str, ...]:
    """Return deterministic rejection reasons for one decoded FineMath row."""

    text = record.get("text")
    language_score = record.get("language_score")
    token_count = record.get("token_count")
    int_score = record.get("int_score")
    reasons = []
    if record.get("language") != POLICY["required_language"]:
        reasons.append("language_not_english")
    if (
        isinstance(language_score, bool)
        or not isinstance(language_score, (int, float))
        or not 0 <= language_score <= 1
        or int(language_score * 1_000_000) < POLICY["minimum_language_score_ppm"]
    ):
        reasons.append("language_confidence_below_0p90")
    if (
        isinstance(int_score, bool)
        or not isinstance(int_score, int)
        or int_score != POLICY["required_upstream_integer_score"]
    ):
        reasons.append("upstream_integer_score_below_5")
    metadata = record.get("metadata")
    try:
        metadata_payload = (
            json.loads(metadata) if isinstance(metadata, str) else metadata
        )
    except json.JSONDecodeError:
        metadata_payload = None
    if (
        not isinstance(metadata_payload, dict)
        or metadata_payload.get("found_math") is not True
    ):
        reasons.append("found_math_absent_or_metadata_invalid")
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, int)
        or not POLICY["minimum_upstream_tokens"]
        <= token_count
        < POLICY["maximum_upstream_tokens_exclusive"]
    ):
        reasons.append("token_count_outside_128_to_32767")
    if (
        not isinstance(text, str)
        or len(text.encode()) < POLICY["minimum_text_utf8_bytes"]
    ):
        reasons.append("text_below_512_utf8_bytes")
    url = record.get("url")
    if not isinstance(url, str) or _domain(url) in {"invalid_url", "missing_host"}:
        reasons.append("url_host_missing_or_invalid")
    return tuple(reasons)


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_json_line(handle, payload: dict[str, Any]) -> None:
    handle.write(
        (
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode()
    )


def extract_shard(
    source_root: Path, shard_index: int, output_root: Path
) -> dict[str, Any]:
    """Hash-verify and extract one complete source shard in source order."""

    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < EXPECTED_SHARDS
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise FineMathCandidateError("candidate shard geometry differs")
    try:
        source_manifest, files = _load_source_manifest(source_root)
    except FineMathFullCensusError as error:
        raise FineMathCandidateError("source manifest differs") from error
    source_descriptor = files[shard_index]
    source_path = source_root / source_descriptor["path"]
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_nlink != 1
        or source_path.stat().st_size != source_descriptor["bytes"]
        or sha256_file(source_path) != source_descriptor["sha256"]
    ):
        raise FineMathCandidateError("source shard bytes differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment dependent
        raise FineMathCandidateError("pyarrow is required") from error
    parquet = pq.ParquetFile(source_path)
    columns = (
        "url",
        "text",
        "token_count",
        "metadata",
        "int_score",
        "language",
        "language_score",
    )
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise FineMathCandidateError("FineMath schema differs")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, mode=0o700)
    scanned = accepted = accepted_bytes = accepted_tokens = 0
    rejection_reasons: Counter[str] = Counter()
    accepted_hosts: Counter[str] = Counter()
    accepted_identity_digest = hashlib.sha256()
    try:
        with (
            (temporary / RAW_FILE).open("wb") as raw_handle,
            (temporary / LINEAGE_FILE).open("wb") as lineage_handle,
        ):
            for batch in parquet.iter_batches(
                batch_size=2048, columns=list(columns), use_threads=False
            ):
                values = batch.to_pydict()
                for tuple_values in zip(
                    *(values[column] for column in columns), strict=True
                ):
                    row = dict(zip(columns, tuple_values, strict=True))
                    row_index = scanned
                    scanned += 1
                    reasons = candidate_reasons(row)
                    if reasons:
                        rejection_reasons.update(reasons)
                        continue
                    text = row["text"]
                    host = _domain(row["url"])
                    source = {
                        "dataset": f"{REPOSITORY}/{SUBSET}",
                        "revision": REVISION,
                        "source_file": source_descriptor["path"],
                        "row_index": row_index,
                        "license": LICENSE,
                        "domain": "math",
                    }
                    parent_id = parent_row_id(source)
                    raw = {"schema": RAW_SCHEMA, "text": text, "source": source}
                    identity = canonical_sha256(raw)
                    lineage = {
                        "schema": "sai-finemath-conservative-candidate-lineage-v1",
                        "candidate_identity_sha256": identity,
                        "parent_row_id": parent_id,
                        "source": {
                            "repository": REPOSITORY,
                            "revision": REVISION,
                            "subset": SUBSET,
                            "file": source_descriptor["path"],
                            "file_sha256": source_descriptor["sha256"],
                            "row_index": row_index,
                            "url": row["url"],
                            "host": host,
                        },
                        "measurements": {
                            "language_score_ppm": int(
                                row["language_score"] * 1_000_000
                            ),
                            "upstream_integer_score": row["int_score"],
                            "upstream_token_count": row["token_count"],
                            "text_utf8_bytes": len(text.encode()),
                            "found_math": True,
                        },
                        "policy_sha256": POLICY_SHA256,
                    }
                    _write_json_line(raw_handle, raw)
                    _write_json_line(lineage_handle, lineage)
                    accepted += 1
                    accepted_bytes += len(text.encode())
                    accepted_tokens += row["token_count"]
                    accepted_hosts[host] += 1
                    accepted_identity_digest.update(bytes.fromhex(identity))
        if scanned != parquet.metadata.num_rows or not accepted:
            raise FineMathCandidateError("candidate shard row coverage differs")
        payload = {
            "schema": SHARD_SCHEMA,
            "status": "complete_mechanical_candidate_not_admitted",
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "subset": SUBSET,
                "manifest_sha256": source_manifest["manifest_sha256"],
                "shard_index": shard_index,
                "file": source_descriptor["path"],
                "bytes": source_descriptor["bytes"],
                "sha256": source_descriptor["sha256"],
            },
            "policy": POLICY,
            "policy_sha256": POLICY_SHA256,
            "summary": {
                "scanned_rows": scanned,
                "accepted_rows": accepted,
                "accepted_text_utf8_bytes": accepted_bytes,
                "accepted_upstream_tokens": accepted_tokens,
                "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
                "top_accepted_hosts": [
                    {"host": host, "rows": count}
                    for host, count in sorted(
                        accepted_hosts.items(), key=lambda item: (-item[1], item[0])
                    )[:100]
                ],
                "accepted_identity_sha256": accepted_identity_digest.hexdigest(),
            },
            "raw_candidates": _descriptor(temporary / RAW_FILE),
            "lineage": _descriptor(temporary / LINEAGE_FILE),
            "full_source_shard_scanned": True,
            "global_normalized_deduplication_complete": False,
            "benchmark_decontamination_complete": False,
            "semantic_quality_review_complete": False,
            "rights_admission_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        (temporary / RECEIPT_FILE).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        os.replace(temporary, output_root)
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_shard(root: Path, shard_index: int) -> dict[str, Any]:
    receipt_path = root / f"shard_{shard_index:05d}" / RECEIPT_FILE
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FineMathCandidateError("candidate shard receipt is missing")
    payload = json.loads(receipt_path.read_text())
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SHARD_SCHEMA
        or payload.get("source", {}).get("shard_index") != shard_index
        or payload.get("policy") != POLICY
        or payload.get("policy_sha256") != POLICY_SHA256
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise FineMathCandidateError("candidate shard receipt differs")
    shard_root = receipt_path.parent
    for key in ("raw_candidates", "lineage"):
        descriptor = payload.get(key, {})
        path = shard_root / descriptor.get("file", "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise FineMathCandidateError("candidate shard output differs")
    return payload


def aggregate_shards(shard_root: Path, output_root: Path) -> dict[str, Any]:
    """Replay all shards and concatenate candidates in immutable source order."""

    if output_root.exists() or output_root.is_symlink():
        raise FineMathCandidateError("candidate aggregate output exists")
    receipts = [_load_shard(shard_root, index) for index in range(EXPECTED_SHARDS)]
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, mode=0o700)
    try:
        for filename, key in ((RAW_FILE, "raw_candidates"), (LINEAGE_FILE, "lineage")):
            with (temporary / filename).open("wb") as target:
                for index, receipt in enumerate(receipts):
                    source = shard_root / f"shard_{index:05d}" / receipt[key]["file"]
                    with source.open("rb") as handle:
                        shutil.copyfileobj(handle, target, length=1 << 20)
        summary = {
            key: sum(receipt["summary"][key] for receipt in receipts)
            for key in (
                "scanned_rows",
                "accepted_rows",
                "accepted_text_utf8_bytes",
                "accepted_upstream_tokens",
            )
        }
        if summary["scanned_rows"] != 6_699_493 or not summary["accepted_rows"]:
            raise FineMathCandidateError("candidate aggregate coverage differs")
        payload = {
            "schema": AGGREGATE_SCHEMA,
            "status": "complete_mechanical_candidate_not_admitted",
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "subset": SUBSET,
                "shards": EXPECTED_SHARDS,
            },
            "policy": POLICY,
            "policy_sha256": POLICY_SHA256,
            "summary": summary,
            "raw_candidates": _descriptor(temporary / RAW_FILE),
            "lineage": _descriptor(temporary / LINEAGE_FILE),
            "shard_receipt_sha256s": [
                receipt["receipt_sha256"] for receipt in receipts
            ],
            "full_source_population_scanned": True,
            "global_normalized_deduplication_complete": False,
            "benchmark_decontamination_complete": False,
            "semantic_quality_review_complete": False,
            "rights_admission_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        (temporary / RECEIPT_FILE).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        os.replace(temporary, output_root)
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--source-root", type=Path, required=True)
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--shard-root", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        payload = extract_shard(args.source_root, args.shard_index, args.output_root)
    else:
        payload = aggregate_shards(args.shard_root, args.output_root)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
