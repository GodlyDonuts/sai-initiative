"""Run a full, shard-parallel mechanical census of pinned FineMath-4plus."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import unicodedata
import urllib.parse
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file

REPOSITORY = "HuggingFaceTB/finemath"
REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
SUBSET = "finemath-4plus"
EXPECTED_SHARDS = 64
SOURCE_MANIFEST_SCHEMA = "sai-hf-source-manifest-v1"
SHARD_SCHEMA = "sai-finemath-full-census-shard-v1"
AGGREGATE_SCHEMA = "sai-finemath-full-census-aggregate-v1"
EXACT_HASH_FILE = "exact_content_sha256.bin"
NORMALIZED_HASH_FILE = "normalized_content_sha256.bin"
_NONZERO_MATH_FEATURE = re.compile(r'"[^"\\]+":\s*[1-9]\d*')

LANGUAGE_SCORE_BINS = (
    ("lt_0_50", 0.50),
    ("0_50_to_0_70", 0.70),
    ("0_70_to_0_80", 0.80),
    ("0_80_to_0_90", 0.90),
    ("0_90_to_0_95", 0.95),
    ("ge_0_95", float("inf")),
)
TOKEN_COUNT_BINS = (
    ("lt_64", 64),
    ("64_to_127", 128),
    ("128_to_511", 512),
    ("512_to_2047", 2048),
    ("2048_to_8191", 8192),
    ("8192_to_32767", 32768),
    ("ge_32768", 2**63),
)


class FineMathFullCensusError(RuntimeError):
    """A source shard, census receipt, or aggregate claim differs."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_source_manifest(source_root: Path) -> tuple[dict[str, Any], list[dict]]:
    path = source_root / "source-manifest.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise FineMathFullCensusError("source manifest is unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FineMathFullCensusError("source manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise FineMathFullCensusError("source manifest differs")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    files = payload.get("files")
    if (
        payload.get("schema") != SOURCE_MANIFEST_SCHEMA
        or payload.get("dataset") != REPOSITORY
        or payload.get("revision") != REVISION
        or payload.get("subset") != SUBSET
        or payload.get("status") != "download_complete_candidate_only"
        or payload.get("file_count") != EXPECTED_SHARDS
        or not isinstance(files, list)
        or len(files) != EXPECTED_SHARDS
        or payload.get("total_bytes")
        != sum(row.get("bytes", 0) for row in files if isinstance(row, dict))
        or payload.get("source_admitted") is not False
        or payload.get("training_authorized") is not False
        or payload.get("manifest_sha256") != canonical_sha256(unsigned)
    ):
        raise FineMathFullCensusError("source manifest identity differs")
    for index, row in enumerate(files):
        if (
            not isinstance(row, dict)
            or row.get("path") != f"train-{index:05d}-of-00064.parquet"
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not _is_sha256(row.get("sha256"))
        ):
            raise FineMathFullCensusError("source manifest file differs")
    return payload, files


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _bin(value: float | int, bins: tuple[tuple[str, float | int], ...]) -> str:
    for name, upper in bins:
        if value < upper:
            return name
    raise FineMathFullCensusError("census bin geometry differs")


def candidate_profiles(
    *,
    language: str,
    language_score: float,
    int_score: int,
    token_count: int,
    text_bytes: int,
) -> tuple[str, ...]:
    """Return nested measurement profiles; these are not admission decisions."""

    result = []
    if (
        language == "en"
        and language_score >= 0.70
        and int_score >= 3
        and 64 <= token_count < 32768
        and text_bytes >= 256
    ):
        result.append("broad_mechanical_profile")
    if (
        language == "en"
        and language_score >= 0.80
        and int_score >= 4
        and 128 <= token_count < 32768
        and text_bytes >= 512
    ):
        result.append("core_mechanical_profile")
    if (
        language == "en"
        and language_score >= 0.90
        and int_score >= 5
        and 128 <= token_count < 32768
        and text_bytes >= 512
    ):
        result.append("elite_mechanical_profile")
    return tuple(result)


def _write_sorted_hashes(path: Path, values: list[bytes]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FineMathFullCensusError("hash output exists")
    values.sort()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for value in values:
                if len(value) != 32:
                    raise FineMathFullCensusError("content hash differs")
                handle.write(value)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {
        "path": path.name,
        "rows": len(values),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "sorted": True,
        "digest_bytes": 32,
    }


def _domain(value: str) -> str:
    try:
        host = (urllib.parse.urlsplit(value).hostname or "").casefold().rstrip(".")
    except ValueError:
        return "invalid_url"
    if not host:
        return "missing_host"
    return host[4:] if host.startswith("www.") else host


def scan_shard(
    source_root: Path, shard_index: int, output_root: Path
) -> dict[str, Any]:
    """Hash-verify and scan one immutable Parquet shard without retaining text."""

    if (
        isinstance(shard_index, bool)
        or not 0 <= shard_index < EXPECTED_SHARDS
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise FineMathFullCensusError("shard census geometry differs")
    source_manifest, files = _load_source_manifest(source_root)
    descriptor = files[shard_index]
    source_path = source_root / descriptor["path"]
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_nlink != 1
        or source_path.stat().st_size != descriptor["bytes"]
        or sha256_file(source_path) != descriptor["sha256"]
    ):
        raise FineMathFullCensusError("source shard differs")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FineMathFullCensusError("pyarrow is required") from error
    parquet = pq.ParquetFile(source_path)
    required_columns = (
        "url",
        "text",
        "token_count",
        "char_count",
        "metadata",
        "score",
        "int_score",
        "language",
        "language_score",
    )
    if not set(required_columns).issubset(parquet.schema_arrow.names):
        raise FineMathFullCensusError("FineMath schema differs")
    rows = 0
    text_bytes_total = 0
    token_count_total = 0
    char_count_total = 0
    null_counts: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    language_score_bins: Counter[str] = Counter()
    token_count_bins: Counter[str] = Counter()
    int_scores: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    profile_text_bytes: Counter[str] = Counter()
    profile_tokens: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    exact_hashes: list[bytes] = []
    normalized_hashes: list[bytes] = []
    found_math_true = 0
    nonzero_math_feature = 0
    char_count_mismatch_rows = 0
    for batch in parquet.iter_batches(
        batch_size=2048, columns=list(required_columns), use_threads=False
    ):
        columns = batch.to_pydict()
        for values in zip(
            *(columns[column] for column in required_columns), strict=True
        ):
            record = dict(zip(required_columns, values, strict=True))
            rows += 1
            for key, value in record.items():
                if value is None:
                    null_counts[key] += 1
            text = record["text"]
            if not isinstance(text, str):
                text = ""
            encoded = text.encode()
            text_bytes = len(encoded)
            text_bytes_total += text_bytes
            exact_hashes.append(hashlib.sha256(encoded).digest())
            normalized_hashes.append(hashlib.sha256(_normalize(text).encode()).digest())
            token_count = record["token_count"]
            char_count = record["char_count"]
            language_score = record["language_score"]
            int_score = record["int_score"]
            language = record["language"]
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 0
                or isinstance(char_count, bool)
                or not isinstance(char_count, int)
                or char_count < 0
                or isinstance(int_score, bool)
                or not isinstance(int_score, int)
                or not isinstance(language_score, (int, float))
                or isinstance(language_score, bool)
                or not 0 <= language_score <= 1
                or not isinstance(language, str)
                or not language
            ):
                raise FineMathFullCensusError("FineMath row accounting differs")
            token_count_total += token_count
            char_count_total += char_count
            char_count_mismatch_rows += char_count != len(text)
            languages[language] += 1
            language_score_bins[_bin(language_score, LANGUAGE_SCORE_BINS)] += 1
            token_count_bins[_bin(token_count, TOKEN_COUNT_BINS)] += 1
            int_scores[str(int_score)] += 1
            domains[_domain(record["url"] or "")] += 1
            metadata = record["metadata"]
            if isinstance(metadata, str):
                found_math_true += '"found_math": true' in metadata
                nonzero_math_feature += bool(_NONZERO_MATH_FEATURE.search(metadata))
            for profile in candidate_profiles(
                language=language,
                language_score=float(language_score),
                int_score=int_score,
                token_count=token_count,
                text_bytes=text_bytes,
            ):
                profiles[profile] += 1
                profile_text_bytes[profile] += text_bytes
                profile_tokens[profile] += token_count
    if rows != parquet.metadata.num_rows or len(exact_hashes) != rows:
        raise FineMathFullCensusError("Parquet row coverage differs")
    output_root.mkdir(parents=True, mode=0o700)
    try:
        exact_descriptor = _write_sorted_hashes(
            output_root / EXACT_HASH_FILE, exact_hashes
        )
        normalized_descriptor = _write_sorted_hashes(
            output_root / NORMALIZED_HASH_FILE, normalized_hashes
        )
        payload = {
            "schema": SHARD_SCHEMA,
            "status": "complete_source_safe_mechanical_census_shard",
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "subset": SUBSET,
                "manifest_sha256": source_manifest["manifest_sha256"],
                "shard_index": shard_index,
                "path": descriptor["path"],
                "compressed_bytes": descriptor["bytes"],
                "sha256": descriptor["sha256"],
                "parquet_rows": parquet.metadata.num_rows,
                "parquet_row_groups": parquet.metadata.num_row_groups,
                "schema_sha256": canonical_sha256(str(parquet.schema_arrow)),
            },
            "summary": {
                "rows": rows,
                "text_utf8_bytes": text_bytes_total,
                "token_count": token_count_total,
                "declared_char_count": char_count_total,
                "char_count_mismatch_rows": char_count_mismatch_rows,
                "null_counts": dict(sorted(null_counts.items())),
                "languages": dict(sorted(languages.items())),
                "language_score_bins": dict(sorted(language_score_bins.items())),
                "token_count_bins": dict(sorted(token_count_bins.items())),
                "int_scores": dict(sorted(int_scores.items())),
                "found_math_true_rows": found_math_true,
                "nonzero_math_feature_rows": nonzero_math_feature,
                "measurement_profiles": dict(sorted(profiles.items())),
                "measurement_profile_text_utf8_bytes": dict(
                    sorted(profile_text_bytes.items())
                ),
                "measurement_profile_tokens": dict(sorted(profile_tokens.items())),
                "top_domains": [
                    {"domain": domain, "rows": count}
                    for domain, count in sorted(
                        domains.items(), key=lambda item: (-item[1], item[0])
                    )[:100]
                ],
            },
            "exact_content_hashes": exact_descriptor,
            "normalized_content_hashes": normalized_descriptor,
            "source_file_sha256_verified": True,
            "full_shard_scanned": True,
            "measurement_profiles_are_training_admissions": False,
            "source_text_persisted": False,
            "benchmark_decontamination_complete": False,
            "global_semantic_deduplication_complete": False,
            "hermes_full_population_quality_compilation_complete": False,
            "rights_admission_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        for path in output_root.iterdir():
            if path.is_file() and not path.is_symlink():
                path.unlink()
        output_root.rmdir()
        raise


def _digest_iterator(handle: BinaryIO) -> Iterator[bytes]:
    while value := handle.read(32):
        if len(value) != 32:
            raise FineMathFullCensusError("digest file is truncated")
        yield value


def summarize_digest_files(paths: list[Path]) -> dict[str, int]:
    """Merge sorted digest files and return exact global multiplicities."""

    if not paths:
        raise FineMathFullCensusError("digest population is empty")
    handles = [path.open("rb") for path in paths]
    try:
        merged = heapq.merge(*(_digest_iterator(handle) for handle in handles))
        total_rows = 0
        unique_hashes = 0
        duplicate_rows = 0
        duplicate_groups = 0
        maximum_multiplicity = 0
        current: bytes | None = None
        multiplicity = 0
        for digest in merged:
            total_rows += 1
            if digest == current:
                multiplicity += 1
                continue
            if current is not None:
                unique_hashes += 1
                duplicate_rows += max(0, multiplicity - 1)
                duplicate_groups += multiplicity > 1
                maximum_multiplicity = max(maximum_multiplicity, multiplicity)
            current = digest
            multiplicity = 1
        if current is not None:
            unique_hashes += 1
            duplicate_rows += max(0, multiplicity - 1)
            duplicate_groups += multiplicity > 1
            maximum_multiplicity = max(maximum_multiplicity, multiplicity)
        return {
            "rows": total_rows,
            "unique_hashes": unique_hashes,
            "duplicate_rows_after_keep_first": duplicate_rows,
            "duplicate_groups": duplicate_groups,
            "maximum_multiplicity": maximum_multiplicity,
        }
    finally:
        for handle in handles:
            handle.close()


def _load_shard_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise FineMathFullCensusError("shard receipt is unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FineMathFullCensusError("shard receipt is unreadable") from error
    if not isinstance(payload, dict):
        raise FineMathFullCensusError("shard receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SHARD_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("full_shard_scanned") is not True
        or payload.get("source_text_persisted") is not False
        or payload.get("training_ready") is not False
    ):
        raise FineMathFullCensusError("shard receipt identity differs")
    return payload


def _sum_nested(receipts: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: Counter[str] = Counter()
    for receipt in receipts:
        values = receipt["summary"].get(key, {})
        if not isinstance(values, dict) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise FineMathFullCensusError("census counter differs")
        result.update(values)
    return dict(sorted(result.items()))


def aggregate_census(
    source_root: Path, shards_root: Path, output_path: Path
) -> dict[str, Any]:
    """Replay every shard receipt and calculate global exact multiplicities."""

    if output_path.exists() or output_path.is_symlink():
        raise FineMathFullCensusError("aggregate output exists")
    source_manifest, files = _load_source_manifest(source_root)
    receipts = []
    exact_paths = []
    normalized_paths = []
    receipt_hashes = []
    for index, source_file in enumerate(files):
        root = shards_root / f"shard_{index:05d}"
        receipt_path = root / "receipt.json"
        receipt = _load_shard_receipt(receipt_path)
        source = receipt.get("source", {})
        if (
            source.get("shard_index") != index
            or source.get("path") != source_file["path"]
            or source.get("compressed_bytes") != source_file["bytes"]
            or source.get("sha256") != source_file["sha256"]
            or source.get("manifest_sha256") != source_manifest["manifest_sha256"]
        ):
            raise FineMathFullCensusError("aggregate shard source differs")
        for descriptor, target in (
            (receipt.get("exact_content_hashes"), exact_paths),
            (receipt.get("normalized_content_hashes"), normalized_paths),
        ):
            if not isinstance(descriptor, dict):
                raise FineMathFullCensusError("aggregate hash descriptor differs")
            path = root / descriptor.get("path", "")
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or descriptor.get("rows") != receipt["summary"].get("rows")
                or descriptor.get("bytes") != path.stat().st_size
                or descriptor.get("bytes") != descriptor.get("rows") * 32
                or descriptor.get("sha256") != sha256_file(path)
                or descriptor.get("sorted") is not True
            ):
                raise FineMathFullCensusError("aggregate hash custody differs")
            target.append(path)
        receipts.append(receipt)
        receipt_hashes.append(receipt["receipt_sha256"])
    total_rows = sum(receipt["summary"]["rows"] for receipt in receipts)
    exact = summarize_digest_files(exact_paths)
    normalized = summarize_digest_files(normalized_paths)
    if exact["rows"] != total_rows or normalized["rows"] != total_rows:
        raise FineMathFullCensusError("aggregate digest coverage differs")
    domain_counts: Counter[str] = Counter()
    for receipt in receipts:
        for row in receipt["summary"]["top_domains"]:
            domain_counts[row["domain"]] += row["rows"]
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_source_safe_full_mechanical_census",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "subset": SUBSET,
            "manifest_sha256": source_manifest["manifest_sha256"],
            "files": EXPECTED_SHARDS,
            "compressed_bytes": source_manifest["total_bytes"],
        },
        "shards": {
            "rows": len(receipts),
            "ordered_receipts_sha256": canonical_sha256(receipt_hashes),
        },
        "summary": {
            "rows": total_rows,
            "text_utf8_bytes": sum(
                receipt["summary"]["text_utf8_bytes"] for receipt in receipts
            ),
            "token_count": sum(
                receipt["summary"]["token_count"] for receipt in receipts
            ),
            "declared_char_count": sum(
                receipt["summary"]["declared_char_count"] for receipt in receipts
            ),
            "char_count_mismatch_rows": sum(
                receipt["summary"]["char_count_mismatch_rows"] for receipt in receipts
            ),
            "null_counts": _sum_nested(receipts, "null_counts"),
            "languages": _sum_nested(receipts, "languages"),
            "language_score_bins": _sum_nested(receipts, "language_score_bins"),
            "token_count_bins": _sum_nested(receipts, "token_count_bins"),
            "int_scores": _sum_nested(receipts, "int_scores"),
            "found_math_true_rows": sum(
                receipt["summary"]["found_math_true_rows"] for receipt in receipts
            ),
            "nonzero_math_feature_rows": sum(
                receipt["summary"]["nonzero_math_feature_rows"] for receipt in receipts
            ),
            "measurement_profiles": _sum_nested(receipts, "measurement_profiles"),
            "measurement_profile_text_utf8_bytes": _sum_nested(
                receipts, "measurement_profile_text_utf8_bytes"
            ),
            "measurement_profile_tokens": _sum_nested(
                receipts, "measurement_profile_tokens"
            ),
            "top_domains_from_shard_top_100": [
                {"domain": domain, "rows": count}
                for domain, count in sorted(
                    domain_counts.items(), key=lambda item: (-item[1], item[0])
                )[:250]
            ],
        },
        "exact_content_multiplicity": exact,
        "normalized_content_multiplicity": normalized,
        "all_source_file_sha256_values_verified": True,
        "all_source_rows_scanned": True,
        "global_exact_duplicate_census_complete": True,
        "global_exact_deduplication_applied": False,
        "measurement_profiles_are_training_admissions": False,
        "source_text_persisted": False,
        "benchmark_decontamination_complete": False,
        "global_semantic_deduplication_complete": False,
        "hermes_full_population_quality_compilation_complete": False,
        "rights_admission_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--aggregate-output", type=Path)
    args = parser.parse_args()
    scan_mode = args.shard_index is not None or args.output_root is not None
    aggregate_mode = args.shards_root is not None or args.aggregate_output is not None
    if scan_mode == aggregate_mode:
        raise FineMathFullCensusError("choose exactly one census mode")
    if scan_mode:
        if args.shard_index is None or args.output_root is None:
            raise FineMathFullCensusError("scan arguments differ")
        result = scan_shard(args.source_root, args.shard_index, args.output_root)
    else:
        if args.shards_root is None or args.aggregate_output is None:
            raise FineMathFullCensusError("aggregate arguments differ")
        result = aggregate_census(
            args.source_root, args.shards_root, args.aggregate_output
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
