"""Acquire a coverage-first Hermes audit population from the 8 TiB reservoir."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import CANDIDATE_SCHEMA, normalize_candidate
from sai.data.source_reservoir import MANIFEST_SCHEMA, RECEIPT_SCHEMA, SOURCE_SPECS
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-reservoir-audit-population-receipt-v1"
LINEAGE_SCHEMA = "sai-reservoir-audit-lineage-v1"
SEED = 20260823
MAX_EXCERPT_BYTES = 32 * 1024


@dataclass(frozen=True)
class Stratum:
    source_id: str
    stratum: str
    path_prefix: str
    quota: int
    source_type: str


def _stratum(
    source_id: str,
    stratum: str,
    path_prefix: str,
    quota: int,
    source_type: str,
) -> Stratum:
    return Stratum(source_id, stratum, path_prefix, quota, source_type)


FINEPDF_LANGUAGES = (
    "spa_Latn",
    "deu_Latn",
    "fra_Latn",
    "rus_Cyrl",
    "jpn_Jpan",
    "ita_Latn",
    "por_Latn",
    "nld_Latn",
    "pol_Latn",
    "cmn_Hani",
    "arb_Arab",
    "ces_Latn",
    "ukr_Cyrl",
    "ell_Grek",
    "fas_Arab",
    "tur_Latn",
    "ind_Latn",
    "tha_Thai",
    "kor_Hang",
    "heb_Hebr",
    "hin_Deva",
    "vie_Latn",
    "ben_Beng",
    "lat_Latn",
)

AUDIT_STRATA = (
    _stratum("finepdfs", "language:eng_Latn", "data/eng_Latn/", 16, "general_web"),
    *(
        _stratum(
            "finepdfs",
            f"language:{language}",
            f"data/{language}/",
            1,
            "general_web",
        )
        for language in FINEPDF_LANGUAGES
    ),
    _stratum("finemath", "finemath-3plus", "finemath-3plus/", 4, "educational_web"),
    _stratum("finemath", "finemath-4plus", "finemath-4plus/", 4, "educational_web"),
    _stratum(
        "finemath", "infiwebmath-3plus", "infiwebmath-3plus/", 4, "educational_web"
    ),
    _stratum(
        "finemath", "infiwebmath-4plus", "infiwebmath-4plus/", 4, "educational_web"
    ),
    _stratum(
        "dolma3_mix_150b",
        "wikipedia",
        "data/dolma1_7-wiki-en/",
        1,
        "reference",
    ),
    _stratum(
        "dolma3_mix_150b",
        "arxiv",
        "data/rpj-proofpile-arxiv/",
        1,
        "research_paper",
    ),
    _stratum(
        "dolma3_mix_150b",
        "mathematics",
        "data/finemath-3plus/",
        1,
        "educational_web",
    ),
    _stratum(
        "dolma3_mix_150b", "code:python", "data/stack_edu-Python/", 1, "code_repository"
    ),
    _stratum(
        "dolma3_mix_150b", "code:rust", "data/stack_edu-Rust/", 1, "code_repository"
    ),
    _stratum(
        "dolma3_mix_150b", "code:java", "data/stack_edu-Java/", 1, "code_repository"
    ),
    *(
        _stratum(
            "dolma3_mix_150b",
            f"pdf:{topic}",
            f"data/olmocr_science_pdfs-{topic}/",
            1,
            "reference",
        )
        for topic in (
            "art_and_design",
            "crime_and_law",
            "health",
            "history_and_geography",
            "literature",
            "religion",
            "science_math_and_technology",
            "finance_and_business",
            "industrial",
            "home_and_hobbies",
            "social_life",
            "food_and_dining",
        )
    ),
    *(
        _stratum(
            "dolma3_mix_150b",
            f"web:{topic}",
            f"data/common_crawl-{topic}-0019/",
            1,
            "general_web",
        )
        for topic in (
            "science_math_and_technology",
            "history_and_geography",
            "literature",
            "crime_and_law",
            "health",
            "home_and_hobbies",
        )
    ),
    _stratum(
        "smollm_corpus",
        "fineweb-edu-dedup",
        "fineweb-edu-dedup/",
        8,
        "educational_web",
    ),
    _stratum("smollm_corpus", "cosmopedia-v2", "cosmopedia-v2/", 8, "synthetic"),
    _stratum("open_web_math", "all", "data/", 8, "educational_web"),
    *(
        _stratum(
            "fineweb_edu_fill",
            f"crawl:{year}",
            f"data/CC-MAIN-{year}",
            4,
            "educational_web",
        )
        for year in range(2013, 2019)
    ),
)

EXPECTED_ROWS = sum(stratum.quota for stratum in AUDIT_STRATA)
if EXPECTED_ROWS != 128:  # pragma: no cover - import-time frozen contract
    raise RuntimeError("reservoir audit geometry differs")


class ReservoirAuditError(RuntimeError):
    """The reservoir identity, sample geometry, or acquired row differs."""


def _selection_key(stratum: Stratum, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{SEED}:{stratum.source_id}:{stratum.stratum}:"
            f"{row['path']}:{row['sha256']}"
        ).encode()
    ).hexdigest()


def _load_reservoir(manifest_path: Path, receipt_path: Path) -> list[dict[str, Any]]:
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise ReservoirAuditError("reservoir evidence is missing or unsafe")
    try:
        rows = [json.loads(line) for line in manifest_path.open()]
        receipt_lines = [json.loads(line) for line in receipt_path.open()]
    except (OSError, json.JSONDecodeError) as error:
        raise ReservoirAuditError("reservoir evidence cannot be decoded") from error
    if not rows or len(receipt_lines) != 1:
        raise ReservoirAuditError("reservoir evidence is empty or duplicated")
    receipt = receipt_lines[0]
    expected_sources = {spec.source_id: spec for spec in SOURCE_SPECS}
    identities = set()
    for ordinal, row in enumerate(rows):
        source = expected_sources.get(row.get("source_id"))
        if (
            source is None
            or row.get("schema") != MANIFEST_SCHEMA
            or row.get("repository") != source.repository
            or row.get("revision") != source.revision
            or row.get("license") != source.license
            or row.get("access") != source.access
            or row.get("epistemic_function") != source.epistemic_function
            or row.get("ordinal") != ordinal
            or row.get("raw_source_is_training_ready") is not False
            or not isinstance(row.get("path"), str)
            or not row["path"].endswith(source.suffix)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or (row["repository"], row["path"]) in identities
        ):
            raise ReservoirAuditError("reservoir manifest row differs")
        identities.add((row["repository"], row["path"]))
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("source_reservoir_complete") is not True
        or receipt.get("training_ready") is not False
        or receipt.get("selected_files") != len(rows)
        or receipt.get("selected_bytes") != sum(row["bytes"] for row in rows)
        or receipt.get("manifest", {}).get("path") != manifest_path.name
        or receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path)
        or receipt.get("manifest", {}).get("ordered_rows_sha256")
        != canonical_sha256(rows)
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise ReservoirAuditError("reservoir receipt differs")
    return rows


def build_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select immutable parent files from every frozen coverage stratum."""

    plan = []
    for stratum in AUDIT_STRATA:
        matches = [
            row
            for row in rows
            if row["source_id"] == stratum.source_id
            and row["path"].startswith(stratum.path_prefix)
        ]
        ranked = sorted(
            matches, key=lambda row: (_selection_key(stratum, row), row["path"])
        )
        if len(ranked) < stratum.quota:
            raise ReservoirAuditError(
                f"audit stratum is underfilled: {stratum.stratum}"
            )
        for row in ranked[: stratum.quota]:
            plan.append(
                {
                    "ordinal": len(plan),
                    "source_id": row["source_id"],
                    "stratum": stratum.stratum,
                    "source_type": stratum.source_type,
                    "repository": row["repository"],
                    "revision": row["revision"],
                    "license": row["license"],
                    "access": row["access"],
                    "path": row["path"],
                    "parent_file_bytes": row["bytes"],
                    "parent_file_sha256": row["sha256"],
                    "selection_key": _selection_key(stratum, row),
                }
            )
    if len(plan) != EXPECTED_ROWS or len({row["selection_key"] for row in plan}) != len(
        plan
    ):
        raise ReservoirAuditError("audit plan geometry or identity differs")
    return plan


def _excerpt(text: str) -> tuple[str, str]:
    text = text.strip()
    encoded = text.encode("utf-8")
    if len(encoded) < 200:
        raise ReservoirAuditError("selected source row is too short")
    if len(encoded) <= MAX_EXCERPT_BYTES:
        return text, "complete"
    separator = "\n\n[... source excerpt gap ...]\n\n"
    segment_bytes = (MAX_EXCERPT_BYTES - 2 * len(separator.encode())) // 3
    middle_start = max(0, (len(encoded) - segment_bytes) // 2)
    parts = (
        encoded[:segment_bytes].decode("utf-8", errors="ignore"),
        encoded[middle_start : middle_start + segment_bytes].decode(
            "utf-8", errors="ignore"
        ),
        encoded[-segment_bytes:].decode("utf-8", errors="ignore"),
    )
    excerpt = separator.join(parts).strip()
    while len(excerpt.encode()) > MAX_EXCERPT_BYTES:
        excerpt = excerpt[:-1]
    return excerpt, "utf8_beginning_middle_end_32768"


def _parquet_row(plan: dict[str, Any], token: str) -> dict[str, Any]:
    try:
        import fsspec
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_url
    except ImportError as error:
        raise ReservoirAuditError(
            "fsspec, pyarrow, and huggingface_hub are required"
        ) from error
    url = hf_hub_url(
        plan["repository"],
        plan["path"],
        repo_type="dataset",
        revision=plan["revision"],
    )
    text_column = plan.get("text_column", "text")
    if not isinstance(text_column, str) or not text_column:
        raise ReservoirAuditError("remote parquet text column differs")
    filesystem = fsspec.filesystem("http", headers={"Authorization": f"Bearer {token}"})
    try:
        with filesystem.open(url, "rb", block_size=8 << 20) as handle:
            if handle.size != plan["parent_file_bytes"]:
                raise ReservoirAuditError("remote parquet size differs")
            source = parquet.ParquetFile(handle)
            if (
                text_column not in source.schema_arrow.names
                or source.metadata.num_row_groups <= 0
            ):
                raise ReservoirAuditError("remote parquet schema differs")
            group_index = (
                int(
                    hashlib.sha256(
                        f"{plan['selection_key']}:row-group".encode()
                    ).hexdigest(),
                    16,
                )
                % source.metadata.num_row_groups
            )
            table = source.read_row_group(
                group_index, columns=[text_column], use_threads=False
            )
            if table.num_rows <= 0:
                raise ReservoirAuditError("selected parquet row group is empty")
            start = (
                int(
                    hashlib.sha256(f"{plan['selection_key']}:row".encode()).hexdigest(),
                    16,
                )
                % table.num_rows
            )
            for offset in range(table.num_rows):
                row_index = (start + offset) % table.num_rows
                text = table[text_column][row_index].as_py()
                if isinstance(text, str) and len(text.strip().encode("utf-8")) >= 200:
                    global_index = (
                        sum(
                            source.metadata.row_group(index).num_rows
                            for index in range(group_index)
                        )
                        + row_index
                    )
                    return {
                        "text": text.strip(),
                        "locator": {
                            "format": "parquet",
                            "row_group": group_index,
                            "row_in_group": row_index,
                            "row_index": global_index,
                        },
                        "full_file_content_verified": False,
                    }
    except ReservoirAuditError:
        raise
    except Exception as error:
        raise ReservoirAuditError("remote parquet acquisition failed") from error
    raise ReservoirAuditError("selected parquet row group has no usable text")


def _zstd_row(plan: dict[str, Any], token: str) -> dict[str, Any]:
    try:
        import zstandard
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ReservoirAuditError(
            "zstandard and huggingface_hub are required"
        ) from error
    path = Path(
        hf_hub_download(
            plan["repository"],
            plan["path"],
            repo_type="dataset",
            revision=plan["revision"],
            token=token,
        )
    )
    if (
        path.stat().st_size != plan["parent_file_bytes"]
        or sha256_file(path) != plan["parent_file_sha256"]
    ):
        raise ReservoirAuditError("downloaded zstd file identity differs")
    selected: tuple[str, int, str, str | None] | None = None
    try:
        with path.open("rb") as compressed:
            reader = zstandard.ZstdDecompressor().stream_reader(compressed)
            with io.TextIOWrapper(reader, encoding="utf-8") as decoded:
                for line_number, line in enumerate(decoded, start=1):
                    row = json.loads(line)
                    text = row.get("text") if isinstance(row, dict) else None
                    if (
                        not isinstance(text, str)
                        or len(text.strip().encode("utf-8")) < 200
                    ):
                        continue
                    native_id = row.get("id")
                    if not isinstance(native_id, str):
                        native_id = None
                    selection_material = (
                        f"{plan['selection_key']}:{line_number}:{native_id or ''}"
                    )
                    key = hashlib.sha256(selection_material.encode()).hexdigest()
                    if selected is None or key < selected[0]:
                        selected = (key, line_number, text.strip(), native_id)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReservoirAuditError("zstd source row differs") from error
    if selected is None:
        raise ReservoirAuditError("zstd shard has no usable text")
    _, line_number, text, native_id = selected
    return {
        "text": text,
        "locator": {
            "format": "jsonl.zst",
            "line_number": line_number,
            "native_id": native_id,
        },
        "full_file_content_verified": True,
    }


def _acquire_one(plan: dict[str, Any], token: str) -> dict[str, Any]:
    if plan["path"].endswith(".parquet"):
        return _parquet_row(plan, token)
    if plan["path"].endswith(".jsonl.zst"):
        return _zstd_row(plan, token)
    raise ReservoirAuditError("audit source format differs")


def _candidate_and_lineage(
    plan: dict[str, Any], acquired: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_text = acquired.get("text")
    locator = acquired.get("locator")
    if (
        not isinstance(full_text, str)
        or not isinstance(locator, dict)
        or not isinstance(acquired.get("full_file_content_verified"), bool)
    ):
        raise ReservoirAuditError("acquired source row fields differ")
    excerpt, excerpt_method = _excerpt(full_text)
    full_text_sha256 = hashlib.sha256(full_text.encode()).hexdigest()
    row_locator = {
        "repository": plan["repository"],
        "revision": plan["revision"],
        "path": plan["path"],
        "parent_file_sha256": plan["parent_file_sha256"],
        "locator": locator,
    }
    if "text_column" in plan:
        row_locator["text_column"] = plan["text_column"]
    provenance = canonical_sha256(
        {
            "row_locator": row_locator,
            "full_text_sha256": full_text_sha256,
            "excerpt_method": excerpt_method,
            "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        }
    )
    row_id = canonical_sha256(row_locator)
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text": excerpt,
        "source": {
            "dataset": plan["repository"],
            "revision": plan["revision"],
            "row_id": row_id,
            "license": plan["license"],
            "source_type": plan["source_type"],
        },
        "source_content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "provenance_sha256": provenance,
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    candidate = normalize_candidate(candidate)
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "ordinal": plan["ordinal"],
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_id": plan["source_id"],
        "stratum": plan["stratum"],
        "selection_key": plan["selection_key"],
        "repository": plan["repository"],
        "revision": plan["revision"],
        "license": plan["license"],
        "access": plan["access"],
        "path": plan["path"],
        "parent_file_bytes": plan["parent_file_bytes"],
        "parent_file_sha256": plan["parent_file_sha256"],
        "locator": locator,
        "full_file_content_verified": acquired["full_file_content_verified"],
        "full_text_bytes": len(full_text.encode()),
        "full_text_sha256": full_text_sha256,
        "excerpt_method": excerpt_method,
        "excerpt_bytes": len(excerpt.encode()),
        "excerpt_sha256": candidate["source_content_sha256"],
        "raw_source_is_training_ready": False,
    }
    if "text_column" in plan:
        lineage["text_column"] = plan["text_column"]
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return candidate, lineage


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    output_root: Path,
    *,
    token: str,
    acquire_function: Callable[[dict[str, Any], str], dict[str, Any]] = _acquire_one,
) -> dict[str, Any]:
    """Acquire and seal all 128 coverage-first compiler candidates."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise ReservoirAuditError("audit credential or output boundary differs")
    rows = _load_reservoir(manifest_path, reservoir_receipt_path)
    plan = build_plan(rows)
    candidates = []
    lineage = []
    for index, item in enumerate(plan, start=1):
        candidate, source_lineage = _candidate_and_lineage(
            item, acquire_function(item, token)
        )
        candidates.append(candidate)
        lineage.append(source_lineage)
        if index % 8 == 0 or index == len(plan):
            print(
                json.dumps(
                    {
                        "event": "reservoir_audit_acquisition_progress",
                        "acquired": index,
                        "remaining": len(plan) - index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(identities) != len(set(identities)) or len(candidates) != EXPECTED_ROWS:
        raise ReservoirAuditError("acquired candidate identities differ")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise ReservoirAuditError("audit temporary output already exists")
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        by_source = Counter(row["source_id"] for row in lineage)
        by_stratum = Counter(f"{row['source_id']}::{row['stratum']}" for row in lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "coverage_first_frozen_strata_then_lowest_sha256_parent_files_"
                "and_deterministic_source_rows"
            ),
            "statistically_representative": False,
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_sha256": sha256_file(reservoir_receipt_path),
                "selected_files": len(rows),
                "selected_bytes": sum(row["bytes"] for row in rows),
            },
            "population": {
                "path": candidate_path.name,
                "rows": len(candidates),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage),
            },
            "by_source": dict(sorted(by_source.items())),
            "by_stratum": dict(sorted(by_stratum.items())),
            "range_read_parent_files": sum(
                not row["full_file_content_verified"] for row in lineage
            ),
            "fully_verified_parent_files": sum(
                row["full_file_content_verified"] for row in lineage
            ),
            "hermes_judgments_complete": False,
            "training_ready": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_jsonl(receipt_path, [receipt])
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservoir-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    receipt = build_population(
        args.manifest,
        args.reservoir_receipt,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
