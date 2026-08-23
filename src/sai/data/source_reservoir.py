"""Freeze an exact Hugging Face source reservoir of at least eight TiB."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file

TARGET_BYTES = 8 * 1024**4
MANIFEST_SCHEMA = "sai-source-reservoir-file-v1"
RECEIPT_SCHEMA = "sai-source-reservoir-receipt-v1"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    repository: str
    revision: str
    license: str
    access: str
    epistemic_function: str
    suffix: str
    fill_source: bool = False


SOURCE_SPECS = (
    SourceSpec(
        "finepdfs",
        "HuggingFaceFW/finepdfs",
        "220bac3acbf07789502c621d2d33952f51ac7f86",
        "odc-by-1.0",
        "public",
        "global_reality_anchor_pdfs",
        ".parquet",
    ),
    SourceSpec(
        "institutional_books",
        "institutional/institutional-books-hl-enriched-text",
        "92fcdf938eb87edfe0fbf09d4f692fa3d8bc9bcd",
        "pinned_early_access_terms",
        "gated_authenticated_reference_only",
        "global_books_human_expression_and_knowledge",
        ".parquet",
    ),
    SourceSpec(
        "finemath",
        "HuggingFaceTB/finemath",
        "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "odc-by-1.0",
        "public",
        "mathematics_reality_anchor",
        ".parquet",
    ),
    SourceSpec(
        "dolma3_mix_150b",
        "allenai/dolma3_mix-150B",
        "afa92bfb22366821c5e6cd427cdd036b34b713ef",
        "odc-by-1.0",
        "public",
        "broad_multidomain_reality_anchor",
        ".jsonl.zst",
    ),
    SourceSpec(
        "smollm_corpus",
        "HuggingFaceTB/smollm-corpus",
        "3ba9d605774198c5868892d7a8deda78031a781f",
        "odc-by-1.0",
        "public",
        "curated_education_code_and_synthetic_textbooks",
        ".parquet",
    ),
    SourceSpec(
        "open_web_math",
        "open-web-math/open-web-math",
        "fde8ef8de2300f5e778f56261843dab89f230815",
        "upstream_terms",
        "public",
        "mathematical_exposition",
        ".parquet",
    ),
    SourceSpec(
        "fineweb_edu_fill",
        "HuggingFaceFW/fineweb-edu",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "odc-by-1.0",
        "public",
        "broad_educational_web_fill",
        ".parquet",
        True,
    ),
)


class SourceReservoirError(RuntimeError):
    """A source revision, file identity, or byte target differs."""


def select_reservoir(
    inventories: dict[str, list[dict[str, Any]]], target_bytes: int = TARGET_BYTES
) -> list[dict[str, Any]]:
    """Select every specialist source, then the minimum deterministic web fill."""

    if (
        isinstance(target_bytes, bool)
        or not isinstance(target_bytes, int)
        or target_bytes <= 0
        or set(inventories) != {spec.source_id for spec in SOURCE_SPECS}
    ):
        raise SourceReservoirError("source reservoir inputs differ")
    selected: list[dict[str, Any]] = []
    filler: list[dict[str, Any]] | None = None
    identities: set[tuple[str, str]] = set()
    for spec in SOURCE_SPECS:
        rows = sorted(inventories[spec.source_id], key=lambda row: row["path"])
        if not rows:
            raise SourceReservoirError(f"{spec.source_id} inventory is empty")
        normalized = []
        for row in rows:
            if set(row) != {"path", "bytes", "sha256"}:
                raise SourceReservoirError("source file fields differ")
            path, size, digest = row["path"], row["bytes"], row["sha256"]
            if (
                not isinstance(path, str)
                or not path.endswith(spec.suffix)
                or path.startswith("/")
                or ".." in Path(path).parts
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or (spec.repository, path) in identities
            ):
                raise SourceReservoirError("source file identity differs")
            identities.add((spec.repository, path))
            normalized.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "source_id": spec.source_id,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "license": spec.license,
                    "access": spec.access,
                    "epistemic_function": spec.epistemic_function,
                    "path": path,
                    "bytes": size,
                    "sha256": digest,
                    "raw_source_is_training_ready": False,
                }
            )
        if spec.fill_source:
            if filler is not None:
                raise SourceReservoirError("multiple fill sources are forbidden")
            filler = normalized
        else:
            selected.extend(normalized)
    if filler is None:
        raise SourceReservoirError("source reservoir fill inventory is absent")
    total = sum(row["bytes"] for row in selected)
    for row in filler:
        if total >= target_bytes:
            break
        selected.append(row)
        total += row["bytes"]
    if total < target_bytes:
        raise SourceReservoirError("source reservoir cannot reach its byte target")
    return [dict(row, ordinal=index) for index, row in enumerate(selected)]


def _fetch_inventories(token: str) -> dict[str, list[dict[str, Any]]]:
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    except ImportError as error:
        raise SourceReservoirError("huggingface_hub is required") from error
    api = HfApi(token=token)
    inventories = {}
    for spec in SOURCE_SPECS:
        info = api.dataset_info(
            spec.repository, revision=spec.revision, files_metadata=True
        )
        if info.sha != spec.revision:
            raise SourceReservoirError(f"{spec.source_id} revision differs")
        siblings = [
            sibling
            for sibling in info.siblings
            if sibling.rfilename.endswith(spec.suffix)
        ]
        if not siblings:
            raise SourceReservoirError(f"{spec.source_id} data files are absent")
        probe = siblings[0]
        metadata = get_hf_file_metadata(
            hf_hub_url(
                spec.repository,
                probe.rfilename,
                repo_type="dataset",
                revision=spec.revision,
            ),
            token=token,
        )
        if metadata.size != probe.size:
            raise SourceReservoirError(f"{spec.source_id} access probe differs")
        rows = []
        for sibling in siblings:
            lfs = sibling.lfs
            if lfs is None or lfs.size != sibling.size:
                raise SourceReservoirError(f"{spec.source_id} LFS identity differs")
            rows.append(
                {
                    "path": sibling.rfilename,
                    "bytes": sibling.size,
                    "sha256": lfs.sha256,
                }
            )
        inventories[spec.source_id] = rows
    return inventories


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise SourceReservoirError("source reservoir output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_reservoir(
    manifest_path: Path, receipt_path: Path, *, token: str
) -> dict[str, Any]:
    """Resolve, select, and seal the live source reservoir."""

    if not token:
        raise SourceReservoirError("HF_TOKEN is required")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SourceReservoirError("source reservoir receipt already exists")
    rows = select_reservoir(_fetch_inventories(token))
    _atomic_jsonl(manifest_path, rows)
    by_source_bytes = Counter()
    by_source_files = Counter()
    for row in rows:
        by_source_bytes[row["source_id"]] += row["bytes"]
        by_source_files[row["source_id"]] += 1
    selected_bytes = sum(by_source_bytes.values())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "target_bytes": TARGET_BYTES,
        "target_tib": 8,
        "selected_bytes": selected_bytes,
        "selected_tib": selected_bytes / 1024**4,
        "overshoot_bytes": selected_bytes - TARGET_BYTES,
        "selected_files": len(rows),
        "by_source_bytes": dict(sorted(by_source_bytes.items())),
        "by_source_files": dict(sorted(by_source_files.items())),
        "manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "ordered_rows_sha256": canonical_sha256(rows),
        },
        "all_revisions_exact": True,
        "all_selected_files_lfs_sha256_bound": True,
        "all_sources_access_probed": True,
        "source_reservoir_complete": True,
        "cross_source_deduplication_complete": False,
        "quality_compilation_complete": False,
        "translation_complete": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_jsonl(receipt_path, [receipt])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    receipt = build_reservoir(
        args.manifest,
        args.receipt,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
