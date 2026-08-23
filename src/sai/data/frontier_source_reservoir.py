"""Freeze an exact high-quality-source augmentation reservoir of eight TiB."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.source_reservoir import TARGET_BYTES
from sai.data.token_stream import canonical_sha256, sha256_file

MANIFEST_SCHEMA = "sai-frontier-source-candidate-file-v1"
RECEIPT_SCHEMA = "sai-frontier-source-reservoir-receipt-v1"


@dataclass(frozen=True)
class FrontierSourceSpec:
    source_id: str
    repository: str
    revision: str
    prefixes: tuple[str, ...]
    suffix: str
    text_column: str
    license: str
    access: str
    epistemic_function: str


SOURCE_SPECS = (
    FrontierSourceSpec(
        "ultrafineweb_l2_en_20260820",
        "openbmb/Ultra-FineWeb",
        "02c85641e3d19a854be2e09139c25adaa9518063",
        ("data/ultrafineweb_l1_en_hq/",),
        ".parquet",
        "content",
        "apache-2.0_project_upstream_source_terms_apply",
        "public",
        "current_english_model_selected_web",
    ),
    FrontierSourceSpec(
        "ultrafineweb_l2_en_2025",
        "openbmb/Ultra-FineWeb",
        "02c85641e3d19a854be2e09139c25adaa9518063",
        ("data/ultrafineweb_en/",),
        ".parquet",
        "content",
        "apache-2.0_project_upstream_source_terms_apply",
        "public",
        "benchmark_validated_english_model_selected_web",
    ),
    FrontierSourceSpec(
        "fineweb2_hq_multilingual",
        "epfml/FineWeb2-HQ",
        "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e",
        ("",),
        ".parquet",
        "text",
        "odc-by-1.0_and_commoncrawl_terms",
        "public",
        "high_value_non_english_translation_discovery",
    ),
    FrontierSourceSpec(
        "nemotron_specialized_reasoning",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "9ed3718b5f2ae29074c5e34e64115432b7c4320f",
        (
            "Nemotron-Pretraining-RQA/",
            "Nemotron-Pretraining-InfiniByte-Reasoning/",
            "Nemotron-Pretraining-Math-Textbooks/",
            "Nemotron-Pretraining-Scientific-Coding/",
        ),
        ".parquet",
        "text",
        "cc-by-4.0_with_scientific_code_sharealike_and_generator_terms",
        "public",
        "cross_domain_and_stem_reasoning",
    ),
    FrontierSourceSpec(
        "ultradata_math_l1",
        "openbmb/UltraData-Math-L1",
        "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
        ("data/UltraData-Math-L1/",),
        ".parquet",
        "content",
        "apache-2.0_project_upstream_source_terms_apply",
        "public",
        "math_filtered_deduplicated_source",
    ),
)

GATED_CANDIDATE_REPOSITORIES = (
    "nvidia/Nemotron-CC-v2.1",
    "nvidia/Nemotron-CC-Code-v1",
    "nvidia/Nemotron-Pretraining-Code-v2",
)


class FrontierSourceReservoirError(RuntimeError):
    """A source revision, selected slice, or file identity differs."""


def select_frontier_sources(
    inventories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Select only explicitly named high-value slices without a bulk filler."""

    if set(inventories) != {spec.source_id for spec in SOURCE_SPECS}:
        raise FrontierSourceReservoirError("frontier inventories differ")
    rows = []
    identities = set()
    for spec in SOURCE_SPECS:
        source_rows = sorted(inventories[spec.source_id], key=lambda row: row["path"])
        if not source_rows:
            raise FrontierSourceReservoirError(
                f"frontier source is empty: {spec.source_id}"
            )
        for row in source_rows:
            if set(row) != {"path", "bytes", "sha256"}:
                raise FrontierSourceReservoirError("frontier file fields differ")
            path, size, digest = row["path"], row["bytes"], row["sha256"]
            identity = (spec.repository, path)
            if (
                not isinstance(path, str)
                or not path.endswith(spec.suffix)
                or not any(path.startswith(prefix) for prefix in spec.prefixes)
                or path.startswith("/")
                or ".." in Path(path).parts
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or identity in identities
            ):
                raise FrontierSourceReservoirError("frontier file identity differs")
            identities.add(identity)
            rows.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "source_id": spec.source_id,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "path": path,
                    "physical_bytes": size,
                    "sha256": digest,
                    "text_column": spec.text_column,
                    "license": spec.license,
                    "access": spec.access,
                    "epistemic_function": spec.epistemic_function,
                    "physical_bytes_are_text_payload_bytes": False,
                    "source_candidate_is_training_ready": False,
                    "ordinal": len(rows),
                }
            )
    if sum(row["physical_bytes"] for row in rows) < TARGET_BYTES:
        raise FrontierSourceReservoirError("frontier source bytes are below target")
    return rows


def _fetch_inventories(token: str) -> dict[str, list[dict[str, Any]]]:
    try:
        from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    except ImportError as error:
        raise FrontierSourceReservoirError("huggingface_hub is required") from error
    api = HfApi(token=token)
    repository_cache = {}
    inventories = {}
    for spec in SOURCE_SPECS:
        cache_key = (spec.repository, spec.revision)
        info = repository_cache.get(cache_key)
        if info is None:
            info = api.dataset_info(
                spec.repository, revision=spec.revision, files_metadata=True
            )
            if info.sha != spec.revision:
                raise FrontierSourceReservoirError(
                    f"frontier revision differs: {spec.source_id}"
                )
            repository_cache[cache_key] = info
        selected = [
            sibling
            for sibling in info.siblings
            if sibling.rfilename.endswith(spec.suffix)
            and any(sibling.rfilename.startswith(prefix) for prefix in spec.prefixes)
        ]
        if not selected:
            raise FrontierSourceReservoirError(
                f"frontier paths are absent: {spec.source_id}"
            )
        probe = selected[0]
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
            raise FrontierSourceReservoirError(
                f"frontier access probe differs: {spec.source_id}"
            )
        source_rows = []
        for sibling in selected:
            if sibling.lfs is None or sibling.lfs.size != sibling.size:
                raise FrontierSourceReservoirError(
                    f"frontier LFS identity differs: {spec.source_id}"
                )
            source_rows.append(
                {
                    "path": sibling.rfilename,
                    "bytes": sibling.size,
                    "sha256": sibling.lfs.sha256,
                }
            )
        inventories[spec.source_id] = source_rows
    return inventories


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise FrontierSourceReservoirError("frontier output already exists")
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


def build_frontier_reservoir(
    manifest_path: Path, receipt_path: Path, *, token: str
) -> dict[str, Any]:
    """Resolve exact revisions and seal the source-only augmentation inventory."""

    if not token or receipt_path.exists() or receipt_path.is_symlink():
        raise FrontierSourceReservoirError(
            "frontier credential or output boundary differs"
        )
    rows = select_frontier_sources(_fetch_inventories(token))
    _atomic_jsonl(manifest_path, rows)
    by_source_bytes = Counter()
    by_source_files = Counter()
    for row in rows:
        by_source_bytes[row["source_id"]] += row["physical_bytes"]
        by_source_files[row["source_id"]] += 1
    selected_bytes = sum(by_source_bytes.values())
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "minimum_source_physical_bytes": TARGET_BYTES,
        "selected_physical_bytes": selected_bytes,
        "selected_physical_tib": selected_bytes / 1024**4,
        "selected_files": len(rows),
        "by_source_physical_bytes": dict(sorted(by_source_bytes.items())),
        "by_source_files": dict(sorted(by_source_files.items())),
        "manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "ordered_rows_sha256": canonical_sha256(rows),
        },
        "all_revisions_exact": True,
        "all_selected_files_lfs_sha256_bound": True,
        "all_selected_slices_access_probed": True,
        "physical_bytes_are_text_payload_bytes": False,
        "text_payload_bytes_measured": False,
        "selected_slices_are_source_candidates": True,
        "gated_candidate_repositories_not_counted": list(GATED_CANDIDATE_REPOSITORIES),
        "overlap_with_prior_reservoir_resolved": False,
        "quality_compilation_complete": False,
        "rights_review_complete": False,
        "benchmark_decontamination_complete": False,
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
    payload = build_frontier_reservoir(
        args.manifest,
        args.receipt,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
