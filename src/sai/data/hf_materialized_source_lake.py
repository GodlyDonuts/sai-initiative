"""Verify and freeze the materialized Hugging Face source lake."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file

DESTINATION_REPOSITORY = "Godlydonuts/Sai"
TARGET_BYTES = 8 * 1024**4
FILE_SCHEMA = "sai-hf-materialized-source-file-v1"
RECEIPT_SCHEMA = "sai-hf-materialized-source-lake-receipt-v1"
SOURCE_MANIFEST_SCHEMA = "sai-hf-source-publication-manifest-v1"
FINEPDFS_REPOSITORY = "HuggingFaceFW/finepdfs"
FINEPDFS_REVISION = "220bac3acbf07789502c621d2d33952f51ac7f86"
FINEPDFS_PREFIX = "sources/reality-anchors/global-pdfs/finepdfs/220bac3a-r1/"
SOURCE_MANIFEST_PREFIXES = (
    "sources/reality-anchors/common-pile/",
    "sources/reality-anchors/global-open-corpus/pleias-common-corpus/",
    "sources/reality-anchors/math/finemath-4plus/",
    "sources/reasoning-and-synthesis/",
)


class MaterializedSourceLakeError(RuntimeError):
    """A remote source identity or materialized-lake claim differs."""


@dataclass(frozen=True)
class RemoteIdentity:
    path: str
    bytes: int
    sha256: str


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _lfs_value(lfs: Any, key: str) -> Any:
    if isinstance(lfs, dict):
        return lfs.get(key)
    return getattr(lfs, key, None)


def _remote_identities(siblings: Iterable[Any]) -> dict[str, RemoteIdentity]:
    result: dict[str, RemoteIdentity] = {}
    for sibling in siblings:
        lfs = getattr(sibling, "lfs", None)
        if lfs is None:
            continue
        path = getattr(sibling, "rfilename", None)
        size = _lfs_value(lfs, "size")
        digest = _lfs_value(lfs, "sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not _is_sha256(digest)
            or path in result
        ):
            raise MaterializedSourceLakeError("remote LFS identity differs")
        result[path] = RemoteIdentity(path, size, digest)
    return result


def _validate_source_manifest(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise MaterializedSourceLakeError("source manifest is not an object")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    files = payload.get("files", payload.get("source_files"))
    if (
        payload.get("schema") != SOURCE_MANIFEST_SCHEMA
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("status") != "complete_source_snapshot_candidate_only"
        or payload.get("training_ready") is not False
        or not isinstance(payload.get("source_repository"), str)
        or not _is_git_sha(payload.get("source_revision"))
        or not isinstance(files, list)
        or not files
        or payload.get("file_count") != len(files)
        or payload.get("compressed_bytes")
        != sum(row.get("bytes", 0) for row in files if isinstance(row, dict))
    ):
        raise MaterializedSourceLakeError("source manifest differs")
    identities = set()
    for row in files:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or row["path"].startswith("/")
            or ".." in Path(row["path"]).parts
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not _is_sha256(row.get("sha256"))
            or row["path"] in identities
        ):
            raise MaterializedSourceLakeError("source manifest file differs")
        identities.add(row["path"])
    return files


def _lake_file(
    *,
    destination_revision: str,
    destination_path: str,
    source_id: str,
    source_repository: str,
    source_revision: str,
    source_path: str,
    size: int,
    digest: str,
    source_manifest_path: str | None,
) -> dict[str, Any]:
    return {
        "schema": FILE_SCHEMA,
        "destination_repository": DESTINATION_REPOSITORY,
        "destination_revision": destination_revision,
        "destination_path": destination_path,
        "source_id": source_id,
        "source_repository": source_repository,
        "source_revision": source_revision,
        "source_path": source_path,
        "bytes": size,
        "sha256": digest,
        "source_manifest_path": source_manifest_path,
        "raw_source_is_training_ready": False,
    }


def summarize_lake(
    rows: list[dict[str, Any]],
    destination_revision: str,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate file rows and return deterministic aggregate accounting."""

    if not _is_git_sha(destination_revision) or not rows or not components:
        raise MaterializedSourceLakeError("source-lake inputs differ")
    paths = set()
    identities = set()
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema") != FILE_SCHEMA
            or row.get("destination_repository") != DESTINATION_REPOSITORY
            or row.get("destination_revision") != destination_revision
            or not isinstance(row.get("destination_path"), str)
            or not isinstance(row.get("source_id"), str)
            or not isinstance(row.get("source_repository"), str)
            or not _is_git_sha(row.get("source_revision"))
            or not isinstance(row.get("source_path"), str)
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or not _is_sha256(row.get("sha256"))
            or row.get("raw_source_is_training_ready") is not False
            or row["destination_path"] in paths
            or (
                row["source_repository"],
                row["source_revision"],
                row["source_path"],
            )
            in identities
        ):
            raise MaterializedSourceLakeError("source-lake file differs")
        paths.add(row["destination_path"])
        identities.add(
            (
                row["source_repository"],
                row["source_revision"],
                row["source_path"],
            )
        )
        source = by_source[row["source_id"]]
        source["files"] += 1
        source["bytes"] += row["bytes"]
    total_bytes = sum(row["bytes"] for row in rows)
    if total_bytes < TARGET_BYTES:
        raise MaterializedSourceLakeError("materialized source lake is below target")
    component_files = sum(row.get("materialized_files", 0) for row in components)
    component_bytes = sum(row.get("materialized_bytes", 0) for row in components)
    if component_files != len(rows) or component_bytes != total_bytes:
        raise MaterializedSourceLakeError("component accounting differs")
    return {
        "destination_repository": DESTINATION_REPOSITORY,
        "destination_revision": destination_revision,
        "materialized_files": len(rows),
        "materialized_bytes": total_bytes,
        "materialized_decimal_tb": total_bytes / 10**12,
        "materialized_tib": total_bytes / 1024**4,
        "target_bytes": TARGET_BYTES,
        "target_met": True,
        "bytes_above_target": total_bytes - TARGET_BYTES,
        "by_source": dict(sorted(by_source.items())),
        "components": components,
    }


def _load_remote_json(
    repository: str, path: str, revision: str, token: str
) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise MaterializedSourceLakeError("huggingface_hub is required") from error
    local_path = hf_hub_download(
        repository,
        path,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    try:
        payload = json.loads(Path(local_path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializedSourceLakeError("source manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise MaterializedSourceLakeError("source manifest is not an object")
    return payload


def collect_remote_lake(
    token: str, destination_revision: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay every destination LFS identity against its pinned source."""

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise MaterializedSourceLakeError("huggingface_hub is required") from error
    if not token or not _is_git_sha(destination_revision):
        raise MaterializedSourceLakeError("remote collection inputs differ")
    api = HfApi(token=token)
    destination_info = api.dataset_info(
        DESTINATION_REPOSITORY,
        revision=destination_revision,
        files_metadata=True,
    )
    if destination_info.sha != destination_revision:
        raise MaterializedSourceLakeError("destination revision differs")
    destination = _remote_identities(destination_info.siblings or [])
    manifest_paths = sorted(
        sibling.rfilename
        for sibling in destination_info.siblings or []
        if sibling.rfilename.endswith("/source-manifest.json")
        and sibling.rfilename.startswith(SOURCE_MANIFEST_PREFIXES)
    )
    if not manifest_paths:
        raise MaterializedSourceLakeError("source manifests are absent")
    rows: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    source_cache: dict[tuple[str, str], dict[str, RemoteIdentity]] = {}
    for manifest_path in manifest_paths:
        manifest = _load_remote_json(
            DESTINATION_REPOSITORY, manifest_path, destination_revision, token
        )
        source_files = _validate_source_manifest(manifest)
        repository = manifest["source_repository"]
        revision = manifest["source_revision"]
        source_key = (repository, revision)
        if source_key not in source_cache:
            source_info = api.dataset_info(
                repository, revision=revision, files_metadata=True
            )
            if source_info.sha != revision:
                raise MaterializedSourceLakeError("upstream revision differs")
            source_cache[source_key] = _remote_identities(source_info.siblings or [])
        upstream = source_cache[source_key]
        target_root = f"{manifest_path.rsplit('/', 1)[0]}/data/"
        source_id = manifest.get("source_id") or manifest.get("source_config")
        if not isinstance(source_id, str) or not source_id:
            raise MaterializedSourceLakeError("source identifier differs")
        for file_row in source_files:
            source_path = file_row["path"]
            target_relative = source_path
            source_config = manifest.get("source_config")
            if isinstance(source_config, str) and source_path.startswith(
                f"{source_config}/"
            ):
                target_relative = source_path[len(source_config) + 1 :]
            target_path = f"{target_root}{target_relative}"
            source_identity = upstream.get(source_path)
            target_identity = destination.get(target_path)
            expected = (file_row["bytes"], file_row["sha256"])
            if (
                source_identity is None
                or target_identity is None
                or (source_identity.bytes, source_identity.sha256) != expected
                or (target_identity.bytes, target_identity.sha256) != expected
            ):
                raise MaterializedSourceLakeError(
                    f"mirrored source identity differs: {repository}:{source_path}"
                )
            rows.append(
                _lake_file(
                    destination_revision=destination_revision,
                    destination_path=target_path,
                    source_id=source_id,
                    source_repository=repository,
                    source_revision=revision,
                    source_path=source_path,
                    size=file_row["bytes"],
                    digest=file_row["sha256"],
                    source_manifest_path=manifest_path,
                )
            )
        components.append(
            {
                "source_id": source_id,
                "source_repository": repository,
                "source_revision": revision,
                "source_manifest_path": manifest_path,
                "source_manifest_receipt_sha256": manifest["receipt_sha256"],
                "status": manifest["status"],
                "materialized_files": len(source_files),
                "materialized_bytes": sum(row["bytes"] for row in source_files),
                "complete_source_snapshot": True,
                "training_ready": False,
            }
        )

    finepdfs_info = api.dataset_info(
        FINEPDFS_REPOSITORY,
        revision=FINEPDFS_REVISION,
        files_metadata=True,
    )
    if finepdfs_info.sha != FINEPDFS_REVISION:
        raise MaterializedSourceLakeError("FinePDFs revision differs")
    finepdfs_upstream = _remote_identities(finepdfs_info.siblings or [])
    finepdfs_destination = {
        path[len(f"{FINEPDFS_PREFIX}data/") :]: identity
        for path, identity in destination.items()
        if path.startswith(f"{FINEPDFS_PREFIX}data/")
    }
    if not finepdfs_destination:
        raise MaterializedSourceLakeError("FinePDFs materialization is absent")
    for source_path, target_identity in sorted(finepdfs_destination.items()):
        source_identity = finepdfs_upstream.get(source_path)
        if source_identity is None or (
            source_identity.bytes,
            source_identity.sha256,
        ) != (target_identity.bytes, target_identity.sha256):
            raise MaterializedSourceLakeError("FinePDFs identity differs")
        rows.append(
            _lake_file(
                destination_revision=destination_revision,
                destination_path=f"{FINEPDFS_PREFIX}data/{source_path}",
                source_id="finepdfs_quota_boundary",
                source_repository=FINEPDFS_REPOSITORY,
                source_revision=FINEPDFS_REVISION,
                source_path=source_path,
                size=source_identity.bytes,
                digest=source_identity.sha256,
                source_manifest_path=None,
            )
        )
    components.append(
        {
            "source_id": "finepdfs_quota_boundary",
            "source_repository": FINEPDFS_REPOSITORY,
            "source_revision": FINEPDFS_REVISION,
            "source_manifest_path": None,
            "source_manifest_receipt_sha256": None,
            "status": "partial_source_snapshot_candidate_only_public_storage_boundary",
            "materialized_files": len(finepdfs_destination),
            "materialized_bytes": sum(
                identity.bytes for identity in finepdfs_destination.values()
            ),
            "upstream_files": len(finepdfs_upstream),
            "remaining_upstream_files": len(finepdfs_upstream)
            - len(finepdfs_destination),
            "complete_source_snapshot": len(finepdfs_destination)
            == len(finepdfs_upstream),
            "training_ready": False,
        }
    )
    rows.sort(key=lambda row: row["destination_path"])
    components.sort(key=lambda row: (row["source_repository"], row["source_id"]))
    summarize_lake(rows, destination_revision, components)
    return rows, components


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def freeze_remote_lake(
    token: str, destination_revision: str, output_root: Path
) -> dict[str, Any]:
    """Write a file-level manifest and aggregate source-lake receipt."""

    if output_root.exists() or output_root.is_symlink():
        raise MaterializedSourceLakeError("output root already exists")
    output_root.mkdir(parents=True, mode=0o700)
    try:
        rows, components = collect_remote_lake(token, destination_revision)
        summary = summarize_lake(rows, destination_revision, components)
        manifest_path = output_root / "manifest.jsonl"
        _write_manifest(manifest_path, rows)
        payload = {
            "schema": RECEIPT_SCHEMA,
            "status": "complete_materialized_source_lake_quota_boundary",
            **summary,
            "manifest": {
                "path": manifest_path.name,
                "rows": len(rows),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
            "all_destination_lfs_identities_replayed_against_pinned_upstream": True,
            "finepdfs_snapshot_complete": False,
            "hugging_face_public_storage_gate_observed": True,
            "source_provenance_is_final_training_admission": False,
            "source_wide_rights_clearance_complete": False,
            "global_exact_and_near_deduplication_complete": False,
            "full_benchmark_decontamination_complete": False,
            "hermes_full_population_quality_compilation_complete": False,
            "translation_complete": False,
            "curriculum_assignment_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        receipt_path = output_root / "receipt.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(receipt_path, flags, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        return payload
    except BaseException:
        for path in output_root.iterdir():
            if path.is_file() and not path.is_symlink():
                path.unlink()
        output_root.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise MaterializedSourceLakeError(f"{args.token_env} is required")
    result = freeze_remote_lake(token, args.destination_revision, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
