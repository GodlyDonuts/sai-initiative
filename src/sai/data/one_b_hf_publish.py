"""Publish Sai's self-contained packed 1B stream and portable manifest to HF."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_stage_schedule import SCHEMA as SCHEDULE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file, sha256_tree
from sai.tokenizer.production_qualification import SCHEMA as TOKENIZER_SCHEMA
from sai.training.one_b_olmo_config import SCHEMA as CONFIG_SCHEMA

SCHEMA = "sai-1b-packed-hf-publication-v1"
REPOSITORY = "Godlydonuts/Sai"
PREFIX = "training/packed/one-b/20260826-r2"


class OneBHfPublishError(RuntimeError):
    """A schedule, tokenizer, packed file, upload, or remote identity differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBHfPublishError("signed publication input differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBHfPublishError("signed publication input differs")
    return value


def _packed_files(
    schedule: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: dict[str, dict[str, Any]] = {}
    exposures = []
    for stage in schedule["stages"]:
        for phase in ("body", "boundary"):
            for entry in stage[f"{phase}_entries"]:
                path = Path(entry["path"])
                sha256 = entry["sha256"]
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or sha256_file(path) != sha256
                    or path.stat().st_size != entry["sequences_per_repeat"] * 4_096 * 2
                ):
                    raise OneBHfPublishError("packed schedule file differs")
                remote = f"{PREFIX}/data/{sha256[:2]}/{sha256}.bin"
                descriptor = {
                    "sha256": sha256,
                    "local_path": str(path.resolve()),
                    "remote_path": remote,
                    "bytes": path.stat().st_size,
                    "sequences": entry["sequences_per_repeat"],
                    "tokens": entry["tokens_per_repeat"],
                }
                prior = files.get(sha256)
                if prior is not None:
                    comparable = ("remote_path", "bytes", "sequences", "tokens")
                    if any(prior[key] != descriptor[key] for key in comparable):
                        raise OneBHfPublishError("packed content identity overlaps")
                else:
                    files[sha256] = descriptor
                exposures.append(
                    {
                        "stage": stage["index"],
                        "stage_name": stage["stage"],
                        "phase": phase,
                        "band": entry["band"],
                        "source": entry["source"],
                        "remote_path": remote,
                        "sha256": sha256,
                        "sequences_per_repeat": entry["sequences_per_repeat"],
                        "repeat": entry["repeat"],
                    }
                )
    return sorted(files.values(), key=lambda row: row["remote_path"]), exposures


def _tokenizer_files(root: Path, identity: str) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink() or sha256_tree(root) != identity:
        raise OneBHfPublishError("production tokenizer tree differs")
    values = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file() or path.is_symlink():
            raise OneBHfPublishError("production tokenizer member differs")
        relative = path.relative_to(root)
        values.append(
            {
                "sha256": sha256_file(path),
                "local_path": str(path.resolve()),
                "remote_path": f"{PREFIX}/tokenizer/{relative.as_posix()}",
                "bytes": path.stat().st_size,
            }
        )
    if not values:
        raise OneBHfPublishError("production tokenizer tree is empty")
    return values


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def publish(
    schedule_path: Path,
    qualification_path: Path,
    tokenizer_root: Path,
    config_receipt_path: Path,
    output: Path,
    token: str,
) -> dict[str, Any]:
    """Upload content-addressed training bytes and verify every remote LFS SHA."""

    if output.exists() or output.is_symlink() or not token:
        raise OneBHfPublishError("publication arguments differ")
    schedule = _load_signed(schedule_path, SCHEDULE_SCHEMA)
    qualification = _load_signed(qualification_path, TOKENIZER_SCHEMA)
    configs = _load_signed(config_receipt_path, CONFIG_SCHEMA)
    if (
        qualification.get("status") != "qualified_production_48k"
        or schedule.get("tokenizer_identity_sha256")
        != qualification.get("tokenizer_identity_sha256")
        or configs.get("schedule_receipt_sha256") != schedule["receipt_sha256"]
        or configs.get("tokenizer_qualification_receipt_sha256")
        != qualification["receipt_sha256"]
    ):
        raise OneBHfPublishError("packed publication lineage differs")
    packed, exposures = _packed_files(schedule)
    tokenizer_files = _tokenizer_files(
        tokenizer_root, qualification["tokenizer_identity_sha256"]
    )
    metadata_root = output.parent / f".hf-publication.partial.{uuid.uuid4().hex}"
    metadata_root.mkdir(parents=True)
    try:
        file_manifest = metadata_root / "physical-files.jsonl"
        exposure_manifest = metadata_root / "stage-exposures.jsonl"
        _write_jsonl(
            file_manifest,
            [
                {
                    key: row[key]
                    for key in ("remote_path", "sha256", "bytes", "sequences", "tokens")
                }
                for row in packed
            ],
        )
        _write_jsonl(exposure_manifest, exposures)
        card = metadata_root / "README.md"
        card.write_text(
            "# Sai 1B packed production stream\n\n"
            "This immutable release contains the directly trainable uint16 token "
            "stream, the qualified 48K tokenizer, and an exact 4T stage/exposure "
            "manifest. It is "
            "the self-contained payload complement to the source locator registry. "
            "Development rows are physically excluded. Full source text is not "
            "duplicated because the token bytes are the production training "
            "representation.\n",
            encoding="utf-8",
        )
        metadata = [
            {
                "local_path": str(file_manifest),
                "remote_path": f"{PREFIX}/manifests/{file_manifest.name}",
                "bytes": file_manifest.stat().st_size,
                "sha256": sha256_file(file_manifest),
            },
            {
                "local_path": str(exposure_manifest),
                "remote_path": f"{PREFIX}/manifests/{exposure_manifest.name}",
                "bytes": exposure_manifest.stat().st_size,
                "sha256": sha256_file(exposure_manifest),
            },
            {
                "local_path": str(card),
                "remote_path": f"{PREFIX}/README.md",
                "bytes": card.stat().st_size,
                "sha256": sha256_file(card),
            },
            {
                "local_path": str(schedule_path.resolve()),
                "remote_path": f"{PREFIX}/receipts/schedule.json",
                "bytes": schedule_path.stat().st_size,
                "sha256": sha256_file(schedule_path),
            },
            {
                "local_path": str(qualification_path.resolve()),
                "remote_path": f"{PREFIX}/receipts/tokenizer-qualification.json",
                "bytes": qualification_path.stat().st_size,
                "sha256": sha256_file(qualification_path),
            },
            {
                "local_path": str(config_receipt_path.resolve()),
                "remote_path": f"{PREFIX}/receipts/olmo-config-bundle.json",
                "bytes": config_receipt_path.stat().st_size,
                "sha256": sha256_file(config_receipt_path),
            },
        ]
        uploads = packed + tokenizer_files + metadata
        try:
            from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        except ImportError as error:
            raise OneBHfPublishError("huggingface_hub is required") from error
        api = HfApi(token=token)
        api.create_repo(REPOSITORY, repo_type="dataset", exist_ok=True, private=False)
        existing_info = api.dataset_info(REPOSITORY, files_metadata=True)
        existing = {row.rfilename: row for row in existing_info.siblings}
        pending = []
        for row in uploads:
            remote = existing.get(row["remote_path"])
            if remote is not None:
                lfs = remote.lfs
                if remote.size != row["bytes"] or (
                    lfs is not None and lfs.sha256 != row["sha256"]
                ):
                    raise OneBHfPublishError("existing remote packed file differs")
                if lfs is not None:
                    continue
            pending.append(row)
        for offset in range(0, len(pending), 32):
            batch = pending[offset : offset + 32]
            api.create_commit(
                REPOSITORY,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=row["remote_path"],
                        path_or_fileobj=row["local_path"],
                    )
                    for row in batch
                ],
                commit_message=(
                    f"Publish Sai 1B packed files "
                    f"{offset + 1}-{offset + len(batch)}"
                ),
                num_threads=8,
            )
        final_info = api.dataset_info(REPOSITORY, files_metadata=True)
        remotes = {row.rfilename: row for row in final_info.siblings}
        for row in uploads:
            remote = remotes.get(row["remote_path"])
            lfs = None if remote is None else remote.lfs
            if remote is None or remote.size != row["bytes"]:
                raise OneBHfPublishError("remote publication identity differs")
            if lfs is not None:
                if lfs.sha256 != row["sha256"]:
                    raise OneBHfPublishError("remote publication identity differs")
            else:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=REPOSITORY,
                        filename=row["remote_path"],
                        repo_type="dataset",
                        revision=final_info.sha,
                        token=token,
                    )
                )
                if (
                    downloaded.stat().st_size != row["bytes"]
                    or sha256_file(downloaded) != row["sha256"]
                ):
                    raise OneBHfPublishError("remote publication identity differs")
        payload = {
            "schema": SCHEMA,
            "status": "complete_self_contained_1b_packed_hf_publication",
            "repository": REPOSITORY,
            "revision": final_info.sha,
            "prefix": PREFIX,
            "schedule_receipt_sha256": schedule["receipt_sha256"],
            "tokenizer_qualification_receipt_sha256": qualification["receipt_sha256"],
            "config_receipt_sha256": configs["receipt_sha256"],
            "physical_data_files": len(packed),
            "physical_data_bytes": sum(row["bytes"] for row in packed),
            "physical_unique_tokens": sum(row["tokens"] for row in packed),
            "tokenizer_files": len(tokenizer_files),
            "stage_exposure_rows": len(exposures),
            "all_packed_lfs_identities_verified": True,
            "all_remote_identities_verified": True,
            "development_rows_excluded": True,
            "source_text_uploaded": False,
            "packed_training_tokens_uploaded": True,
            "directly_trainable_after_download": True,
            "model_training_started": False,
            "one_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output, payload)
        return payload
    finally:
        for path in sorted(metadata_root.glob("*")):
            path.unlink(missing_ok=True)
        metadata_root.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--config-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", required=True)
    args = parser.parse_args()
    value = publish(
        args.schedule,
        args.qualification,
        args.tokenizer_root,
        args.config_receipt,
        args.output,
        os.environ.get(args.token_env, ""),
    )
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
