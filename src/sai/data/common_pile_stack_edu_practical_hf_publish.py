"""Publish admitted Stack-Edu locators and their custody metadata to Sai."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.common_pile_stack_edu_practical_admission import (
    SCHEMA as ADMISSION_SCHEMA,
)
from sai.data.common_pile_stack_edu_practical_scan import LOCATOR_SCHEMA
from sai.data.pleias_production_materializer import upload_verified
from sai.data.token_stream import canonical_sha256, sha256_file

DESTINATION_REPOSITORY = "Godlydonuts/Sai"
LOCATOR_PREFIX = "training/practical/code-stack-edu/20260826-r1"
METADATA_PREFIX = "training/practical/metadata/20260826-r3/code-stack-edu"
SHARD_SCHEMA = "sai-stack-edu-practical-hf-publish-shard-v1"
AGGREGATE_SCHEMA = "sai-stack-edu-practical-hf-publish-aggregate-v1"
METADATA_SCHEMA = "sai-stack-edu-practical-hf-publish-metadata-v1"


class StackEduPracticalHfPublishError(RuntimeError):
    """A Stack-Edu admission, locator, or remote identity differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise StackEduPracticalHfPublishError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise StackEduPracticalHfPublishError("signed input is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise StackEduPracticalHfPublishError("signed input differs")
    return payload


def _admission(root: Path) -> dict[str, Any]:
    admission = _load_signed(root / "receipt.json", ADMISSION_SCHEMA)
    quarantine = admission.get("source", {}).get("quarantine_registry", {})
    policy = admission.get("policy", {})
    if (
        admission.get("status") != "complete_common_pile_stack_edu_practical_admission"
        or admission.get("practical_pretraining_ready") is not True
        or admission.get("training_ready") is not True
        or admission.get("global_exact_content_deduplication_complete") is not True
        or admission.get("known_quarantine_exclusions_applied") is not True
        or policy.get("byte_cap_selection_policy") != "canonical_content_sha256_order"
        or policy.get("output_partition_policy")
        != "canonical_source_path_sha256_modulo"
        or not isinstance(quarantine, dict)
        or not isinstance(quarantine.get("rows"), int)
        or isinstance(quarantine.get("rows"), bool)
        or quarantine["rows"] < 1
        or not isinstance(quarantine.get("unique_content_hashes"), int)
        or isinstance(quarantine.get("unique_content_hashes"), bool)
        or not 1 <= quarantine["unique_content_hashes"] <= quarantine["rows"]
        or admission.get("source_text_copied") is not False
    ):
        raise StackEduPracticalHfPublishError("Stack-Edu admission differs")
    return admission


def _descriptor(admission: dict[str, Any], shard_index: int) -> dict[str, Any] | None:
    matches = [
        row
        for row in admission.get("outputs", {}).get("descriptors", [])
        if row.get("shard_index") == shard_index
    ]
    if len(matches) > 1:
        raise StackEduPracticalHfPublishError("output shard overlaps")
    return matches[0] if matches else None


def publish_shard(
    admission_root: Path,
    output_root: Path,
    shard_index: int,
    logical_shards: int,
    token: str,
) -> dict[str, Any]:
    """Upload one text-free code locator shard and verify its remote identity."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not token
        or not 0 <= shard_index < logical_shards
    ):
        raise StackEduPracticalHfPublishError("publish shard arguments differ")
    admission = _admission(admission_root)
    descriptor = _descriptor(admission, shard_index)
    remote = None
    if descriptor is not None:
        path = admission_root / descriptor.get("path", "")
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise StackEduPracticalHfPublishError("local locator differs")
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise StackEduPracticalHfPublishError("pyarrow is required") from error
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if (
            "text" in names
            or "content" in names
            or "source_row_identity_sha256" not in names
            or "content_sha256" not in names
            or parquet.metadata.num_rows != descriptor.get("rows")
        ):
            raise StackEduPracticalHfPublishError("locator schema differs")
        rows = 0
        differs = False
        for batch in parquet.iter_batches(
            batch_size=65_536, columns=["schema"], use_threads=False
        ):
            values = batch.column(0).to_pylist()
            rows += len(values)
            differs = differs or any(value != LOCATOR_SCHEMA for value in values)
        if rows != descriptor["rows"] or differs:
            raise StackEduPracticalHfPublishError("locator rows differ")
        remote_path = (
            f"{LOCATOR_PREFIX}/shards/shard_{shard_index:05d}/locators.parquet"
        )
        remote = upload_verified(
            path, remote_path, token, repository=DESTINATION_REPOSITORY
        )
        if (
            remote["bytes"] != descriptor["bytes"]
            or remote["sha256"] != descriptor["sha256"]
        ):
            raise StackEduPracticalHfPublishError("remote locator differs")
    payload = {
        "schema": SHARD_SCHEMA,
        "status": (
            "complete_stack_edu_practical_hf_publish_shard"
            if descriptor is not None
            else "complete_stack_edu_practical_hf_publish_empty_shard"
        ),
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "admission_receipt_sha256": admission["receipt_sha256"],
        "source_descriptor": descriptor,
        "remote_output": remote,
        "source_text_uploaded": False,
        "training_ready": True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output_root.mkdir(parents=True)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate_publish(
    admission_root: Path,
    publish_root: Path,
    logical_shards: int,
    output: Path,
    token: str,
) -> dict[str, Any]:
    """Replay every shard receipt and every remote locator identity."""

    if output.exists() or output.is_symlink() or not token or logical_shards < 1:
        raise StackEduPracticalHfPublishError("aggregate arguments differ")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise StackEduPracticalHfPublishError("huggingface_hub is required") from error
    admission = _admission(admission_root)
    receipt_hashes = []
    remotes: dict[str, dict[str, Any]] = {}
    rows = locator_bytes = text_bytes = 0
    for shard_index in range(logical_shards):
        receipt = _load_signed(
            publish_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
            SHARD_SCHEMA,
        )
        descriptor = _descriptor(admission, shard_index)
        remote = receipt.get("remote_output")
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("admission_receipt_sha256") != admission["receipt_sha256"]
            or receipt.get("source_descriptor") != descriptor
            or receipt.get("source_text_uploaded") is not False
            or receipt.get("training_ready") is not True
            or (descriptor is None) != (remote is None)
        ):
            raise StackEduPracticalHfPublishError("shard receipt differs")
        if descriptor is not None:
            if (
                receipt.get("status") != "complete_stack_edu_practical_hf_publish_shard"
                or not isinstance(remote, dict)
                or remote.get("repository") != DESTINATION_REPOSITORY
                or remote.get("path") in remotes
                or remote.get("bytes") != descriptor["bytes"]
                or remote.get("sha256") != descriptor["sha256"]
            ):
                raise StackEduPracticalHfPublishError("published shard differs")
            remotes[remote["path"]] = remote
            rows += descriptor["rows"]
            locator_bytes += descriptor["bytes"]
            text_bytes += descriptor["text_utf8_bytes"]
        elif (
            receipt.get("status")
            != "complete_stack_edu_practical_hf_publish_empty_shard"
        ):
            raise StackEduPracticalHfPublishError("empty shard receipt differs")
        receipt_hashes.append(receipt["receipt_sha256"])
    counts = admission["counts"]
    if (
        rows != counts["admitted_rows"]
        or text_bytes != counts["admitted_text_utf8_bytes"]
    ):
        raise StackEduPracticalHfPublishError("publication accounting differs")
    info = HfApi(token=token).dataset_info(DESTINATION_REPOSITORY, files_metadata=True)
    siblings = {row.rfilename: row for row in info.siblings or []}
    for path, expected in remotes.items():
        sibling = siblings.get(path)
        lfs = None if sibling is None else sibling.lfs
        if (
            sibling is None
            or lfs is None
            or sibling.size != expected["bytes"]
            or lfs.size != expected["bytes"]
            or lfs.sha256 != expected["sha256"]
        ):
            raise StackEduPracticalHfPublishError("remote replay differs")
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_stack_edu_practical_hf_locator_publication",
        "admission_receipt_sha256": admission["receipt_sha256"],
        "logical_shards": logical_shards,
        "ordered_publish_receipts_sha256": canonical_sha256(receipt_hashes),
        "remote": {
            "repository": DESTINATION_REPOSITORY,
            "revision_verified": info.sha,
            "prefix": LOCATOR_PREFIX,
            "files": len(remotes),
            "locator_parquet_bytes": locator_bytes,
            "logical_training_text_utf8_bytes": text_bytes,
            "rows": rows,
        },
        "all_remote_lfs_identities_verified": True,
        "source_text_uploaded": False,
        "training_ready": True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def publish_metadata(
    admission_root: Path,
    aggregate_path: Path,
    output: Path,
    token: str,
) -> dict[str, Any]:
    """Publish the code admission and remote replay receipts."""

    if output.exists() or output.is_symlink() or not token:
        raise StackEduPracticalHfPublishError("metadata arguments differ")
    admission = _admission(admission_root)
    aggregate = _load_signed(aggregate_path, AGGREGATE_SCHEMA)
    if (
        aggregate.get("admission_receipt_sha256") != admission["receipt_sha256"]
        or aggregate.get("all_remote_lfs_identities_verified") is not True
        or aggregate.get("training_ready") is not True
    ):
        raise StackEduPracticalHfPublishError("metadata inputs differ")
    remotes = []
    for path, remote_path in (
        (admission_root / "receipt.json", f"{METADATA_PREFIX}/admission.json"),
        (aggregate_path, f"{METADATA_PREFIX}/publish-aggregate.json"),
    ):
        remotes.append(
            upload_verified(path, remote_path, token, repository=DESTINATION_REPOSITORY)
        )
    payload = {
        "schema": METADATA_SCHEMA,
        "status": "complete_stack_edu_practical_hf_metadata_publication",
        "admission_receipt_sha256": admission["receipt_sha256"],
        "locator_publication_receipt_sha256": aggregate["receipt_sha256"],
        "remote_repository": DESTINATION_REPOSITORY,
        "remote_outputs": remotes,
        "source_text_uploaded": False,
        "training_ready": True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--admission-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--token-env", default="HF_TOKEN")
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--admission-root", type=Path, required=True)
    aggregate.add_argument("--publish-root", type=Path, required=True)
    aggregate.add_argument("--logical-shards", type=int, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--token-env", default="HF_TOKEN")
    metadata = commands.add_parser("metadata")
    metadata.add_argument("--admission-root", type=Path, required=True)
    metadata.add_argument("--aggregate", type=Path, required=True)
    metadata.add_argument("--output", type=Path, required=True)
    metadata.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if args.command == "shard":
        result = publish_shard(
            args.admission_root,
            args.output_root,
            args.shard_index,
            args.logical_shards,
            token,
        )
    elif args.command == "aggregate":
        result = aggregate_publish(
            args.admission_root,
            args.publish_root,
            args.logical_shards,
            args.output,
            token,
        )
    else:
        result = publish_metadata(
            args.admission_root, args.aggregate, args.output, token
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
