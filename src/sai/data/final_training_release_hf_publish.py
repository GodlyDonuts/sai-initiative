"""Publish and replay the signed Sai training-data release manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.final_training_release import SCHEMA as RELEASE_SCHEMA
from sai.data.pleias_production_materializer import upload_verified
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-final-training-release-hf-publication-v1"
DESTINATION_REPOSITORY = "Godlydonuts/Sai"
DESTINATION_PATH = "training/final/metadata/20260826-r1/release.lfs.json"


class FinalTrainingReleaseHfPublishError(RuntimeError):
    """The final release manifest or its remote LFS identity differs."""


def _load_release(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise FinalTrainingReleaseHfPublishError("release manifest is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalTrainingReleaseHfPublishError("release manifest differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != RELEASE_SCHEMA
        or payload.get("status") != "complete_sai_training_data_release"
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("all_required_components_present") is not True
        or payload.get("verified_cross_domain_connection_overlay_complete") is not True
        or payload.get("connection_development_rows_physically_excluded") is not True
        or payload.get("training_data_ready") is not True
        or payload.get("model_training_started") is not False
        or payload.get("four_b_training_authorized") is not False
    ):
        raise FinalTrainingReleaseHfPublishError("release manifest differs")
    return payload


def publish(release_path: Path, output: Path, token: str) -> dict[str, Any]:
    """Upload the manifest as LFS and independently replay its remote identity."""

    if output.exists() or output.is_symlink() or not token:
        raise FinalTrainingReleaseHfPublishError("publication arguments differ")
    release = _load_release(release_path)
    remote = upload_verified(
        release_path,
        DESTINATION_PATH,
        token,
        repository=DESTINATION_REPOSITORY,
    )
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise FinalTrainingReleaseHfPublishError(
            "huggingface_hub is required"
        ) from error
    info = HfApi(token=token).dataset_info(DESTINATION_REPOSITORY, files_metadata=True)
    sibling = next(
        (row for row in info.siblings or [] if row.rfilename == DESTINATION_PATH), None
    )
    lfs = None if sibling is None else sibling.lfs
    if (
        sibling is None
        or sibling.size != remote["bytes"]
        or lfs is None
        or lfs.size != remote["bytes"]
        or lfs.sha256 != remote["sha256"]
    ):
        raise FinalTrainingReleaseHfPublishError("remote release LFS differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_sai_training_data_release_hf_publication",
        "release_receipt_sha256": release["receipt_sha256"],
        "remote_repository": DESTINATION_REPOSITORY,
        "remote_revision_verified": info.sha,
        "remote_output": remote,
        "all_remote_lfs_identities_verified": True,
        "verified_cross_domain_connection_overlay_in_release": True,
        "training_data_ready": True,
        "model_training_started": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = publish(args.release, args.output, os.environ.get(args.token_env, ""))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
