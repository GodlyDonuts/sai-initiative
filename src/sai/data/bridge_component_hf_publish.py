"""Publish the admitted train-only connection component to Godlydonuts/Sai."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bridge_component_admission import SCHEMA as ADMISSION_SCHEMA
from sai.data.pleias_production_materializer import upload_verified
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bridge-training-component-hf-publication-v1"
DESTINATION_REPOSITORY = "Godlydonuts/Sai"
DESTINATION_PREFIX = "training/final/cross-domain-connections/20260826-r1"


class BridgeComponentHfPublishError(RuntimeError):
    """An admitted component, local file, or remote LFS identity differs."""


def _load_admission(root: Path) -> dict[str, Any]:
    path = root / "receipt.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BridgeComponentHfPublishError("admission receipt is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeComponentHfPublishError("admission receipt differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != ADMISSION_SCHEMA
        or payload.get("status") != "complete_bridge_training_component_admission"
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("development_rows_physically_excluded") is not True
        or payload.get("transfer_ablation_complete") is not True
        or payload.get("connection_component_admission_authorized") is not True
        or payload.get("training_ready") is not True
        or payload.get("four_b_training_authorized") is not False
    ):
        raise BridgeComponentHfPublishError("admission receipt differs")
    return payload


def publish(admission_root: Path, output: Path, token: str) -> dict[str, Any]:
    """Upload and independently replay the exact train gzip and admission receipt."""

    if output.exists() or output.is_symlink() or not token:
        raise BridgeComponentHfPublishError("publication arguments differ")
    admission = _load_admission(admission_root)
    descriptor = admission.get("train")
    if not isinstance(descriptor, dict):
        raise BridgeComponentHfPublishError("train descriptor differs")
    train = admission_root / str(descriptor.get("path"))
    receipt = admission_root / "receipt.json"
    if (
        not train.is_file()
        or train.is_symlink()
        or train.stat().st_nlink != 1
        or train.stat().st_size != descriptor.get("bytes")
        or sha256_file(train) != descriptor.get("sha256")
    ):
        raise BridgeComponentHfPublishError("admitted train file differs")
    remotes = [
        upload_verified(
            train,
            f"{DESTINATION_PREFIX}/train.jsonl.gz",
            token,
            repository=DESTINATION_REPOSITORY,
        ),
        upload_verified(
            receipt,
            f"{DESTINATION_PREFIX}/receipt.json",
            token,
            repository=DESTINATION_REPOSITORY,
        ),
    ]
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise BridgeComponentHfPublishError("huggingface_hub is required") from error
    info = HfApi(token=token).dataset_info(DESTINATION_REPOSITORY, files_metadata=True)
    siblings = {row.rfilename: row for row in info.siblings or []}
    for remote in remotes:
        sibling = siblings.get(remote["path"])
        lfs = None if sibling is None else sibling.lfs
        if (
            sibling is None
            or sibling.size != remote["bytes"]
            or lfs is None
            or lfs.size != remote["bytes"]
            or lfs.sha256 != remote["sha256"]
        ):
            raise BridgeComponentHfPublishError("remote LFS replay differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_bridge_training_component_hf_publication",
        "admission_receipt_sha256": admission["receipt_sha256"],
        "remote_repository": DESTINATION_REPOSITORY,
        "remote_revision_verified": info.sha,
        "remote_prefix": DESTINATION_PREFIX,
        "remote_outputs": remotes,
        "train_documents": admission["counts"]["train_documents"],
        "train_text_utf8_bytes": descriptor["text_utf8_bytes"],
        "development_rows_uploaded": False,
        "transfer_ablation_complete": True,
        "connection_component_admission_authorized": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = publish(
        args.admission_root,
        args.output,
        os.environ.get(args.token_env, ""),
    )
    print(
        json.dumps(
            {
                "remote_revision": result["remote_revision_verified"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
