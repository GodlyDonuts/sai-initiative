"""Seal the immutable Sai 1B OLMo environment without launching training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file
from sai.training.one_b_production_contract import OLMO_COMMIT, OLMO_CORE_COMMIT

SCHEMA = "sai-1b-olmo-environment-receipt-v1"


class OneBEnvironmentReceiptError(RuntimeError):
    """An upstream tree, package version, or environment identity differs."""


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OneBEnvironmentReceiptError("upstream git tree differs") from error


def build(
    olmo_root: Path,
    core_root: Path,
    python_executable: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    """Verify exact clean sources and import the production CUDA packages."""

    if (
        _git(olmo_root, "rev-parse", "HEAD") != OLMO_COMMIT
        or _git(core_root, "rev-parse", "HEAD") != OLMO_CORE_COMMIT
        or _git(olmo_root, "status", "--porcelain")
        or _git(core_root, "status", "--porcelain")
        or not python_executable.is_file()
        or not freeze_path.is_file()
        or freeze_path.is_symlink()
    ):
        raise OneBEnvironmentReceiptError("immutable environment input differs")
    probe = (
        "import json,torch,flash_attn,olmo,olmo_core;"
        "print(json.dumps({"
        "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'flash_attn':flash_attn.__version__,"
        "'olmo':getattr(olmo,'__version__',getattr(olmo,'VERSION',None)),"
        "'olmo_core':getattr(olmo_core,'__version__',"
        "getattr(olmo_core,'VERSION',None))},sort_keys=True))"
    )
    try:
        packages = json.loads(
            subprocess.check_output([str(python_executable), "-c", probe], text=True)
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise OneBEnvironmentReceiptError(
            "production package import differs"
        ) from error
    if (
        not str(packages.get("torch", "")).startswith("2.5.1")
        or packages.get("torch_cuda") != "12.4"
        or packages.get("flash_attn") != "2.6.3"
    ):
        raise OneBEnvironmentReceiptError("production CUDA package versions differ")
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_1b_olmo_environment",
        "upstream": {
            "olmo_commit": OLMO_COMMIT,
            "olmo_tree": _git(olmo_root, "rev-parse", "HEAD^{tree}"),
            "olmo_core_commit": OLMO_CORE_COMMIT,
            "olmo_core_tree": _git(core_root, "rev-parse", "HEAD^{tree}"),
        },
        "python_executable": str(python_executable.absolute()),
        "python_executable_sha256": sha256_file(python_executable.resolve()),
        "pip_freeze": {
            "path": str(freeze_path.resolve()),
            "bytes": freeze_path.stat().st_size,
            "sha256": sha256_file(freeze_path),
        },
        "packages": packages,
        "model_training_started": False,
        "optimizer_update_performed": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise OneBEnvironmentReceiptError("environment receipt output exists")
    value = build(args.olmo_root, args.core_root, args.python, args.freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    _atomic_create(temporary, value)
    os.replace(temporary, args.output)
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
