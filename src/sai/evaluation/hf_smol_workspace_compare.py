"""Compare SmolLM3 parent, recurrent, and matched-reset development results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256
from sai.evaluation.hf_workspace_compare import (
    BENCHMARKS,
    HFWorkspaceComparisonError,
    compare,
    write_comparison,
)
from sai.evaluation.hf_workspace_compare import (
    SCHEMA as QWEN_COMPARISON_SCHEMA,
)
from sai.training.hf_smol_workspace_screen import SCHEMA as TRAINING_SCHEMA

SCHEMA = "sai-smollm3-3b-matched-workspace-comparison-v1"


def _load_qwen_factor_receipt(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise HFWorkspaceComparisonError("Qwen factor receipt is missing or unsafe")
    encoded = path.read_bytes()
    try:
        receipt = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFWorkspaceComparisonError("Qwen factor receipt is unreadable") from error
    if not isinstance(receipt, dict):
        raise HFWorkspaceComparisonError("Qwen factor receipt differs")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    checks = receipt.get("checks")
    if (
        receipt.get("schema") != QWEN_COMPARISON_SCHEMA
        or receipt.get("status") != "complete"
        or claimed != canonical_sha256(unsigned)
        or receipt.get("pass") is not True
        or receipt.get("action") != "authorize_sub4b_confirmation"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or receipt.get("architecture_locked") is not False
        or receipt.get("four_b_training_executed") is not False
        or receipt.get("four_b_training_authorized") is not False
    ):
        raise HFWorkspaceComparisonError("Qwen factor receipt did not pass exactly")
    return receipt, hashlib.sha256(encoded).hexdigest()


def compare_cross_family(
    *,
    qwen_factor_receipt: Path,
    parent_paths: dict[str, Path],
    recurrent_paths: dict[str, Path],
    reset_paths: dict[str, Path],
    recurrent_training_result: Path,
    reset_training_result: Path,
) -> dict[str, Any]:
    """Bind a passing Qwen factor receipt to the exact Smol matched decision."""

    qwen, qwen_file_sha256 = _load_qwen_factor_receipt(qwen_factor_receipt)
    payload = compare(
        parent_paths=parent_paths,
        recurrent_paths=recurrent_paths,
        reset_paths=reset_paths,
        recurrent_training_result=recurrent_training_result,
        reset_training_result=reset_training_result,
        training_schema=TRAINING_SCHEMA,
        comparison_schema=SCHEMA,
        pass_action="cross_family_factor_confirmed_await_user_4b_authorization",
        fail_action="reject_recurrent_workspace_cross_family",
        claim_limit=(
            "A pass confirms this factor on 0.8B Qwen and 3B Smol hosts but still "
            "does not authorize or execute 4B training. A failure rejects transfer."
        ),
    )
    payload.pop("receipt_sha256")
    payload["qwen_factor_evidence"] = {
        "path": str(Path(qwen_factor_receipt).resolve()),
        "file_sha256": qwen_file_sha256,
        "receipt_sha256": qwen["receipt_sha256"],
        "schema": qwen["schema"],
        "pass": True,
        "action": qwen["action"],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for benchmark in BENCHMARKS:
        parser.add_argument(
            f"--parent-{benchmark.replace('_', '-')}", type=Path, required=True
        )
        parser.add_argument(
            f"--recurrent-{benchmark.replace('_', '-')}", type=Path, required=True
        )
        parser.add_argument(
            f"--reset-{benchmark.replace('_', '-')}", type=Path, required=True
        )
    parser.add_argument("--recurrent-training-result", type=Path, required=True)
    parser.add_argument("--reset-training-result", type=Path, required=True)
    parser.add_argument("--qwen-factor-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_cross_family(
        qwen_factor_receipt=args.qwen_factor_receipt,
        parent_paths={
            benchmark: getattr(args, f"parent_{benchmark}") for benchmark in BENCHMARKS
        },
        recurrent_paths={
            benchmark: getattr(args, f"recurrent_{benchmark}")
            for benchmark in BENCHMARKS
        },
        reset_paths={
            benchmark: getattr(args, f"reset_{benchmark}") for benchmark in BENCHMARKS
        },
        recurrent_training_result=args.recurrent_training_result,
        reset_training_result=args.reset_training_result,
    )
    write_comparison(args.output, payload)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "action": payload["action"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
