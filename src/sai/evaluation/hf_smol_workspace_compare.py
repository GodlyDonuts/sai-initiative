"""Compare SmolLM3 parent, recurrent, and matched-reset development results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sai.evaluation.hf_workspace_compare import (
    BENCHMARKS,
    compare,
    write_comparison,
)
from sai.training.hf_smol_workspace_screen import SCHEMA as TRAINING_SCHEMA

SCHEMA = "sai-smollm3-3b-matched-workspace-comparison-v1"


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
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
        training_schema=TRAINING_SCHEMA,
        comparison_schema=SCHEMA,
        pass_action="cross_family_factor_confirmed_await_user_4b_authorization",
        fail_action="reject_recurrent_workspace_cross_family",
        claim_limit=(
            "A pass confirms this factor on 0.8B Qwen and 3B Smol hosts but still "
            "does not authorize or execute 4B training. A failure rejects transfer."
        ),
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
