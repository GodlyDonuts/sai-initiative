"""Fail-closed 300M/1B Sai training entry point for an externally selected family.

This module selects neither a mixer family nor a scale winner.  It only admits a
300M or 1B run after a hash-bound, source-disjoint matched-control benchmark gate
names the exact family.  It reuses the receipt-bound single-GPU training core and
cannot authorize or execute a 4B model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256
from sai.model.planner import SaiModelPlanError, validate_plan
from sai.training.short_screen import FAMILIES, SUB4B_SCALES, ShortScreenError
from sai.training.short_screen import run as run_receipt_bound_training

ADMISSION_SCHEMA = "sai-sub-4b-scale-promotion-v1"
_ADMISSION_KEYS = {
    "schema",
    "target_scale",
    "prior_scale",
    "selected_family",
    "real_development_benchmark_gate_passed",
    "matched_equal_compute_control",
    "source_disjoint_evaluation",
    "evidence_receipt_sha256",
    "receipt_sha256",
}
_PRIOR_SCALE = {"300m": "100m", "1b": "300m"}


class ScaleTrainingError(ShortScreenError):
    """A scale admission receipt or generic run argument differs."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ScaleTrainingError(f"{field} must be a lowercase SHA256")
    return value


def load_scale_admission(path: Path, *, scale: str, family: str) -> dict[str, Any]:
    """Validate the external evidence decision without manufacturing a winner."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ScaleTrainingError("scale admission receipt is missing or unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScaleTrainingError("scale admission receipt is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != _ADMISSION_KEYS:
        raise ScaleTrainingError("scale admission receipt keys differ")
    if scale not in SUB4B_SCALES or family not in FAMILIES:
        raise ScaleTrainingError("scale or family is outside the sub-4B ladder")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload["receipt_sha256"] != canonical_sha256(unsigned):
        raise ScaleTrainingError("scale admission receipt hash differs")
    _sha256(payload["evidence_receipt_sha256"], "benchmark evidence receipt")
    _sha256(payload["receipt_sha256"], "scale admission receipt")
    if (
        payload["schema"] != ADMISSION_SCHEMA
        or payload["target_scale"] != scale
        or payload["prior_scale"] != _PRIOR_SCALE[scale]
        or payload["selected_family"] != family
        or payload["real_development_benchmark_gate_passed"] is not True
        or payload["matched_equal_compute_control"] is not True
        or payload["source_disjoint_evaluation"] is not True
    ):
        raise ScaleTrainingError("scale admission evidence differs")
    return payload


def validate_scale_geometry_plan(path: Path) -> dict[str, Any]:
    """Require the exact deterministic, self-hashed geometry plan."""

    try:
        geometry_payload = json.loads(Path(path).read_text())
        return validate_plan(geometry_payload)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        SaiModelPlanError,
    ) as error:
        raise ScaleTrainingError("deterministic geometry plan differs") from error


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Bind a validated promotion decision, then invoke the generic training core."""

    validate_scale_geometry_plan(args.geometry)
    admission = load_scale_admission(
        args.admission_receipt, scale=args.scale, family=args.family
    )
    args.promotion_receipt_sha256 = admission["receipt_sha256"]
    return run_receipt_bound_training(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=SUB4B_SCALES, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--train-stream", type=Path, required=True)
    parser.add_argument("--train-identity", required=True)
    parser.add_argument("--development-stream", type=Path, required=True)
    parser.add_argument("--development-identity", required=True)
    parser.add_argument("--development-sequences", type=int, required=True)
    parser.add_argument("--development-batch-size", type=int, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--training-sequences", type=int, required=True)
    parser.add_argument("--sequences-per-update", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mechanics-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "run_sha256": payload["run_sha256"],
                "scale": payload["scale"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
