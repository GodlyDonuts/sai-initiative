"""Build and validate Sai's exact 4T-token 1B spiral contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-1b-4t-spiral-contract-v1"
BANDS = ("foundation", "intermediate", "advanced", "expert")
SEQUENCE_LENGTH = 2_048
TOTAL_TOKENS = 4_000_000_000_000
STAGES = (
    ("foundation", 1_000_000_000_000, (65, 25, 8, 2)),
    ("expansion", 1_400_000_000_000, (40, 40, 15, 5)),
    ("depth", 1_000_000_000_000, (20, 40, 30, 10)),
    ("synthesis", 400_000_000_000, (10, 25, 40, 25)),
    ("annealing", 200_000_000_000, (10, 20, 35, 35)),
)


class OneBSpiralContractError(RuntimeError):
    """The 1B token horizon, stages, or spiral allocations differ."""


def _allocate(total: int, weights: tuple[int, ...]) -> list[int]:
    """Allocate integer sequences by stable largest-remainder rounding."""

    denominator = sum(weights)
    floors = [total * value // denominator for value in weights]
    remainders = [total * value % denominator for value in weights]
    missing = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda index: (-remainders[index], index))
    for index in order[:missing]:
        floors[index] += 1
    return floors


def build_contract() -> dict[str, Any]:
    """Return the canonical prospective 1B spiral contract."""

    stages = []
    cumulative_tokens = 0
    for index, (name, tokens, weights) in enumerate(STAGES):
        if tokens % SEQUENCE_LENGTH:
            raise OneBSpiralContractError("stage boundary is not sequence aligned")
        sequences = tokens // SEQUENCE_LENGTH
        band_sequences = _allocate(sequences, weights)
        cumulative_tokens += tokens
        stages.append(
            {
                "index": index,
                "stage": name,
                "tokens": tokens,
                "sequences": sequences,
                "cumulative_tokens": cumulative_tokens,
                "band_percentages": dict(zip(BANDS, weights, strict=True)),
                "band_sequences": dict(zip(BANDS, band_sequences, strict=True)),
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "prospective_1b_spiral_contract_complete",
        "objective": "one_billion_parameter_english_polymath",
        "target_tokens": TOTAL_TOKENS,
        "sequence_length": SEQUENCE_LENGTH,
        "target_sequences": TOTAL_TOKENS // SEQUENCE_LENGTH,
        "stages": stages,
        "foundational_rehearsal_in_every_stage": True,
        "expert_exposure_in_every_stage": True,
        "deterministic_largest_remainder_allocation": True,
        "maximum_connection_document_exposures": 16,
        "bulk_internal_development_fraction_ppm": 1_000,
        "production_tokenizer_capacity": 48_000,
        "production_tokenizer_selected": False,
        "curriculum_index_complete": False,
        "packed_stream_smoke_complete": False,
        "model_training_started": False,
        "one_b_training_authorized": False,
        "four_b_target_retired": True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_contract(payload: Any) -> dict[str, Any]:
    """Recompute every immutable token and allocation invariant."""

    expected = build_contract()
    if payload != expected:
        raise OneBSpiralContractError("1B spiral contract differs")
    if (
        sum(stage["tokens"] for stage in payload["stages"]) != TOTAL_TOKENS
        or sum(stage["sequences"] for stage in payload["stages"])
        != TOTAL_TOKENS // SEQUENCE_LENGTH
        or any(
            sum(stage["band_sequences"].values()) != stage["sequences"]
            for stage in payload["stages"]
        )
        or any(
            stage["band_sequences"]["foundation"] <= 0
            for stage in payload["stages"]
        )
        or any(
            stage["band_sequences"]["expert"] <= 0
            for stage in payload["stages"]
        )
    ):
        raise OneBSpiralContractError("1B spiral arithmetic differs")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_contract(build_contract())
    if args.output.exists() or args.output.is_symlink():
        raise OneBSpiralContractError("contract output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
