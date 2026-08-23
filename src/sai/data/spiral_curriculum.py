"""Build and validate Sai's prerequisite-aware spiral curriculum policy."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _bounded_int, _exact
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-spiral-curriculum-policy-v1"
PHASES = ("early", "foundation", "growth", "advanced", "annealing")
BANDS = ("basic", "intermediate", "advanced", "expert")
SHARES_PPM = {
    "early": {
        "basic": 650_000,
        "intermediate": 250_000,
        "advanced": 80_000,
        "expert": 20_000,
    },
    "foundation": {
        "basic": 400_000,
        "intermediate": 400_000,
        "advanced": 150_000,
        "expert": 50_000,
    },
    "growth": {
        "basic": 200_000,
        "intermediate": 400_000,
        "advanced": 300_000,
        "expert": 100_000,
    },
    "advanced": {
        "basic": 100_000,
        "intermediate": 250_000,
        "advanced": 400_000,
        "expert": 250_000,
    },
    "annealing": {
        "basic": 100_000,
        "intermediate": 200_000,
        "advanced": 350_000,
        "expert": 350_000,
    },
}


class SpiralCurriculumError(RuntimeError):
    """A spiral policy or phase allocation differs from the frozen design."""


def _allocate(total: int, shares: dict[str, int]) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 100:
        raise SpiralCurriculumError("phase sequence budget differs")
    exact = {band: total * shares[band] for band in BANDS}
    result = {band: exact[band] // 1_000_000 for band in BANDS}
    remaining = total - sum(result.values())
    order = sorted(
        BANDS,
        key=lambda band: (-(exact[band] % 1_000_000), BANDS.index(band)),
    )
    for band in order[:remaining]:
        result[band] += 1
    if result["basic"] <= 0 or sum(result.values()) != total:
        raise SpiralCurriculumError("spiral allocation lost foundation rehearsal")
    return result


def build_policy(phase_sequences: dict[str, int]) -> dict[str, Any]:
    """Create a prospective schedule without collapsing complexity to one scalar."""

    if not isinstance(phase_sequences, dict) or set(phase_sequences) != set(PHASES):
        raise SpiralCurriculumError("phase sequence fields differ")
    allocations = {
        phase: _allocate(phase_sequences[phase], SHARES_PPM[phase]) for phase in PHASES
    }
    payload = {
        "schema": SCHEMA,
        "status": "prospective",
        "phase_order": list(PHASES),
        "bands": list(BANDS),
        "shares_ppm": SHARES_PPM,
        "phase_sequences": phase_sequences,
        "phase_band_sequences": allocations,
        "complexity_axes": [
            "linguistic_complexity",
            "conceptual_complexity",
            "reasoning_complexity",
        ],
        "scalar_difficulty_used": False,
        "band_assignment_basis": "prerequisite_graph_readiness",
        "prerequisite_admission": {
            "acyclic_graph_required": True,
            "edge_evidence_required": True,
            "all_prerequisites_seen_before_or_rehearsed_in_phase": True,
            "minimum_prior_exposure_is_taxonomy_bound": True,
        },
        "foundation_rehearsal_in_every_phase": True,
        "mixed_sampling_after_curriculum_warmup": True,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return validate_policy(payload)


def validate_policy(payload: Any) -> dict[str, Any]:
    """Validate the exact spiral shares, graph boundary, and self hash."""

    row = _exact(
        payload,
        {
            "schema",
            "status",
            "phase_order",
            "bands",
            "shares_ppm",
            "phase_sequences",
            "phase_band_sequences",
            "complexity_axes",
            "scalar_difficulty_used",
            "band_assignment_basis",
            "prerequisite_admission",
            "foundation_rehearsal_in_every_phase",
            "mixed_sampling_after_curriculum_warmup",
            "training_authorized",
            "four_b_training_authorized",
            "receipt_sha256",
        },
        "spiral policy",
    )
    if (
        row["schema"] != SCHEMA
        or row["status"] != "prospective"
        or row["phase_order"] != list(PHASES)
        or row["bands"] != list(BANDS)
        or row["shares_ppm"] != SHARES_PPM
        or row["complexity_axes"]
        != [
            "linguistic_complexity",
            "conceptual_complexity",
            "reasoning_complexity",
        ]
        or row["scalar_difficulty_used"] is not False
        or row["band_assignment_basis"] != "prerequisite_graph_readiness"
        or row["foundation_rehearsal_in_every_phase"] is not True
        or row["mixed_sampling_after_curriculum_warmup"] is not True
        or row["training_authorized"] is not False
        or row["four_b_training_authorized"] is not False
    ):
        raise SpiralCurriculumError("spiral policy contract differs")
    prerequisite = _exact(
        row["prerequisite_admission"],
        {
            "acyclic_graph_required",
            "edge_evidence_required",
            "all_prerequisites_seen_before_or_rehearsed_in_phase",
            "minimum_prior_exposure_is_taxonomy_bound",
        },
        "prerequisite admission",
    )
    if any(value is not True for value in prerequisite.values()):
        raise SpiralCurriculumError("prerequisite admission differs")
    sequences = _exact(row["phase_sequences"], set(PHASES), "phase sequences")
    allocations = _exact(row["phase_band_sequences"], set(PHASES), "phase allocations")
    for phase in PHASES:
        expected = _allocate(sequences[phase], SHARES_PPM[phase])
        actual = _exact(allocations[phase], set(BANDS), f"{phase} allocation")
        for band in BANDS:
            _bounded_int(actual[band], 0, sequences[phase], f"{phase} {band}")
        if actual != expected or actual["basic"] <= 0:
            raise SpiralCurriculumError("phase band allocation differs")
    unsigned = {key: value for key, value in row.items() if key != "receipt_sha256"}
    if row["receipt_sha256"] != canonical_sha256(unsigned):
        raise SpiralCurriculumError("spiral policy receipt hash differs")
    return row


def _atomic_create(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SpiralCurriculumError("spiral policy output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-sequences", type=int, required=True)
    parser.add_argument("--foundation-sequences", type=int, required=True)
    parser.add_argument("--growth-sequences", type=int, required=True)
    parser.add_argument("--advanced-sequences", type=int, required=True)
    parser.add_argument("--annealing-sequences", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = build_policy(
        {
            "early": args.early_sequences,
            "foundation": args.foundation_sequences,
            "growth": args.growth_sequences,
            "advanced": args.advanced_sequences,
            "annealing": args.annealing_sequences,
        }
    )
    _atomic_create(args.output, policy)
    print(json.dumps(policy, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
