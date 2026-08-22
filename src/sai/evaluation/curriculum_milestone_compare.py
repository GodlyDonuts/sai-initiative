"""Compare matched curriculum/control acquisition and retention curves."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file
from sai.evaluation.curriculum_milestone_nll import SCHEMA as MILESTONE_SCHEMA

SCHEMA = "sai-curriculum-milestone-comparison-v1"


class CurriculumMilestoneComparisonError(RuntimeError):
    """A curve, matched identity, or phase comparison differs."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise CurriculumMilestoneComparisonError("output parent or target is unsafe")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise CurriculumMilestoneComparisonError("output already exists") from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise CurriculumMilestoneComparisonError("milestone receipt is unsafe")
    file_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurriculumMilestoneComparisonError(
            "milestone receipt is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise CurriculumMilestoneComparisonError("milestone receipt differs")
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if (
        payload.get("schema") != MILESTONE_SCHEMA
        or payload.get("status") != "complete"
        or receipt_sha256 != canonical_sha256(unsigned)
        or payload.get("optimizer_steps") != 0
        or payload.get("backward_calls") != 0
        or payload.get("training_authorized") is not False
        or payload.get("architecture_promotion_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
    ):
        raise CurriculumMilestoneComparisonError("milestone receipt differs")
    return payload, file_sha256


def _score(row: dict[str, Any], phase: str) -> float:
    value = (
        row.get("development_nll", {})
        .get("strata", {})
        .get(phase, {})
        .get("nll_per_target")
    )
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CurriculumMilestoneComparisonError("phase score differs")
    return float(value)


def compare_curves(
    curriculum: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    """Require exact matching and compute prospective phasewise decisions."""

    if curriculum.get("development_stream") != control.get("development_stream"):
        raise CurriculumMilestoneComparisonError("development population differs")
    if curriculum.get("milestone_steps") != control.get("milestone_steps"):
        raise CurriculumMilestoneComparisonError("milestone schedule differs")
    curriculum_curve = curriculum.get("learning_curve")
    control_curve = control.get("learning_curve")
    if not isinstance(curriculum_curve, dict) or not isinstance(control_curve, dict):
        raise CurriculumMilestoneComparisonError("learning curve differs")
    if (
        curriculum_curve.get("observation_steps")
        != control_curve.get("observation_steps")
        or curriculum_curve.get("phase_order") != control_curve.get("phase_order")
        or not isinstance(curriculum.get("observations"), list)
        or not isinstance(control.get("observations"), list)
        or len(curriculum["observations"]) != len(control["observations"])
    ):
        raise CurriculumMilestoneComparisonError("learning-curve geometry differs")
    phases = curriculum_curve["phase_order"]
    steps = curriculum_curve["observation_steps"]
    if not isinstance(phases, list) or not phases or not isinstance(steps, list):
        raise CurriculumMilestoneComparisonError("learning-curve geometry differs")
    curriculum_by_step = {
        row.get("optimizer_step"): row for row in curriculum["observations"]
    }
    control_by_step = {
        row.get("optimizer_step"): row for row in control["observations"]
    }
    if set(curriculum_by_step) != set(steps) or set(control_by_step) != set(steps):
        raise CurriculumMilestoneComparisonError("learning-curve steps differ")
    if curriculum["observations"][0].get("model_state_sha256") != control[
        "observations"
    ][0].get("model_state_sha256") or any(
        _score(curriculum_by_step[0], phase) != _score(control_by_step[0], phase)
        for phase in phases
    ):
        raise CurriculumMilestoneComparisonError("initial model evidence differs")
    curriculum_run = curriculum.get("training_run")
    control_run = control.get("training_run")
    if (
        not isinstance(curriculum_run, dict)
        or not isinstance(control_run, dict)
        or curriculum_run.get("model_sha256") != control_run.get("model_sha256")
        or curriculum_run.get("run_sha256") == control_run.get("run_sha256")
        or curriculum_run.get("training_stream_identity_sha256")
        == control_run.get("training_stream_identity_sha256")
    ):
        raise CurriculumMilestoneComparisonError("matched training identity differs")

    curriculum_phase_summary = curriculum_curve.get("phases")
    control_phase_summary = control_curve.get("phases")
    if (
        not isinstance(curriculum_phase_summary, dict)
        or set(curriculum_phase_summary) != set(phases)
        or not isinstance(control_phase_summary, dict)
        or set(control_phase_summary) != set(phases)
    ):
        raise CurriculumMilestoneComparisonError("phase summary differs")
    rows: dict[str, Any] = {}
    for phase in phases:
        curriculum_phase = curriculum_phase_summary[phase]
        control_phase = control_phase_summary[phase]
        completion_step = curriculum_phase.get("completion_step")
        if (
            completion_step != control_phase.get("completion_step")
            or completion_step not in curriculum_by_step
        ):
            raise CurriculumMilestoneComparisonError("phase completion step differs")
        curriculum_completion = _score(curriculum_by_step[completion_step], phase)
        control_completion = _score(control_by_step[completion_step], phase)
        curriculum_terminal = _score(curriculum_by_step[steps[-1]], phase)
        control_terminal = _score(control_by_step[steps[-1]], phase)
        curriculum_initial = _score(curriculum_by_step[0], phase)
        control_initial = _score(control_by_step[0], phase)
        curriculum_forgetting = curriculum_terminal - curriculum_completion
        control_forgetting = control_terminal - control_completion
        rows[phase] = {
            "completion_step": completion_step,
            "curriculum_acquisition_delta": curriculum_completion - curriculum_initial,
            "control_acquisition_delta": control_completion - control_initial,
            "completion_curriculum_minus_control": curriculum_completion
            - control_completion,
            "curriculum_post_completion_forgetting": curriculum_forgetting,
            "control_post_completion_forgetting": control_forgetting,
            "forgetting_curriculum_minus_control": curriculum_forgetting
            - control_forgetting,
            "terminal_curriculum_minus_control": curriculum_terminal - control_terminal,
            "curriculum_acquired_by_completion": curriculum_completion
            < curriculum_initial,
            "curriculum_no_worse_at_completion": curriculum_completion
            <= control_completion,
            "curriculum_no_more_forgetting": curriculum_forgetting
            <= control_forgetting,
            "curriculum_no_worse_at_terminal": curriculum_terminal <= control_terminal,
        }
    progression_supported = all(
        row["curriculum_acquired_by_completion"]
        and row["curriculum_no_worse_at_completion"]
        and row["curriculum_no_more_forgetting"]
        and row["curriculum_no_worse_at_terminal"]
        for row in rows.values()
    )
    return {
        "phase_order": phases,
        "observation_steps": steps,
        "phases": rows,
        "curriculum_progression_mechanics_supported": progression_supported,
        "real_benchmark_confirmation_still_required": True,
        "data_promotion_authorized": False,
        "four_b_training_authorized": False,
    }


def compare_paths(curriculum_path: Path, control_path: Path) -> dict[str, Any]:
    curriculum, curriculum_file_sha256 = _load(curriculum_path)
    control, control_file_sha256 = _load(control_path)
    comparison = compare_curves(curriculum, control)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "inputs": {
            "curriculum": {
                "path": Path(curriculum_path).name,
                "file_sha256": curriculum_file_sha256,
                "receipt_sha256": curriculum["receipt_sha256"],
            },
            "order_control": {
                "path": Path(control_path).name,
                "file_sha256": control_file_sha256,
                "receipt_sha256": control["receipt_sha256"],
            },
        },
        "comparison": comparison,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_authorized": False,
        "architecture_promotion_authorized": False,
        "four_b_training_authorized": False,
        "claim_limit": (
            "Matched learning-dynamics evidence only; real source-disjoint "
            "benchmark confirmation is mandatory before data promotion."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--order-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_paths(args.curriculum, args.order_control)
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "status": payload["status"],
                "supported": payload["comparison"][
                    "curriculum_progression_mechanics_supported"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
