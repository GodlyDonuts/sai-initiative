"""Exact non-4B geometry adapter for Sai checkpoint evaluation.

Training runners may use different budgets at 100M, 300M, and 1B, but their
completed mechanics checkpoint format and real development boards are shared.
This module resolves only deterministic checked-in geometry; it never permits
the sealed 4B row or authorizes training.
"""

from __future__ import annotations

import json
from pathlib import Path

from sai.model.config import SaiModelConfig, parameter_ledger
from sai.model.planner import SaiModelPlanError, validate_plan

EVALUATION_SCALES = ("100m", "300m", "1b")
FAMILIES = ("gated_gqa", "gdn_hybrid", "kda_mla_hybrid")
TARGET_PARAMETERS = {"100m": 100_000_000, "300m": 300_000_000, "1b": 1_000_000_000}
MAX_RELATIVE_ERROR = 0.01


class ScaleCheckpointError(RuntimeError):
    """A scale, family, or deterministic geometry identity differs."""


def load_evaluation_config(
    path: Path, family: str, scale: str
) -> tuple[SaiModelConfig, dict]:
    """Resolve one exact non-4B row from the deterministic geometry plan."""

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ScaleCheckpointError("geometry artifact is missing or unsafe")
    if scale not in EVALUATION_SCALES:
        raise ScaleCheckpointError("evaluation scale must be 100m, 300m, or 1b")
    if family not in FAMILIES:
        raise ScaleCheckpointError("mixer family differs")
    try:
        payload = json.loads(path.read_text())
        payload = validate_plan(payload)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        SaiModelPlanError,
    ) as error:
        raise ScaleCheckpointError("deterministic geometry plan differs") from error
    rows = [
        row
        for row in payload["geometries"]
        if row.get("scale") == scale and row.get("mixer_family") == family
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("config"), dict):
        raise ScaleCheckpointError("scale/family geometry is not unique")
    row = rows[0]
    config = SaiModelConfig(**row["config"])
    ledger = parameter_ledger(config)
    target = TARGET_PARAMETERS[scale]
    relative_error = (ledger["total"] - target) / target
    if (
        config.mixer_family != family
        or row.get("target_parameters") != target
        or row.get("parameter_ledger") != ledger
        or row.get("relative_error") != relative_error
        or abs(relative_error) > MAX_RELATIVE_ERROR
    ):
        raise ScaleCheckpointError("scale geometry ledger differs")
    return config, row
