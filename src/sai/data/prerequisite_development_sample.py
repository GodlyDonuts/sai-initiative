"""Select a semantic-prerequisite audit from a qualified development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sai.data.curriculum import BANDS, PHASES
from sai.data.prerequisite_sample import (
    ACTIVE_STRATA,
    EXCLUDED_STRATA,
    SELECTION_SALT,
    PrerequisiteSampleError,
    _encoded_rows,
    _select,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-semantic-prerequisite-development-audit-population-v1"
SPLIT_SCHEMA = "sai-curriculum-train-development-split-v1"
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "source_split",
    "selection",
    "output",
    "limitations",
    "receipt_sha256",
}


class PrerequisiteDevelopmentSampleError(RuntimeError):
    """The split, development population, selection, or receipt differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrerequisiteDevelopmentSampleError(message)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} differs",
    )
    return value


def _load_split(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "split receipt is missing or unsafe",
    )
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteDevelopmentSampleError(
            "split receipt is unreadable"
        ) from error
    _require(isinstance(payload, dict), "split receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    _require(
        payload.get("receipt_sha256") == canonical_sha256(unsigned),
        "split receipt self hash differs",
    )
    expected_checks = {
        "all_curriculum_documents_emitted_once": True,
        "both_populations_have_every_phase": True,
        "exact_identity_assignment_disjoint": True,
        "train_progression_qualified": True,
    }
    development = payload.get("development")
    _require(
        payload.get("schema") == SPLIT_SCHEMA
        and payload.get("status") == "qualified"
        and payload.get("split_qualified") is True
        and payload.get("training_authorized") is False
        and payload.get("four_b_training_authorized") is False
        and payload.get("checks") == expected_checks
        and isinstance(development, dict)
        and development.get("curriculum_qualified") is True,
        "split receipt does not admit development sampling",
    )
    source = Path(development.get("path", ""))
    phases = development.get("phases")
    _sha256(development.get("sha256"), "development SHA256")
    _sha256(development.get("identity_sha256"), "development identity")
    _require(
        source.is_absolute()
        and source.is_file()
        and not source.is_symlink()
        and source.stat().st_nlink == 1
        and source.stat().st_size == development.get("bytes")
        and sha256_file(source) == development.get("sha256")
        and isinstance(phases, dict)
        and list(phases) == list(PHASES),
        "development population differs",
    )
    total = 0
    for index, phase in enumerate(PHASES):
        row = phases.get(phase)
        _require(
            isinstance(row, dict)
            and row.get("index") == index
            and isinstance(row.get("documents"), int)
            and not isinstance(row.get("documents"), bool)
            and row["documents"] > 0,
            "development phase geometry differs",
        )
        total += row["documents"]
    grounding = phases["grounding"]
    _require(
        isinstance(grounding.get("by_band"), dict)
        and grounding["by_band"].get("specialization") == 0
        and isinstance(development.get("progression_checks"), dict)
        and development["progression_checks"].get("grounding_has_no_specialization")
        is True,
        "development excluded stratum is not structurally empty",
    )
    _require(
        total == development.get("documents"), "development document total differs"
    )
    curriculum_view = {
        "output": {"path": str(source.resolve())},
        "phases": {
            phase: {"documents": phases[phase]["documents"]} for phase in PHASES
        },
    }
    return payload, curriculum_view


def _source_descriptor(split_receipt: Path, split: dict[str, Any]) -> dict[str, Any]:
    development = split["development"]
    return {
        "split_receipt_path": str(split_receipt.resolve()),
        "split_receipt_file_sha256": sha256_file(split_receipt),
        "split_receipt_sha256": split["receipt_sha256"],
        "development_path": str(Path(development["path"]).resolve()),
        "development_bytes": development["bytes"],
        "development_sha256": development["sha256"],
        "development_documents": development["documents"],
        "development_identity_sha256": development["identity_sha256"],
    }


def build_development_audit_population(
    split_receipt: Path,
    output: Path,
    receipt: Path,
    *,
    per_stratum: int = 8,
) -> dict[str, Any]:
    """Publish a deterministic unreviewed sample without replaying train bytes."""

    _require(output.parent == receipt.parent, "audit outputs must share one parent")
    _require(
        not any(path.exists() or path.is_symlink() for path in (output, receipt)),
        "audit output already exists",
    )
    split, curriculum_view = _load_split(split_receipt)
    try:
        rows, candidates = _select(
            curriculum_view,
            per_stratum=per_stratum,
            active_strata=ACTIVE_STRATA,
        )
    except PrerequisiteSampleError as error:
        raise PrerequisiteDevelopmentSampleError(
            "development audit stratum differs"
        ) from error
    encoded = _encoded_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"partial.{os.getpid()}"
    output_stage = output.with_name(f".{output.name}.{suffix}")
    receipt_stage = receipt.with_name(f".{receipt.name}.{suffix}")
    try:
        with output_stage.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "selected_unreviewed",
            "training_authorized": False,
            "four_b_training_authorized": False,
            "source_split": _source_descriptor(split_receipt, split),
            "selection": {
                "method": "lowest_identity_hash_per_development_phase_and_surface_band",
                "salt_sha256": hashlib.sha256(SELECTION_SALT).hexdigest(),
                "phases": list(PHASES),
                "surface_bands": list(BANDS),
                "excluded_structurally_empty_strata": [
                    f"{phase}:{band}" for phase, band in EXCLUDED_STRATA
                ],
                "per_stratum": per_stratum,
                "strata": len(ACTIVE_STRATA),
                "selected_documents": len(rows),
                "candidate_documents": candidates,
            },
            "output": {
                "path": str(output.resolve()),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "ordered_population_sha256": canonical_sha256(rows),
            },
            "limitations": [
                "population_is_selected_not_reviewed",
                "development_split_is_source_disjoint_from_training",
                "surface_band_is_not_semantic_difficulty",
                "receipt_authorizes_no_training",
            ],
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        with receipt_stage.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(output_stage, output)
        os.replace(receipt_stage, receipt)
        return payload
    except BaseException:
        output_stage.unlink(missing_ok=True)
        receipt_stage.unlink(missing_ok=True)
        raise


def validate_development_audit_population(receipt: Path) -> dict[str, Any]:
    """Recompute the exact selection from the immutable development split."""

    _require(
        receipt.is_file() and not receipt.is_symlink() and receipt.stat().st_nlink == 1,
        "audit receipt is missing or unsafe",
    )
    try:
        payload = json.loads(receipt.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrerequisiteDevelopmentSampleError(
            "audit receipt is unreadable"
        ) from error
    _require(
        isinstance(payload, dict)
        and set(payload) == _TOP_KEYS
        and payload.get("schema") == SCHEMA
        and payload.get("receipt_sha256")
        == canonical_sha256(
            {key: value for key, value in payload.items() if key != "receipt_sha256"}
        )
        and payload.get("status") == "selected_unreviewed"
        and payload.get("training_authorized") is False
        and payload.get("four_b_training_authorized") is False,
        "audit receipt differs",
    )
    source = payload.get("source_split")
    _require(isinstance(source, dict), "audit source differs")
    split_receipt = Path(source.get("split_receipt_path", ""))
    split, curriculum_view = _load_split(split_receipt)
    _require(
        source == _source_descriptor(split_receipt, split),
        "audit split lineage differs",
    )
    selection = payload.get("selection")
    _require(isinstance(selection, dict), "audit selection differs")
    per_stratum = selection.get("per_stratum")
    try:
        rows, candidates = _select(
            curriculum_view,
            per_stratum=per_stratum,
            active_strata=ACTIVE_STRATA,
        )
    except PrerequisiteSampleError as error:
        raise PrerequisiteDevelopmentSampleError(
            "development audit stratum differs"
        ) from error
    encoded = _encoded_rows(rows)
    expected_selection = {
        "method": "lowest_identity_hash_per_development_phase_and_surface_band",
        "salt_sha256": hashlib.sha256(SELECTION_SALT).hexdigest(),
        "phases": list(PHASES),
        "surface_bands": list(BANDS),
        "excluded_structurally_empty_strata": [
            f"{phase}:{band}" for phase, band in EXCLUDED_STRATA
        ],
        "per_stratum": per_stratum,
        "strata": len(ACTIVE_STRATA),
        "selected_documents": len(rows),
        "candidate_documents": candidates,
    }
    output = Path(payload.get("output", {}).get("path", ""))
    _require(
        selection == expected_selection
        and output.is_file()
        and not output.is_symlink()
        and output.stat().st_nlink == 1
        and output.read_bytes() == encoded
        and payload.get("output")
        == {
            "path": str(output.resolve()),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_population_sha256": canonical_sha256(rows),
        }
        and payload.get("limitations")
        == [
            "population_is_selected_not_reviewed",
            "development_split_is_source_disjoint_from_training",
            "surface_band_is_not_semantic_difficulty",
            "receipt_authorizes_no_training",
        ],
        "audit population replay differs",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--split-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--per-stratum", type=int, default=8)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_development_audit_population(
            args.split_receipt,
            args.output,
            args.receipt,
            per_stratum=args.per_stratum,
        )
    else:
        payload = validate_development_audit_population(args.receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_documents": payload["selection"]["selected_documents"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
