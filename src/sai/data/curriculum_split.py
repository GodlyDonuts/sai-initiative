"""Split a qualified Sai curriculum into disjoint train and development rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from sai.data.curriculum import (
    BANDS,
    PHASES,
    document_signals,
    validate_curriculum,
)
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-curriculum-train-development-split-v1"
DEVELOPMENT_MODULUS = 100
DEVELOPMENT_BUCKET = 0


class CurriculumSplitError(RuntimeError):
    """The qualified curriculum, split assignment, or artifact differs."""


def _is_development(identity: str) -> bool:
    return int(identity[:16], 16) % DEVELOPMENT_MODULUS == DEVELOPMENT_BUCKET


def _empty_phase(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "documents": 0,
        "by_band": Counter(),
        "difficulty_sum": 0.0,
        "identity": hashlib.sha256(),
    }


def _finish_phase(row: dict[str, Any]) -> dict[str, Any]:
    if row["documents"] <= 0:
        raise CurriculumSplitError("curriculum split produced an empty phase")
    return {
        "index": row["index"],
        "documents": row["documents"],
        "by_band": {band: row["by_band"][band] for band in BANDS},
        "mean_difficulty": row["difficulty_sum"] / row["documents"],
        "identity_sha256": row["identity"].hexdigest(),
    }


def _progression(phases: dict[str, dict[str, Any]]) -> dict[str, bool]:
    means = [phases[phase]["mean_difficulty"] for phase in PHASES]
    first = phases[PHASES[0]]["by_band"]
    last = phases[PHASES[-1]]["by_band"]
    return {
        "phase_mean_difficulty_nondecreasing": all(
            left <= right for left, right in zip(means, means[1:], strict=False)
        ),
        "grounding_has_no_specialization": first["specialization"] == 0,
        "foundation_frontloaded": first["foundation"] > last["foundation"],
        "specialization_backloaded": last["specialization"] > first["specialization"],
        "foundation_rehearsed_in_final_phase": last["foundation"] > 0,
    }


def _output_row(
    path: Path,
    *,
    output_path: Path,
    documents: int,
    phases: dict[str, dict[str, Any]],
    identity: hashlib._Hash,
) -> dict[str, Any]:
    progression = _progression(phases)
    return {
        "path": str(output_path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "documents": documents,
        "identity_sha256": identity.hexdigest(),
        "phases": phases,
        "progression_checks": progression,
        "curriculum_qualified": all(progression.values()),
    }


def build_curriculum_split(
    curriculum_receipt: Path,
    train: Path,
    development: Path,
    receipt: Path,
    *,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Create an exact identity-hash split after global near-deduplication."""

    curriculum = validate_curriculum(curriculum_receipt, workers=curriculum_workers)
    source = Path(curriculum["output"]["path"])
    if any(
        path.exists() or path.is_symlink() for path in (train, development, receipt)
    ):
        raise CurriculumSplitError("curriculum split output already exists")
    if train.parent != development.parent or train.parent != receipt.parent:
        raise CurriculumSplitError("curriculum split outputs must share one parent")
    train.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"partial.{os.getpid()}"
    train_stage = train.with_name(f".{train.name}.{suffix}")
    development_stage = development.with_name(f".{development.name}.{suffix}")
    receipt_stage = receipt.with_name(f".{receipt.name}.{suffix}")
    phase_rows = {
        population: {phase: _empty_phase(index) for index, phase in enumerate(PHASES)}
        for population in ("train", "development")
    }
    population_counts = Counter()
    population_identities = {
        "train": hashlib.sha256(),
        "development": hashlib.sha256(),
    }
    handles: dict[str, TextIO] = {}
    try:
        handles = {
            "train": train_stage.open("w"),
            "development": development_stage.open("w"),
        }
        with source.open() as source_handle:
            for phase in PHASES:
                declared = curriculum["phases"][phase]["documents"]
                for _ in range(declared):
                    line = source_handle.readline()
                    if not line:
                        raise CurriculumSplitError("curriculum source ended early")
                    try:
                        row = normalize_document(json.loads(line))
                    except (json.JSONDecodeError, RuntimeError) as error:
                        raise CurriculumSplitError(
                            "curriculum split row differs"
                        ) from error
                    identity = row["identity_sha256"]
                    population = "development" if _is_development(identity) else "train"
                    encoded = (
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    handles[population].write(encoded)
                    signals = document_signals(row["text"])
                    phase_row = phase_rows[population][phase]
                    phase_row["documents"] += 1
                    phase_row["by_band"][signals["band"]] += 1
                    phase_row["difficulty_sum"] += signals["difficulty"]
                    identity_bytes = bytes.fromhex(identity)
                    phase_row["identity"].update(identity_bytes)
                    population_identities[population].update(identity_bytes)
                    population_counts[population] += 1
            if source_handle.read(1):
                raise CurriculumSplitError("curriculum source has undeclared rows")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        finalized = {
            population: {
                phase: _finish_phase(phase_rows[population][phase]) for phase in PHASES
            }
            for population in ("train", "development")
        }
        train_row = _output_row(
            train_stage,
            output_path=train,
            documents=population_counts["train"],
            phases=finalized["train"],
            identity=population_identities["train"],
        )
        development_row = _output_row(
            development_stage,
            output_path=development,
            documents=population_counts["development"],
            phases=finalized["development"],
            identity=population_identities["development"],
        )
        every_phase = all(
            finalized[population][phase]["documents"] > 0
            for population in ("train", "development")
            for phase in PHASES
        )
        qualified = bool(
            train_row["curriculum_qualified"]
            and every_phase
            and train_row["documents"] + development_row["documents"]
            == curriculum["documents"]["accepted"]
            and train_row["identity_sha256"] != development_row["identity_sha256"]
        )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "qualified" if qualified else "failed",
            "split_qualified": qualified,
            "training_authorized": False,
            "four_b_training_authorized": False,
            "source_curriculum": {
                "receipt_path": str(curriculum_receipt.resolve()),
                "receipt_bytes": curriculum_receipt.stat().st_size,
                "receipt_file_sha256": sha256_file(curriculum_receipt),
                "receipt_sha256": curriculum["receipt_sha256"],
                "output_path": str(source.resolve()),
                "output_bytes": source.stat().st_size,
                "output_sha256": sha256_file(source),
            },
            "policy": {
                "method": "document_identity_modulus_after_global_near_deduplication",
                "development_modulus": DEVELOPMENT_MODULUS,
                "development_bucket": DEVELOPMENT_BUCKET,
                "phase_order_preserved": True,
                "source_high_confidence_near_duplicate_filter_inherited": True,
                "near_duplicate_claim": "high_confidence_not_exhaustive",
            },
            "train": train_row,
            "development": development_row,
            "checks": {
                "all_curriculum_documents_emitted_once": train_row["documents"]
                + development_row["documents"]
                == curriculum["documents"]["accepted"],
                "exact_identity_assignment_disjoint": True,
                "both_populations_have_every_phase": every_phase,
                "train_progression_qualified": train_row["curriculum_qualified"],
            },
            "diagnostics": {
                "development_progression_nondecreasing": development_row[
                    "progression_checks"
                ]["phase_mean_difficulty_nondecreasing"],
                "development_curriculum_shape_matches_training_policy": development_row[
                    "curriculum_qualified"
                ],
            },
            "limitations": [
                "development_is_nonpublic_and_for_data_order_selection_only",
                "near_duplicate_filter_is_high_confidence_not_exhaustive",
                "receipt_does_not_authorize_4b_training",
            ],
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        receipt_stage.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(train_stage, train)
        os.replace(development_stage, development)
        os.replace(receipt_stage, receipt)
        return payload
    except BaseException:
        for handle in handles.values():
            handle.close()
        train_stage.unlink(missing_ok=True)
        development_stage.unlink(missing_ok=True)
        receipt_stage.unlink(missing_ok=True)
        raise


def validate_curriculum_split(
    receipt: Path, *, curriculum_workers: int = 1
) -> dict[str, Any]:
    """Reopen the curriculum and prove every row's exact split assignment."""

    if not receipt.is_file() or receipt.is_symlink():
        raise CurriculumSplitError("curriculum split receipt is missing or unsafe")
    payload = json.loads(receipt.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise CurriculumSplitError("curriculum split schema differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise CurriculumSplitError("curriculum split receipt hash differs")
    source_row = payload.get("source_curriculum", {})
    source_receipt = Path(source_row.get("receipt_path", ""))
    curriculum = validate_curriculum(source_receipt, workers=curriculum_workers)
    source = Path(curriculum["output"]["path"])
    if (
        source_row.get("receipt_bytes") != source_receipt.stat().st_size
        or source_row.get("receipt_file_sha256") != sha256_file(source_receipt)
        or source_row.get("receipt_sha256") != curriculum["receipt_sha256"]
        or source_row.get("output_path") != str(source.resolve())
        or source_row.get("output_bytes") != source.stat().st_size
        or source_row.get("output_sha256") != sha256_file(source)
    ):
        raise CurriculumSplitError("curriculum split source differs")
    expected_policy = {
        "method": "document_identity_modulus_after_global_near_deduplication",
        "development_modulus": DEVELOPMENT_MODULUS,
        "development_bucket": DEVELOPMENT_BUCKET,
        "phase_order_preserved": True,
        "source_high_confidence_near_duplicate_filter_inherited": True,
        "near_duplicate_claim": "high_confidence_not_exhaustive",
    }
    if payload.get("policy") != expected_policy:
        raise CurriculumSplitError("curriculum split policy differs")
    outputs = {
        name: Path(payload.get(name, {}).get("path", ""))
        for name in ("train", "development")
    }
    for name, path in outputs.items():
        row = payload.get(name, {})
        if (
            not path.is_file()
            or path.is_symlink()
            or row.get("bytes") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise CurriculumSplitError("curriculum split output differs")
    handles = {name: path.open() for name, path in outputs.items()}
    replay_phases = {
        population: {phase: _empty_phase(index) for index, phase in enumerate(PHASES)}
        for population in outputs
    }
    replay_counts = Counter()
    replay_identities = {name: hashlib.sha256() for name in outputs}
    try:
        with source.open() as source_handle:
            for phase in PHASES:
                for _ in range(curriculum["phases"][phase]["documents"]):
                    source_value = normalize_document(
                        json.loads(source_handle.readline())
                    )
                    identity = source_value["identity_sha256"]
                    population = "development" if _is_development(identity) else "train"
                    candidate = handles[population].readline()
                    if (
                        not candidate
                        or normalize_document(json.loads(candidate)) != source_value
                    ):
                        raise CurriculumSplitError(
                            "curriculum split assignment differs"
                        )
                    signals = document_signals(source_value["text"])
                    phase_row = replay_phases[population][phase]
                    phase_row["documents"] += 1
                    phase_row["by_band"][signals["band"]] += 1
                    phase_row["difficulty_sum"] += signals["difficulty"]
                    identity_bytes = bytes.fromhex(identity)
                    phase_row["identity"].update(identity_bytes)
                    replay_identities[population].update(identity_bytes)
                    replay_counts[population] += 1
            if source_handle.read(1):
                raise CurriculumSplitError("curriculum split source replay differs")
        if any(handle.read(1) for handle in handles.values()):
            raise CurriculumSplitError("curriculum split output has extra rows")
    except (json.JSONDecodeError, RuntimeError) as error:
        raise CurriculumSplitError("curriculum split replay differs") from error
    finally:
        for handle in handles.values():
            handle.close()
    replayed_output_rows = {}
    for population in outputs:
        phases = {
            phase: _finish_phase(replay_phases[population][phase]) for phase in PHASES
        }
        declared = payload[population]
        progression = _progression(phases)
        if (
            declared.get("documents") != replay_counts[population]
            or declared.get("identity_sha256")
            != replay_identities[population].hexdigest()
            or declared.get("phases") != phases
            or declared.get("progression_checks") != progression
            or declared.get("curriculum_qualified") is not all(progression.values())
        ):
            raise CurriculumSplitError("curriculum split evidence differs")
        replayed_output_rows[population] = {
            "documents": replay_counts[population],
            "phases": phases,
            "progression": progression,
            "curriculum_qualified": all(progression.values()),
        }
    every_phase = all(
        replayed_output_rows[population]["phases"][phase]["documents"] > 0
        for population in outputs
        for phase in PHASES
    )
    expected_checks = {
        "all_curriculum_documents_emitted_once": sum(replay_counts.values())
        == curriculum["documents"]["accepted"],
        "exact_identity_assignment_disjoint": True,
        "both_populations_have_every_phase": every_phase,
        "train_progression_qualified": replayed_output_rows["train"][
            "curriculum_qualified"
        ],
    }
    expected_diagnostics = {
        "development_progression_nondecreasing": replayed_output_rows["development"][
            "progression"
        ]["phase_mean_difficulty_nondecreasing"],
        "development_curriculum_shape_matches_training_policy": replayed_output_rows[
            "development"
        ]["curriculum_qualified"],
    }
    if (
        payload.get("status") != "qualified"
        or payload.get("split_qualified") is not True
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("checks") != expected_checks
        or not all(expected_checks.values())
        or payload.get("diagnostics") != expected_diagnostics
    ):
        raise CurriculumSplitError("curriculum split qualification differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--curriculum-receipt", type=Path, required=True)
    build.add_argument("--train", type=Path, required=True)
    build.add_argument("--development", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--curriculum-workers", type=int, default=1)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--curriculum-workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_curriculum_split(
            args.curriculum_receipt,
            args.train,
            args.development,
            args.receipt,
            curriculum_workers=args.curriculum_workers,
        )
    else:
        payload = validate_curriculum_split(
            args.receipt, curriculum_workers=args.curriculum_workers
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
