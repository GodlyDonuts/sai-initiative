"""Select an immutable, phase-and-band-balanced semantic audit population."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.curriculum import BANDS, PHASES, document_signals, validate_curriculum
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-semantic-prerequisite-audit-population-v1"
ROW_SCHEMA = "sai-semantic-prerequisite-audit-document-v1"
SELECTION_SALT = b"sai-semantic-prerequisite-audit-population-v1"
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "curriculum",
    "selection",
    "output",
    "limitations",
    "receipt_sha256",
}


class PrerequisiteSampleError(RuntimeError):
    """The curriculum, selection policy, or audit population differs."""


def _select(
    curriculum: dict[str, Any], *, per_stratum: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if (
        isinstance(per_stratum, bool)
        or not isinstance(per_stratum, int)
        or per_stratum <= 0
    ):
        raise PrerequisiteSampleError("audit population geometry differs")
    source = Path(curriculum["output"]["path"])
    heaps: dict[tuple[str, str], list[tuple[int, int, dict[str, Any]]]] = {
        (phase, band): [] for phase in PHASES for band in BANDS
    }
    candidates: Counter[str] = Counter()
    document_index = 0
    try:
        with source.open() as handle:
            for phase in PHASES:
                declared = curriculum["phases"][phase]["documents"]
                for _ in range(declared):
                    line = handle.readline()
                    if not line:
                        raise PrerequisiteSampleError("curriculum ended early")
                    row = normalize_document(json.loads(line))
                    band = document_signals(row["text"])["band"]
                    identity = row["identity_sha256"]
                    rank = int.from_bytes(
                        hashlib.sha256(
                            SELECTION_SALT + bytes.fromhex(identity)
                        ).digest(),
                        "big",
                    )
                    sample = {
                        "schema": ROW_SCHEMA,
                        "document_index": document_index,
                        "phase": phase,
                        "surface_band": band,
                        "selection_rank_sha256": f"{rank:064x}",
                        "document_identity_sha256": identity,
                        "source": row["source"],
                        "text": row["text"],
                    }
                    key = (phase, band)
                    candidates[f"{phase}:{band}"] += 1
                    item = (-rank, -document_index, sample)
                    if len(heaps[key]) < per_stratum:
                        heapq.heappush(heaps[key], item)
                    elif item > heaps[key][0]:
                        heapq.heapreplace(heaps[key], item)
                    document_index += 1
            if handle.readline():
                raise PrerequisiteSampleError("curriculum has undeclared rows")
    except (json.JSONDecodeError, RuntimeError) as error:
        raise PrerequisiteSampleError("curriculum row differs") from error
    selected = []
    for phase in PHASES:
        for band in BANDS:
            values = heaps[(phase, band)]
            if len(values) != per_stratum:
                raise PrerequisiteSampleError("audit population stratum is incomplete")
            selected.extend(
                item[2]
                for item in sorted(
                    values,
                    key=lambda item: (
                        -item[0],
                        -item[1],
                    ),
                )
            )
    return selected, dict(sorted(candidates.items()))


def _encoded_rows(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def build_audit_population(
    curriculum_receipt: Path,
    output: Path,
    receipt: Path,
    *,
    per_stratum: int = 8,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Select and atomically publish an unreviewed semantic audit population."""

    if output.parent != receipt.parent:
        raise PrerequisiteSampleError("audit outputs must share one parent")
    if any(path.exists() or path.is_symlink() for path in (output, receipt)):
        raise PrerequisiteSampleError("audit output already exists")
    curriculum = validate_curriculum(curriculum_receipt, workers=curriculum_workers)
    rows, candidates = _select(curriculum, per_stratum=per_stratum)
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
            "curriculum": {
                "receipt_path": str(curriculum_receipt.resolve()),
                "receipt_file_sha256": sha256_file(curriculum_receipt),
                "receipt_sha256": curriculum["receipt_sha256"],
                "output_sha256": curriculum["output"]["sha256"],
            },
            "selection": {
                "method": "lowest_identity_hash_per_phase_and_surface_band",
                "salt_sha256": hashlib.sha256(SELECTION_SALT).hexdigest(),
                "phases": list(PHASES),
                "surface_bands": list(BANDS),
                "per_stratum": per_stratum,
                "strata": len(PHASES) * len(BANDS),
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


def validate_audit_population(
    receipt: Path, *, curriculum_workers: int = 1
) -> dict[str, Any]:
    """Recompute an exact semantic audit population from its curriculum."""

    if not receipt.is_file() or receipt.is_symlink() or receipt.stat().st_nlink != 1:
        raise PrerequisiteSampleError("audit receipt is missing or unsafe")
    try:
        payload = json.loads(receipt.read_text())
    except json.JSONDecodeError as error:
        raise PrerequisiteSampleError("audit receipt JSON differs") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _TOP_KEYS
        or payload.get("schema") != SCHEMA
        or payload.get("receipt_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "receipt_sha256"}
        )
    ):
        raise PrerequisiteSampleError("audit receipt differs")
    curriculum_row = payload.get("curriculum", {})
    curriculum_receipt = Path(curriculum_row.get("receipt_path", ""))
    curriculum = validate_curriculum(curriculum_receipt, workers=curriculum_workers)
    if (
        curriculum_row
        != {
            "receipt_path": str(curriculum_receipt.resolve()),
            "receipt_file_sha256": sha256_file(curriculum_receipt),
            "receipt_sha256": curriculum["receipt_sha256"],
            "output_sha256": curriculum["output"]["sha256"],
        }
        or payload.get("status") != "selected_unreviewed"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
    ):
        raise PrerequisiteSampleError("audit curriculum lineage differs")
    selection = payload.get("selection", {})
    per_stratum = selection.get("per_stratum")
    rows, candidates = _select(curriculum, per_stratum=per_stratum)
    encoded = _encoded_rows(rows)
    output = Path(payload.get("output", {}).get("path", ""))
    if (
        selection
        != {
            "method": "lowest_identity_hash_per_phase_and_surface_band",
            "salt_sha256": hashlib.sha256(SELECTION_SALT).hexdigest(),
            "phases": list(PHASES),
            "surface_bands": list(BANDS),
            "per_stratum": per_stratum,
            "strata": len(PHASES) * len(BANDS),
            "selected_documents": len(rows),
            "candidate_documents": candidates,
        }
        or not output.is_file()
        or output.is_symlink()
        or output.stat().st_nlink != 1
        or output.read_bytes() != encoded
        or payload.get("output")
        != {
            "path": str(output.resolve()),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_population_sha256": canonical_sha256(rows),
        }
        or payload.get("limitations")
        != [
            "population_is_selected_not_reviewed",
            "surface_band_is_not_semantic_difficulty",
            "receipt_authorizes_no_training",
        ]
    ):
        raise PrerequisiteSampleError("audit population replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--curriculum-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--per-stratum", type=int, default=8)
    build.add_argument("--curriculum-workers", type=int, default=1)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--curriculum-workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_audit_population(
            args.curriculum_receipt,
            args.output,
            args.receipt,
            per_stratum=args.per_stratum,
            curriculum_workers=args.curriculum_workers,
        )
    else:
        payload = validate_audit_population(
            args.receipt, curriculum_workers=args.curriculum_workers
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
