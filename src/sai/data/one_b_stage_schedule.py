"""Seal Sai's exact 4T virtual path schedule over a physical packed window."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_foundation_aggregate import SCHEMA as WINDOW_SCHEMA
from sai.data.one_b_foundation_pack import BANDS, SEQUENCE_LENGTH
from sai.data.one_b_spiral_contract import build_contract as spiral_contract
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-1b-4t-stage-path-schedule-v1"
CONNECTION_EXPOSURES_BY_STAGE = (2, 3, 4, 5, 2)


class OneBStageScheduleError(RuntimeError):
    """The physical window, exact stage count, or virtual path differs."""


def _load_window(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBStageScheduleError("foundation window differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != WINDOW_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBStageScheduleError("foundation window differs")
    return value


def _verify_part(part: dict[str, Any]) -> None:
    path = Path(part.get("path", ""))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != part.get("bytes")
        or part.get("tokens") != part.get("sequences") * SEQUENCE_LENGTH
        or part.get("bytes") != part.get("tokens") * 2
        or sha256_file(path) != part.get("sha256")
    ):
        raise OneBStageScheduleError("physical schedule part differs")


def _copy_prefix(
    source: dict[str, Any], output: Path, sequences: int
) -> dict[str, Any]:
    remaining = sequences * SEQUENCE_LENGTH * 2
    temporary = output.with_name(f".{output.name}.partial.{uuid.uuid4().hex}")
    with (
        Path(source["path"]).open("rb") as input_handle,
        temporary.open("xb") as output_handle,
    ):
        while remaining:
            block = input_handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise OneBStageScheduleError("stage prefix source ended early")
            output_handle.write(block)
            remaining -= len(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, output)
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "sequences_per_repeat": sequences,
        "tokens_per_repeat": sequences * SEQUENCE_LENGTH,
        "repeat": 1,
        "source": "exact_stage_prefix",
        "source_part_sha256": source["sha256"],
    }


def _cycle(
    parts: list[dict[str, Any]], target: int, stage_root: Path, label: str
) -> list[dict[str, Any]]:
    if target < 0 or not parts:
        raise OneBStageScheduleError("stage cycle target differs")
    for part in parts:
        _verify_part(part)
    physical = sum(part["sequences"] for part in parts)
    full_repeats, remainder = divmod(target, physical)
    entries = []
    if full_repeats:
        entries.extend(
            {
                "path": part["path"],
                "sha256": part["sha256"],
                "sequences_per_repeat": part["sequences"],
                "tokens_per_repeat": part["tokens"],
                "repeat": full_repeats,
                "source": "physical_window",
            }
            for part in parts
        )
    prefix_source: dict[str, Any] | None = None
    for part in parts:
        if part["sequences"] <= remainder:
            entries.append(
                {
                    "path": part["path"],
                    "sha256": part["sha256"],
                    "sequences_per_repeat": part["sequences"],
                    "tokens_per_repeat": part["tokens"],
                    "repeat": 1,
                    "source": "physical_window_remainder",
                }
            )
            remainder -= part["sequences"]
        elif prefix_source is None:
            prefix_source = part
        if not remainder:
            break
    if remainder:
        if prefix_source is None or prefix_source["sequences"] < remainder:
            raise OneBStageScheduleError("stage cycle prefix differs")
        entries.append(
            _copy_prefix(
                prefix_source, stage_root / f"{label}-exact-tail.bin", remainder
            )
        )
    if sum(row["sequences_per_repeat"] * row["repeat"] for row in entries) != target:
        raise OneBStageScheduleError("stage cycle accounting differs")
    return entries


def build(window_path: Path, output_root: Path) -> dict[str, Any]:
    """Build five exact stage datasets without duplicating physical full parts."""

    if output_root.exists() or output_root.is_symlink():
        raise OneBStageScheduleError("stage schedule output exists")
    window = _load_window(window_path)
    spiral = spiral_contract()
    stage_root = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage_root.mkdir(parents=True)
    try:
        connection_by_band = {band: [] for band in BANDS}
        for part in window["connections"]["parts"]:
            _verify_part(part)
            connection_by_band[part["band"]].append(part)
        stages = []
        total_connections = 0
        for stage, connection_exposures in zip(
            spiral["stages"], CONNECTION_EXPOSURES_BY_STAGE, strict=True
        ):
            body_entries = []
            bands = {}
            boundary_sequences = stage["sequences"] % 512
            boundary_entry = _copy_prefix(
                window["bands"]["foundation"]["parts"][0],
                stage_root / f"{stage['index']}-exact-boundary.bin",
                boundary_sequences,
            )
            boundary_entry["source"] = "exact_boundary_batch"
            boundary_entry["band"] = "foundation"
            for band in BANDS:
                target = stage["band_sequences"][band]
                connections = connection_by_band[band]
                connection_sequences = (
                    sum(part["sequences"] for part in connections)
                    * connection_exposures
                )
                if connection_sequences > target:
                    raise OneBStageScheduleError("connection overlay exceeds band")
                reserved_boundary = boundary_sequences if band == "foundation" else 0
                band_entries = _cycle(
                    window["bands"][band]["parts"],
                    target - connection_sequences - reserved_boundary,
                    stage_root,
                    f"{stage['index']}-{band}",
                )
                for part in connections:
                    band_entries.append(
                        {
                            "path": part["path"],
                            "sha256": part["sha256"],
                            "sequences_per_repeat": part["sequences"],
                            "tokens_per_repeat": part["tokens"],
                            "repeat": connection_exposures,
                            "source": "verified_train_only_connections",
                        }
                    )
                for entry in band_entries:
                    entry["band"] = band
                body_entries.extend(band_entries)
                bands[band] = {
                    "target_sequences": target,
                    "connection_sequences": connection_sequences,
                    "boundary_sequences": reserved_boundary,
                    "entries_sha256": canonical_sha256(band_entries),
                }
                total_connections += connection_sequences
            body_sequences = sum(
                row["sequences_per_repeat"] * row["repeat"] for row in body_entries
            )
            stage_sequences = body_sequences + boundary_sequences
            if stage_sequences != stage["sequences"] or body_sequences % 512:
                raise OneBStageScheduleError("exact stage sequence count differs")
            stages.append(
                {
                    "index": stage["index"],
                    "stage": stage["stage"],
                    "sequences": stage_sequences,
                    "tokens": stage_sequences * SEQUENCE_LENGTH,
                    "ordinary_batch_sequences": 512,
                    "ordinary_steps": body_sequences // 512,
                    "body_sequences": body_sequences,
                    "boundary_batch_sequences": boundary_sequences,
                    "connection_document_exposures": connection_exposures,
                    "bands": bands,
                    "body_entries": body_entries,
                    "body_entries_sha256": canonical_sha256(body_entries),
                    "boundary_entries": [boundary_entry],
                    "boundary_entries_sha256": canonical_sha256([boundary_entry]),
                }
            )
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_exact_1b_4t_path_schedule",
            "foundation_window_receipt_sha256": window["receipt_sha256"],
            "spiral_contract_receipt_sha256": spiral["receipt_sha256"],
            "tokenizer_identity_sha256": window["tokenizer_identity_sha256"],
            "stages": stages,
            "total_sequences": sum(stage["sequences"] for stage in stages),
            "total_tokens": sum(stage["tokens"] for stage in stages),
            "connection_document_exposures_total": sum(CONNECTION_EXPOSURES_BY_STAGE),
            "connection_sequence_exposures_total": total_connections,
            "development_rows_excluded": True,
            "full_parts_referenced_not_copied": True,
            "model_training_started": False,
            "one_b_training_authorized": False,
        }
        if (
            payload["total_sequences"] != spiral["target_sequences"]
            or payload["total_tokens"] != spiral["target_tokens"]
            or payload["connection_document_exposures_total"]
            != spiral["maximum_connection_document_exposures"]
        ):
            raise OneBStageScheduleError("4T schedule total differs")
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage_root / "receipt.json", payload)
        os.replace(stage_root, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.window, args.output_root)
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
