"""Replay packed shards and seal an exact physical Sai 1B foundation window."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_foundation_pack import BANDS, SEQUENCE_LENGTH
from sai.data.one_b_foundation_pack import SCHEMA as PACK_SCHEMA
from sai.data.one_b_foundation_window_plan import SCHEMA as PLAN_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-1b-foundation-window-aggregate-v1"
COMPONENT_COUNTS = {"books": 64, "pleias": 128, "code": 128, "connections": 1}


class OneBFoundationAggregateError(RuntimeError):
    """A pack shard, part, exact trim, or aggregate identity differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBFoundationAggregateError("signed aggregate input differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBFoundationAggregateError("signed aggregate input differs")
    return value


def _receipt_paths(root: Path, component: str) -> list[Path]:
    if component == "books":
        return [
            root / component / f"shard_{index:05d}" / "receipt.json"
            for index in range(64)
        ]
    if component in {"pleias", "code"}:
        return [
            root / component / f"bucket_{index:03d}" / "receipt.json"
            for index in range(128)
        ]
    return [root / component / "receipt.json"]


def _parts(
    pack_root: Path, plan: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_band = {band: [] for band in BANDS}
    connection_parts: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    receipt_hashes: list[str] = []
    tokenizer_identity: str | None = None
    for component, expected_receipts in COMPONENT_COUNTS.items():
        receipt_paths = _receipt_paths(pack_root, component)
        if len(receipt_paths) != expected_receipts:
            raise OneBFoundationAggregateError("pack receipt namespace differs")
        for receipt_path in receipt_paths:
            receipt = _load_signed(receipt_path, PACK_SCHEMA)
            if (
                receipt.get("component") != component
                or receipt.get("plan_receipt_sha256") != plan["receipt_sha256"]
                or receipt.get("development_rows_excluded") is not True
                or receipt.get("model_training_started") is not False
            ):
                raise OneBFoundationAggregateError("pack receipt differs")
            identity = receipt.get("tokenizer_identity_sha256")
            if tokenizer_identity is None:
                tokenizer_identity = identity
            elif identity != tokenizer_identity:
                raise OneBFoundationAggregateError("pack tokenizer identity differs")
            receipt_hashes.append(receipt["receipt_sha256"])
            counts["receipts"] += 1
            counts["documents"] += receipt.get("counts", {}).get("documents", 0)
            counts["retained_tokens"] += receipt.get("retained_tokens", 0)
            shard_root = receipt_path.parent
            for band in BANDS:
                output = receipt.get("band_outputs", {}).get(band, {})
                for descriptor in output.get("parts", []):
                    path = shard_root / descriptor.get("path", "")
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_nlink != 1
                        or path.stat().st_size != descriptor.get("bytes")
                        or descriptor.get("tokens")
                        != descriptor.get("sequences") * SEQUENCE_LENGTH
                        or descriptor.get("bytes") != descriptor.get("tokens") * 2
                        or sha256_file(path) != descriptor.get("sha256")
                    ):
                        raise OneBFoundationAggregateError("packed part differs")
                    part = {
                        "component": component,
                        "band": band,
                        "path": str(path.resolve()),
                        "sequences": descriptor["sequences"],
                        "tokens": descriptor["tokens"],
                        "bytes": descriptor["bytes"],
                        "sha256": descriptor["sha256"],
                    }
                    if component == "connections":
                        connection_parts.append(part)
                    else:
                        by_band[band].append(part)
    if tokenizer_identity is None:
        raise OneBFoundationAggregateError("pack population is empty")
    metadata = {
        "counts": dict(sorted(counts.items())),
        "pack_receipts_sha256": canonical_sha256(receipt_hashes),
        "tokenizer_identity_sha256": tokenizer_identity,
        "connection_parts": connection_parts,
    }
    return by_band, metadata


def _priority(part: dict[str, Any]) -> str:
    return canonical_sha256(
        {key: part[key] for key in ("component", "band", "path", "sha256")}
    )


def _copy_prefix(
    source: Path,
    destination: Path,
    sequences: int,
    *,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    remaining = sequences * SEQUENCE_LENGTH * 2
    temporary = destination.with_name(f".{destination.name}.partial.{uuid.uuid4().hex}")
    with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
        while remaining:
            block = input_handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise OneBFoundationAggregateError("trim source ended early")
            output_handle.write(block)
            remaining -= len(block)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, destination)
    return {
        "component": "exact_trim",
        "path": str((receipt_path or destination).resolve()),
        "sequences": sequences,
        "tokens": sequences * SEQUENCE_LENGTH,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source_part_sha256": sha256_file(source),
    }


def _select_exact(
    parts: list[dict[str, Any]],
    target: int,
    stage: Path,
    output_root: Path,
    band: str,
) -> list[dict[str, Any]]:
    ordered = sorted(parts, key=lambda value: (_priority(value), value["path"]))
    selected: list[dict[str, Any]] = []
    remaining = target
    prefix_source: dict[str, Any] | None = None
    for part in ordered:
        if part["sequences"] <= remaining:
            selected.append(part)
            remaining -= part["sequences"]
        elif prefix_source is None:
            prefix_source = part
        if not remaining:
            break
    if remaining:
        if prefix_source is None or prefix_source["sequences"] < remaining:
            raise OneBFoundationAggregateError(f"insufficient packed {band} sequences")
        tail = _copy_prefix(
            Path(prefix_source["path"]),
            stage / f"exact-{band}-tail.bin",
            remaining,
            receipt_path=output_root / f"exact-{band}-tail.bin",
        )
        tail["band"] = band
        selected.append(tail)
    if sum(part["sequences"] for part in selected) != target:
        raise OneBFoundationAggregateError("exact band trim differs")
    return selected


def build(pack_root: Path, plan_path: Path, output_root: Path) -> dict[str, Any]:
    """Replay all expected packs and materialize only exact boundary prefixes."""

    if output_root.exists() or output_root.is_symlink():
        raise OneBFoundationAggregateError("foundation aggregate output exists")
    plan = _load_signed(plan_path, PLAN_SCHEMA)
    by_band, metadata = _parts(pack_root, plan)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        bands = {}
        all_parts = []
        for band in BANDS:
            selected = _select_exact(
                by_band[band],
                plan["bands"][band]["target_sequences"],
                stage,
                output_root,
                band,
            )
            bands[band] = {
                "target_sequences": plan["bands"][band]["target_sequences"],
                "target_tokens": plan["bands"][band]["target_tokens"],
                "parts": selected,
                "parts_sha256": canonical_sha256(selected),
            }
            all_parts.extend(selected)
        connections = metadata.pop("connection_parts")
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_exact_1b_foundation_window",
            "plan_receipt_sha256": plan["receipt_sha256"],
            **metadata,
            "bands": bands,
            "all_foundation_parts_sha256": canonical_sha256(all_parts),
            "window_sequences": sum(row["target_sequences"] for row in bands.values()),
            "window_tokens": sum(row["target_tokens"] for row in bands.values()),
            "connections": {
                "parts": connections,
                "parts_sha256": canonical_sha256(connections),
                "physical_sequences": sum(row["sequences"] for row in connections),
                "maximum_document_exposures": plan[
                    "maximum_connection_document_exposures"
                ],
                "development_rows_excluded": True,
            },
            "sequence_length": SEQUENCE_LENGTH,
            "memmap_dtype": "uint16_little_endian",
            "exact_boundary_prefixes_materialized_only": True,
            "model_training_started": False,
            "one_b_training_authorized": False,
        }
        if payload["window_sequences"] != plan["window_sequences"]:
            raise OneBFoundationAggregateError("foundation window total differs")
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.pack_root, args.plan, args.output_root)
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
