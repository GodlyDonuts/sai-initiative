"""Measure exact text-column yield in hash-selected reservoir members."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_yield_ledger import _load_receipt
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.reservoir_audit_population import _load_reservoir
from sai.data.token_stream import canonical_sha256, sha256_file

PLAN_SCHEMA = "sai-reservoir-text-payload-probe-plan-v1"
RECEIPT_SCHEMA = "sai-reservoir-text-payload-probe-receipt-v1"
SEED = "sai-text-payload-probe-20260825-r1"
MIN_USEFUL_BYTES = 200
MAX_USEFUL_BYTES = 128 * 1024
DEFAULT_MAX_PARENT_BYTES = 4 * 1024**3
SUPPORTED_SUFFIXES = (".parquet", ".jsonl.zst", ".json.gz")


class TextPayloadProbeError(RuntimeError):
    """A reservoir, probe plan, remote member, or measured payload differs."""


def _rank(source_id: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        "\0".join(
            (
                SEED,
                source_id,
                row["repository"],
                row["path"],
                row["sha256"],
            )
        ).encode()
    ).hexdigest()


def _member(row: dict[str, Any], *, frontier: bool) -> dict[str, Any]:
    size_key = "physical_bytes" if frontier else "bytes"
    size = row.get(size_key)
    if (
        not isinstance(row.get("source_id"), str)
        or not row["source_id"]
        or not isinstance(row.get("repository"), str)
        or not row["repository"]
        or not isinstance(row.get("revision"), str)
        or len(row["revision"]) != 40
        or not isinstance(row.get("path"), str)
        or not row["path"].endswith(SUPPORTED_SUFFIXES)
        or not isinstance(row.get("sha256"), str)
        or len(row["sha256"]) != 64
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise TextPayloadProbeError("probe reservoir member differs")
    return {
        "source_id": row["source_id"],
        "repository": row["repository"],
        "revision": row["revision"],
        "path": row["path"],
        "physical_bytes": size,
        "sha256": row["sha256"],
        "text_column": row.get("text_column", "text"),
    }


def _rights_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_receipt(path)
    rows = receipt.get("source_rows")
    if (
        receipt.get("schema") != "sai-reservoir-rights-inventory-v2"
        or not isinstance(rows, list)
        or not rows
        or receipt.get("training_ready") is not False
    ):
        raise TextPayloadProbeError("probe rights inventory differs")
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("source_id"), str)
            or row["source_id"] in by_source
            or not isinstance(row.get("rights_work_route"), str)
        ):
            raise TextPayloadProbeError("probe rights source differs")
        by_source[row["source_id"]] = row
    return by_source, receipt


def build_plan(
    original_manifest: Path,
    original_receipt: Path,
    frontier_manifest: Path,
    frontier_receipt: Path,
    rights_inventory: Path,
    output_path: Path,
    *,
    source_ids: list[str],
    samples_per_source: int = 1,
    maximum_parent_bytes: int = DEFAULT_MAX_PARENT_BYTES,
) -> dict[str, Any]:
    """Select members by hash rank before inspecting size or content."""

    if (
        output_path.exists()
        or output_path.is_symlink()
        or not source_ids
        or len(source_ids) != len(set(source_ids))
        or isinstance(samples_per_source, bool)
        or not isinstance(samples_per_source, int)
        or samples_per_source <= 0
        or isinstance(maximum_parent_bytes, bool)
        or not isinstance(maximum_parent_bytes, int)
        or maximum_parent_bytes <= 0
    ):
        raise TextPayloadProbeError("probe planning boundary differs")
    original = [
        _member(row, frontier=False)
        for row in _load_reservoir(original_manifest, original_receipt)
    ]
    frontier = [
        _member(row, frontier=True)
        for row in load_frontier_reservoir(frontier_manifest, frontier_receipt)
    ]
    members = original + frontier
    rights, rights_receipt = _rights_rows(rights_inventory)
    selections = []
    for source_id in source_ids:
        if source_id not in rights:
            raise TextPayloadProbeError(f"probe source lacks rights row: {source_id}")
        available = [row for row in members if row["source_id"] == source_id]
        ranked = sorted(
            available,
            key=lambda row: (_rank(source_id, row), row["repository"], row["path"]),
        )
        if len(ranked) < samples_per_source:
            raise TextPayloadProbeError(f"probe source is underfilled: {source_id}")
        for row in ranked[:samples_per_source]:
            selections.append(
                {
                    **row,
                    "selection_rank_sha256": _rank(source_id, row),
                    "rights_work_route": rights[source_id]["rights_work_route"],
                    "within_parent_byte_cap": (
                        row["physical_bytes"] <= maximum_parent_bytes
                    ),
                }
            )
    identities = [
        (row["repository"], row["revision"], row["path"])
        for row in selections
    ]
    if len(identities) != len(set(identities)):
        raise TextPayloadProbeError("probe selects one member more than once")
    payload = {
        "schema": PLAN_SCHEMA,
        "status": "complete_prospective_text_payload_probe",
        "seed": SEED,
        "method": {
            "selection_before_content_inspection": True,
            "selection_by_sha256_rank_not_file_size": True,
            "samples_per_source": samples_per_source,
            "maximum_parent_bytes": maximum_parent_bytes,
            "oversized_selected_member_replacement_allowed": False,
            "statistical_yield_estimate": False,
        },
        "inputs": {
            "original_manifest_sha256": sha256_file(original_manifest),
            "original_receipt_sha256": sha256_file(original_receipt),
            "frontier_manifest_sha256": sha256_file(frontier_manifest),
            "frontier_receipt_sha256": sha256_file(frontier_receipt),
            "rights_inventory_file_sha256": sha256_file(rights_inventory),
            "rights_inventory_receipt_sha256": rights_receipt["receipt_sha256"],
        },
        "source_ids": source_ids,
        "selections": selections,
        "selected_members": len(selections),
        "selected_physical_bytes": sum(row["physical_bytes"] for row in selections),
        "members_within_parent_byte_cap": sum(
            row["within_parent_byte_cap"] for row in selections
        ),
        "source_text_persisted": False,
        "content_downloaded": False,
        "source_admitted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def _text_measurement(values: Iterable[Any]) -> dict[str, Any]:
    total_rows = string_rows = empty_rows = 0
    short_rows = useful_rows = oversized_rows = 0
    text_bytes = useful_bytes = 0
    ordered = hashlib.sha256()
    for ordinal, value in enumerate(values):
        total_rows += 1
        if not isinstance(value, str):
            ordered.update(ordinal.to_bytes(8, "big"))
            ordered.update(b"non-string")
            continue
        string_rows += 1
        encoded = value.encode("utf-8")
        size = len(encoded)
        text_bytes += size
        ordered.update(ordinal.to_bytes(8, "big"))
        ordered.update(size.to_bytes(8, "big"))
        ordered.update(hashlib.sha256(encoded).digest())
        if size == 0:
            empty_rows += 1
        if size < MIN_USEFUL_BYTES:
            short_rows += 1
        elif size > MAX_USEFUL_BYTES:
            oversized_rows += 1
        else:
            useful_rows += 1
            useful_bytes += size
    if total_rows == 0:
        raise TextPayloadProbeError("probe member contains no rows")
    return {
        "rows": total_rows,
        "string_text_rows": string_rows,
        "non_string_text_rows": total_rows - string_rows,
        "empty_text_rows": empty_rows,
        "short_text_rows": short_rows,
        "useful_text_rows": useful_rows,
        "oversized_text_rows": oversized_rows,
        "text_utf8_bytes": text_bytes,
        "useful_text_utf8_bytes": useful_bytes,
        "minimum_useful_text_bytes": MIN_USEFUL_BYTES,
        "maximum_useful_text_bytes": MAX_USEFUL_BYTES,
        "ordered_text_identity_sha256": ordered.hexdigest(),
    }


def measure_local_member(path: Path, *, text_column: str) -> dict[str, Any]:
    """Measure one exact local parquet or JSONL-Zstandard member."""

    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or not isinstance(text_column, str)
        or not text_column
    ):
        raise TextPayloadProbeError("probe local member is missing or unsafe")
    if path.name.endswith(".parquet"):
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise TextPayloadProbeError(
                "pyarrow is required for parquet probes"
            ) from error
        source = parquet.ParquetFile(path)
        if text_column not in source.schema_arrow.names:
            raise TextPayloadProbeError("probe parquet text column differs")

        def parquet_values() -> Iterable[Any]:
            for group in range(source.metadata.num_row_groups):
                yield from source.read_row_group(
                    group, columns=[text_column], use_threads=False
                )[text_column].to_pylist()

        return _text_measurement(parquet_values())
    if path.name.endswith(".jsonl.zst"):
        try:
            import zstandard
        except ImportError as error:
            raise TextPayloadProbeError(
                "zstandard is required for JSONL probes"
            ) from error

        def zstd_values() -> Iterable[Any]:
            try:
                with path.open("rb") as compressed:
                    reader = zstandard.ZstdDecompressor().stream_reader(compressed)
                    with io.TextIOWrapper(reader, encoding="utf-8") as decoded:
                        for line_number, line in enumerate(decoded, start=1):
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise TextPayloadProbeError(
                                    f"probe JSONL row {line_number} differs"
                                )
                            yield row.get(text_column)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise TextPayloadProbeError("probe JSONL member differs") from error

        return _text_measurement(zstd_values())
    if path.name.endswith(".json.gz"):

        def gzip_values() -> Iterable[Any]:
            try:
                with gzip.open(path, "rt", encoding="utf-8") as decoded:
                    for line_number, line in enumerate(decoded, start=1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise TextPayloadProbeError(
                                f"probe JSON row {line_number} differs"
                            )
                        yield row.get(text_column)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise TextPayloadProbeError("probe gzip member differs") from error

        return _text_measurement(gzip_values())
    raise TextPayloadProbeError("probe member format is unsupported")


def _download_member(selection: dict[str, Any], token: str, root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise TextPayloadProbeError("huggingface_hub is required for probes") from error
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=selection["repository"],
                filename=selection["path"],
                repo_type="dataset",
                revision=selection["revision"],
                local_dir=root,
                force_download=True,
                token=token,
            )
        )
    except Exception as error:
        raise TextPayloadProbeError("probe member download failed") from error
    try:
        downloaded.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise TextPayloadProbeError("probe download escapes temporary root") from error
    if (
        not downloaded.is_file()
        or downloaded.is_symlink()
        or downloaded.stat().st_nlink != 1
    ):
        raise TextPayloadProbeError("probe downloaded member is unsafe")
    return downloaded


def _validate_plan(path: Path) -> dict[str, Any]:
    payload = _load_receipt(path)
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != "complete_prospective_text_payload_probe"
        or not isinstance(payload.get("selections"), list)
        or not payload["selections"]
        or payload.get("content_downloaded") is not False
        or payload.get("source_admitted") is not False
        or payload.get("training_ready") is not False
    ):
        raise TextPayloadProbeError("probe plan differs")
    return payload


def run_plan(plan_path: Path, output_path: Path, *, token: str) -> dict[str, Any]:
    """Measure selected members one at a time and discard temporary bytes."""

    if not token or output_path.exists() or output_path.is_symlink():
        raise TextPayloadProbeError("probe execution boundary differs")
    plan_file_sha256 = sha256_file(plan_path)
    plan = _validate_plan(plan_path)
    measurements = []
    for selection in plan["selections"]:
        if selection.get("within_parent_byte_cap") is not True:
            measurements.append(
                {
                    "source_id": selection["source_id"],
                    "repository": selection["repository"],
                    "revision": selection["revision"],
                    "path": selection["path"],
                    "selection_rank_sha256": selection["selection_rank_sha256"],
                    "status": "blocked_selected_member_exceeds_parent_byte_cap",
                    "physical_bytes": selection["physical_bytes"],
                    "source_text_persisted": False,
                }
            )
            continue
        with tempfile.TemporaryDirectory(prefix="sai-text-payload-probe-") as root:
            local = _download_member(selection, token, Path(root))
            if (
                local.stat().st_size != selection["physical_bytes"]
                or sha256_file(local) != selection["sha256"]
            ):
                raise TextPayloadProbeError("probe downloaded member identity differs")
            result = measure_local_member(local, text_column=selection["text_column"])
        measurements.append(
            {
                "source_id": selection["source_id"],
                "repository": selection["repository"],
                "revision": selection["revision"],
                "path": selection["path"],
                "selection_rank_sha256": selection["selection_rank_sha256"],
                "status": "measured_exact_member",
                "physical_bytes": selection["physical_bytes"],
                "physical_sha256": selection["sha256"],
                "full_member_size_and_sha256_replayed": True,
                "measurement": result,
                "text_to_physical_ppm": (
                    result["text_utf8_bytes"] * 1_000_000
                )
                // selection["physical_bytes"],
                "useful_text_to_physical_ppm": (
                    result["useful_text_utf8_bytes"] * 1_000_000
                )
                // selection["physical_bytes"],
                "source_text_persisted": False,
            }
        )
    measured = [row for row in measurements if row["status"] == "measured_exact_member"]
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_bounded_exact_member_measurement",
        "plan": {
            "path": str(plan_path.resolve()),
            "file_sha256": plan_file_sha256,
            "receipt_sha256": plan["receipt_sha256"],
        },
        "measurements": measurements,
        "summary": {
            "selected_members": len(measurements),
            "measured_members": len(measured),
            "blocked_members": len(measurements) - len(measured),
            "measured_physical_bytes": sum(row["physical_bytes"] for row in measured),
            "measured_text_utf8_bytes": sum(
                row["measurement"]["text_utf8_bytes"] for row in measured
            ),
            "measured_useful_text_utf8_bytes": sum(
                row["measurement"]["useful_text_utf8_bytes"] for row in measured
            ),
        },
        "temporary_members_removed": True,
        "sample_is_statistical_yield_estimate": False,
        "full_source_yield_extrapolation_allowed": False,
        "source_text_persisted": False,
        "source_admitted": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    if sha256_file(plan_path) != plan_file_sha256:
        raise TextPayloadProbeError("probe plan changed during execution")
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--original-manifest", type=Path, required=True)
    plan.add_argument("--original-receipt", type=Path, required=True)
    plan.add_argument("--frontier-manifest", type=Path, required=True)
    plan.add_argument("--frontier-receipt", type=Path, required=True)
    plan.add_argument("--rights-inventory", type=Path, required=True)
    plan.add_argument("--source-id", action="append", required=True)
    plan.add_argument("--samples-per-source", type=int, default=1)
    plan.add_argument(
        "--maximum-parent-bytes", type=int, default=DEFAULT_MAX_PARENT_BYTES
    )
    plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--token-env", default="HF_TOKEN")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(
            args.original_manifest,
            args.original_receipt,
            args.frontier_manifest,
            args.frontier_receipt,
            args.rights_inventory,
            args.output,
            source_ids=args.source_id,
            samples_per_source=args.samples_per_source,
            maximum_parent_bytes=args.maximum_parent_bytes,
        )
    else:
        result = run_plan(
            args.plan,
            args.output,
            token=os.environ.get(args.token_env, ""),
        )
    print(json.dumps({"receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
