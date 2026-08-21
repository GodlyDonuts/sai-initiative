"""Deterministically shard and merge full development-MC evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sai.evaluation.development_mc import (
    DISJOINT_RECEIPT_SCHEMA,
)
from sai.evaluation.development_mc import (
    SCHEMA as RESULT_SCHEMA,
)

MANIFEST_SCHEMA = "sai-development-mc-shard-manifest-v1"
MERGE_SCHEMA = "sai-development-mc-shard-merge-v1"
BENCHMARKS = ("mmlu_pro", "musr")
RESULT_SHARED_BINDINGS = (
    "training_source_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "tokenizer_sha256",
    "evaluator_code_sha256",
    "runtime_files_sha256",
    "runtime_sha256",
    "decoding_contract_sha256",
    "scoring_contract_sha256",
)


class DevelopmentMCShardError(RuntimeError):
    """A shard population, result, or merge differs from the contract."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise DevelopmentMCShardError("artifact is missing or unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DevelopmentMCShardError(f"{field} differs")
    return value


def _load_json(path: Path, field: str) -> dict[str, Any]:
    _sha256_file(path)
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentMCShardError(f"{field} is unreadable") from error
    if not isinstance(value, dict):
        raise DevelopmentMCShardError(f"{field} differs")
    return value


def _load_rows(path: Path, benchmark: str) -> tuple[list[dict[str, Any]], str]:
    source_sha256 = _sha256_file(path)
    rows = []
    try:
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict) or row.get("benchmark") != benchmark:
                    raise DevelopmentMCShardError("benchmark row differs")
                row_id = row.get("row_id")
                if not isinstance(row_id, str) or not row_id:
                    raise DevelopmentMCShardError("benchmark row identity differs")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentMCShardError("benchmark source is unreadable") from error
    if not rows or len({row["row_id"] for row in rows}) != len(rows):
        raise DevelopmentMCShardError("benchmark row coverage differs")
    return rows, source_sha256


def _validate_full_receipt(
    path: Path,
    *,
    benchmark: str,
    source_sha256: str,
    training_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt_sha256 = _sha256_file(path)
    receipt = _load_json(path, "full disjoint receipt")
    if (
        receipt.get("schema") != DISJOINT_RECEIPT_SCHEMA
        or receipt.get("benchmark") != benchmark
        or receipt.get("benchmark_source_sha256") != source_sha256
        or receipt.get("training_source_sha256") != training_source_sha256
        or receipt.get("source_disjoint") is not True
        or receipt.get("method") != "identity-and-contamination-audit"
    ):
        raise DevelopmentMCShardError("full disjoint receipt differs")
    _sha256(receipt.get("evidence_sha256"), "full disjoint evidence")
    return receipt, receipt_sha256


def _write_new(path: Path, encoded: bytes) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise DevelopmentMCShardError("output path differs")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_shards(
    *,
    benchmark: str,
    source_path: Path,
    full_disjoint_receipt_path: Path,
    training_source_sha256: str,
    shard_count: int,
    output_root: Path,
) -> dict[str, Any]:
    """Create exact contiguous populations and derived disjoint receipts."""

    if benchmark not in BENCHMARKS:
        raise DevelopmentMCShardError("benchmark differs")
    training_source_sha256 = _sha256(training_source_sha256, "training source SHA256")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 2
    ):
        raise DevelopmentMCShardError("shard count differs")
    output_root = Path(output_root)
    if (
        output_root.exists()
        or output_root.is_symlink()
        or not output_root.parent.is_dir()
    ):
        raise DevelopmentMCShardError("shard output root differs")
    rows, source_sha256 = _load_rows(source_path, benchmark)
    if shard_count > len(rows):
        raise DevelopmentMCShardError("shard count exceeds row coverage")
    full_receipt, full_receipt_sha256 = _validate_full_receipt(
        full_disjoint_receipt_path,
        benchmark=benchmark,
        source_sha256=source_sha256,
        training_source_sha256=training_source_sha256,
    )
    output_root.mkdir(mode=0o700)
    created: list[Path] = []
    shards = []
    try:
        for index in range(shard_count):
            start = len(rows) * index // shard_count
            end = len(rows) * (index + 1) // shard_count
            shard_rows = rows[start:end]
            source = output_root / f"shard_{index:02d}.jsonl"
            encoded = b"".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
                for row in shard_rows
            )
            _write_new(source, encoded)
            created.append(source)
            shard_source_sha256 = hashlib.sha256(encoded).hexdigest()
            identities = [row["row_id"] for row in shard_rows]
            identity_order_sha256 = _canonical_sha256(identities)
            receipt = {
                "schema": DISJOINT_RECEIPT_SCHEMA,
                "benchmark": benchmark,
                "benchmark_source_sha256": shard_source_sha256,
                "training_source_sha256": training_source_sha256,
                "source_disjoint": True,
                "method": "identity-and-contamination-audit",
                "evidence_sha256": _canonical_sha256(
                    {
                        "full_disjoint_evidence_sha256": full_receipt[
                            "evidence_sha256"
                        ],
                        "full_disjoint_receipt_sha256": full_receipt_sha256,
                        "full_source_sha256": source_sha256,
                        "identity_order_sha256": identity_order_sha256,
                        "range": [start, end],
                    }
                ),
            }
            receipt_path = output_root / f"shard_{index:02d}.disjoint.json"
            receipt_encoded = (
                json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
            )
            _write_new(receipt_path, receipt_encoded)
            created.append(receipt_path)
            shards.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "rows": len(shard_rows),
                    "source": str(source.resolve()),
                    "source_bytes": len(encoded),
                    "source_sha256": shard_source_sha256,
                    "disjoint_receipt": str(receipt_path.resolve()),
                    "disjoint_receipt_sha256": hashlib.sha256(
                        receipt_encoded
                    ).hexdigest(),
                    "identity_order_sha256": identity_order_sha256,
                }
            )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "benchmark": benchmark,
            "shard_count": shard_count,
            "total_rows": len(rows),
            "full_source": str(Path(source_path).resolve()),
            "full_source_sha256": source_sha256,
            "full_disjoint_receipt": str(Path(full_disjoint_receipt_path).resolve()),
            "full_disjoint_receipt_sha256": full_receipt_sha256,
            "training_source_sha256": training_source_sha256,
            "full_identity_order_sha256": _canonical_sha256(
                [row["row_id"] for row in rows]
            ),
            "shards": shards,
        }
        manifest["receipt_sha256"] = _canonical_sha256(manifest)
        manifest_path = output_root / "manifest.json"
        encoded = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        _write_new(manifest_path, encoded)
        return manifest
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise


def _validate_manifest(path: Path) -> tuple[dict[str, Any], str]:
    file_sha256 = _sha256_file(path)
    manifest = _load_json(path, "shard manifest")
    unsigned = dict(manifest)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    shards = manifest.get("shards")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or receipt_sha256 != _canonical_sha256(unsigned)
        or manifest.get("benchmark") not in BENCHMARKS
        or not isinstance(shards, list)
        or len(shards) != manifest.get("shard_count")
        or not shards
    ):
        raise DevelopmentMCShardError("shard manifest differs")
    position = 0
    identities = []
    root = Path(path).resolve().parent
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or any(
            isinstance(shard.get(field), bool) or not isinstance(shard.get(field), int)
            for field in ("index", "start", "end", "rows", "source_bytes")
        ):
            raise DevelopmentMCShardError("shard geometry differs")
        if (
            shard.get("index") != index
            or shard.get("start") != position
            or shard.get("end") - shard.get("start") != shard.get("rows")
            or shard.get("rows") <= 0
        ):
            raise DevelopmentMCShardError("shard geometry differs")
        source = Path(shard["source"])
        receipt = Path(shard["disjoint_receipt"])
        if (
            source.resolve().parent != root
            or receipt.resolve().parent != root
            or source.name != f"shard_{index:02d}.jsonl"
            or receipt.name != f"shard_{index:02d}.disjoint.json"
        ):
            raise DevelopmentMCShardError("shard path differs")
        rows, observed_source_sha256 = _load_rows(source, manifest["benchmark"])
        if (
            len(rows) != shard["rows"]
            or source.stat().st_size != shard["source_bytes"]
            or observed_source_sha256 != shard["source_sha256"]
            or _sha256_file(receipt) != shard["disjoint_receipt_sha256"]
            or _canonical_sha256([row["row_id"] for row in rows])
            != shard["identity_order_sha256"]
        ):
            raise DevelopmentMCShardError("shard artifact differs")
        _validate_full_receipt(
            receipt,
            benchmark=manifest["benchmark"],
            source_sha256=shard["source_sha256"],
            training_source_sha256=manifest["training_source_sha256"],
        )
        identities.extend(row["row_id"] for row in rows)
        position = shard["end"]
    if (
        position != manifest.get("total_rows")
        or _canonical_sha256(identities) != manifest.get("full_identity_order_sha256")
        or _sha256_file(Path(manifest["full_source"]))
        != manifest.get("full_source_sha256")
        or _sha256_file(Path(manifest["full_disjoint_receipt"]))
        != manifest.get("full_disjoint_receipt_sha256")
    ):
        raise DevelopmentMCShardError("full population binding differs")
    _validate_full_receipt(
        Path(manifest["full_disjoint_receipt"]),
        benchmark=manifest["benchmark"],
        source_sha256=manifest["full_source_sha256"],
        training_source_sha256=manifest["training_source_sha256"],
    )
    return manifest, file_sha256


def _load_result(path: Path, benchmark: str) -> tuple[dict[str, Any], str]:
    file_sha256 = _sha256_file(path)
    result = _load_json(path, "shard result")
    unsigned = dict(result)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("benchmark") != benchmark
        or result.get("development_only") is not True
        or result.get("official_benchmark_result") is not False
        or result.get("public_terminal_result") is not False
        or result.get("architecture_promotion_allowed") is not False
        or receipt_sha256 != _canonical_sha256(unsigned)
    ):
        raise DevelopmentMCShardError("shard result receipt differs")
    return result, file_sha256


def merge_shards(
    manifest_path: Path,
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    """Merge exact ordered shard results into the standard full-row schema."""

    manifest, manifest_file_sha256 = _validate_manifest(manifest_path)
    shards = manifest["shards"]
    if len(result_paths) != len(shards):
        raise DevelopmentMCShardError("shard result count differs")
    loaded = []
    result_files = []
    for shard, path in zip(shards, result_paths, strict=True):
        result, file_sha256 = _load_result(path, manifest["benchmark"])
        bindings = result.get("bindings")
        rows = result.get("rows")
        coverage = result.get("coverage")
        if (
            not isinstance(bindings, dict)
            or bindings.get("benchmark_source_sha256") != shard["source_sha256"]
            or bindings.get("source_disjoint_receipt_sha256")
            != shard["disjoint_receipt_sha256"]
            or bindings.get("identity_order_sha256") != shard["identity_order_sha256"]
            or not isinstance(rows, list)
            or len(rows) != shard["rows"]
            or coverage
            != {"expected_rows": shard["rows"], "scored_rows": shard["rows"]}
        ):
            raise DevelopmentMCShardError("shard result binding differs")
        if _canonical_sha256(result.get("decoding_contract")) != bindings.get(
            "decoding_contract_sha256"
        ) or _canonical_sha256(result.get("scoring_contract")) != bindings.get(
            "scoring_contract_sha256"
        ):
            raise DevelopmentMCShardError("shard scoring contract differs")
        loaded.append(result)
        result_files.append(
            {
                "index": shard["index"],
                "path": str(Path(path).resolve()),
                "sha256": file_sha256,
                "receipt_sha256": result["receipt_sha256"],
            }
        )
    reference_bindings = loaded[0]["bindings"]
    if any(
        any(
            result["bindings"].get(field) != reference_bindings.get(field)
            for field in RESULT_SHARED_BINDINGS
        )
        for result in loaded[1:]
    ):
        raise DevelopmentMCShardError("shard execution binding differs")
    rows = [row for result in loaded for row in result["rows"]]
    identities = [row.get("row_id") for row in rows]
    if (
        len(rows) != manifest["total_rows"]
        or len(set(identities)) != len(rows)
        or _canonical_sha256(identities) != manifest["full_identity_order_sha256"]
        or any(not isinstance(row.get("correct"), bool) for row in rows)
    ):
        raise DevelopmentMCShardError("merged row coverage differs")
    domains: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        domains[row["domain"]].append(row["correct"])
    correct = sum(row["correct"] for row in rows)
    bindings = dict(reference_bindings)
    bindings.update(
        {
            "benchmark_source_sha256": manifest["full_source_sha256"],
            "source_disjoint_receipt_sha256": manifest["full_disjoint_receipt_sha256"],
            "identity_order_sha256": manifest["full_identity_order_sha256"],
        }
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "benchmark": manifest["benchmark"],
        "development_only": True,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "architecture_promotion_allowed": False,
        "bindings": bindings,
        "decoding_contract": loaded[0]["decoding_contract"],
        "scoring_contract": loaded[0]["scoring_contract"],
        "coverage": {"expected_rows": len(rows), "scored_rows": len(rows)},
        "aggregate": {
            "correct": correct,
            "rows": len(rows),
            "accuracy": correct / len(rows),
        },
        "domains": {
            domain: {
                "correct": sum(values),
                "rows": len(values),
                "accuracy": sum(values) / len(values),
            }
            for domain, values in sorted(domains.items())
        },
        "rows": rows,
        "shard_merge": {
            "schema": MERGE_SCHEMA,
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_receipt_sha256": manifest["receipt_sha256"],
            "ordered_results": result_files,
        },
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--full-disjoint-receipt", type=Path, required=True)
    build.add_argument("--training-source-sha256", required=True)
    build.add_argument("--shard-count", type=int, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--result", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_shards(
            benchmark=args.benchmark,
            source_path=args.source,
            full_disjoint_receipt_path=args.full_disjoint_receipt,
            training_source_sha256=args.training_source_sha256,
            shard_count=args.shard_count,
            output_root=args.output_root,
        )
    else:
        from sai.evaluation.development_mc import write_development_mc

        write_development_mc(args.output, merge_shards(args.manifest, args.result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
