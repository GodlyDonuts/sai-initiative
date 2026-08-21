"""Refresh exact development populations against a new decontaminated source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256
from sai.evaluation.population_builder import convert

SCHEMA = "sai-development-mc-populations-refresh-v1"


class PopulationRefreshError(RuntimeError):
    """The refreshed two-board population or its custody differs."""


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise PopulationRefreshError(f"input is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "bytes": Path(path).stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
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


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    """Reconvert both fixed boards and atomically bind their new source receipt."""

    output_root = Path(args.output_root)
    if (
        not isinstance(args.source_commit, str)
        or len(args.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in args.source_commit)
    ):
        raise PopulationRefreshError("source commit differs")
    if (
        output_root.exists()
        or output_root.is_symlink()
        or not output_root.parent.is_dir()
        or output_root.parent.is_symlink()
    ):
        raise PopulationRefreshError("population output boundary differs")
    decontamination_sha256 = _sha256_file(args.decontamination_receipt)
    output_root.mkdir(mode=0o700)
    try:
        population_specs = (
            (
                "mmlu_pro",
                args.mmlu_questions,
                args.mmlu_assessors,
                args.mmlu_questions_sha256,
                args.mmlu_assessors_sha256,
                12_032,
                args.mmlu_identity_order_sha256,
            ),
            (
                "musr",
                args.musr_questions,
                args.musr_assessors,
                args.musr_questions_sha256,
                args.musr_assessors_sha256,
                756,
                args.musr_identity_order_sha256,
            ),
        )
        populations = []
        for (
            benchmark,
            questions,
            assessors,
            questions_sha256,
            assessors_sha256,
            rows,
            identity_order_sha256,
        ) in population_specs:
            source = output_root / f"{benchmark}.jsonl"
            disjoint = output_root / f"{benchmark}.source_disjoint.receipt.json"
            conversion = output_root / f"{benchmark}.conversion.receipt.json"
            receipt = convert(
                benchmark=benchmark,
                questions_path=questions,
                assessors_path=assessors,
                expected_questions_sha256=questions_sha256,
                expected_assessors_sha256=assessors_sha256,
                expected_rows=rows,
                expected_identity_order_sha256=identity_order_sha256,
                training_decontamination_receipt_path=args.decontamination_receipt,
                expected_training_decontamination_receipt_sha256=(
                    decontamination_sha256
                ),
                output_source_path=source,
                output_disjoint_receipt_path=disjoint,
                output_conversion_receipt_path=conversion,
            )
            populations.append(
                {
                    "benchmark": benchmark,
                    "rows": rows,
                    "identity_order_sha256": identity_order_sha256,
                    "source": _artifact(source),
                    "source_disjoint_receipt": _artifact(disjoint),
                    "conversion_receipt": {
                        **_artifact(conversion),
                        "receipt_sha256": receipt["receipt_sha256"],
                    },
                }
            )
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "source_commit": args.source_commit,
            "training_decontamination_receipt": _artifact(args.decontamination_receipt),
            "populations": populations,
            "total_rows": sum(item["rows"] for item in populations),
            "development_only": True,
            "official_benchmark_result": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _write_new(output_root / "populations.aggregate.receipt.json", payload)
        for path in output_root.iterdir():
            if not path.is_file() or path.is_symlink():
                raise PopulationRefreshError("population output geometry differs")
            os.chmod(path, 0o444)
        os.chmod(output_root, 0o555)
        return payload
    except BaseException:
        if output_root.exists():
            os.chmod(output_root, stat.S_IRWXU)
            shutil.rmtree(output_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--decontamination-receipt", type=Path, required=True)
    parser.add_argument("--mmlu-questions", type=Path, required=True)
    parser.add_argument("--mmlu-assessors", type=Path, required=True)
    parser.add_argument("--mmlu-questions-sha256", required=True)
    parser.add_argument("--mmlu-assessors-sha256", required=True)
    parser.add_argument("--mmlu-identity-order-sha256", required=True)
    parser.add_argument("--musr-questions", type=Path, required=True)
    parser.add_argument("--musr-assessors", type=Path, required=True)
    parser.add_argument("--musr-questions-sha256", required=True)
    parser.add_argument("--musr-assessors-sha256", required=True)
    parser.add_argument("--musr-identity-order-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = refresh(args)
    print(
        json.dumps({"status": "complete", "receipt_sha256": receipt["receipt_sha256"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
