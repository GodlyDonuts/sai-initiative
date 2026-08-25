"""Measure exact curriculum-band token estimates for Sai's released 1B index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_curriculum_index import AGGREGATE_SCHEMA, BANDS
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-1b-unique-token-ledger-v1"


class OneBUniqueTokenLedgerError(RuntimeError):
    """The curriculum aggregate, Parquet population, or token ledger differs."""


def _aggregate(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBUniqueTokenLedgerError("curriculum aggregate differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or value.get("schema") != AGGREGATE_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBUniqueTokenLedgerError("curriculum aggregate differs")
    return value


def _index_paths(root: Path) -> list[Path]:
    paths = [root / "books" / "index.parquet"]
    paths.extend(
        root / "pleias" / f"shard_{index:05d}" / "index.parquet"
        for index in range(128)
    )
    paths.extend(
        root / "code" / f"shard_{index:05d}" / "index.parquet"
        for index in range(32)
    )
    paths.append(root / "connections" / "index.parquet")
    return paths


def build(root: Path, output: Path) -> dict[str, Any]:
    """Scan each indexed row once and seal per-band/component token estimates."""

    if output.exists() or output.is_symlink():
        raise OneBUniqueTokenLedgerError("token ledger output exists")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBUniqueTokenLedgerError("pyarrow is required") from error
    aggregate = _aggregate(root / "aggregate.json")
    counts: Counter[str] = Counter()
    files = []
    for path in _index_paths(root):
        if not path.is_file() or path.is_symlink():
            raise OneBUniqueTokenLedgerError("curriculum index file differs")
        rows = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=32_768,
            columns=(
                "component",
                "source_token_estimate",
                "text_utf8_bytes",
                "curriculum_band",
                "split",
            ),
            use_threads=False,
        ):
            for row in batch.to_pylist():
                band = row["curriculum_band"]
                split = row["split"]
                component = row["component"]
                if band not in BANDS or split not in {"train", "development"}:
                    raise OneBUniqueTokenLedgerError("curriculum row differs")
                rows += 1
                counts["rows"] += 1
                counts["source_token_estimate"] += row["source_token_estimate"]
                counts["text_utf8_bytes"] += row["text_utf8_bytes"]
                for prefix in (
                    f"band::{band}",
                    f"split::{split}",
                    f"component::{component}",
                    f"band::{band}::split::{split}",
                ):
                    counts[f"{prefix}::rows"] += 1
                    counts[f"{prefix}::source_token_estimate"] += row[
                        "source_token_estimate"
                    ]
                    counts[f"{prefix}::text_utf8_bytes"] += row["text_utf8_bytes"]
        files.append(
            {
                "path": str(path.relative_to(root)),
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if (
        counts["rows"] != aggregate["counts"]["rows"]
        or counts["text_utf8_bytes"] != aggregate["counts"]["text_utf8_bytes"]
    ):
        raise OneBUniqueTokenLedgerError("token ledger release coverage differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_1b_unique_token_ledger",
        "curriculum_aggregate_receipt_sha256": aggregate["receipt_sha256"],
        "files": files,
        "files_sha256": canonical_sha256(files),
        "counts": dict(sorted(counts.items())),
        "source_token_estimates_are_not_production_48k_counts": True,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.root, args.output)
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
