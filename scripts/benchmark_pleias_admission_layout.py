#!/usr/bin/env python3
"""Compare legacy and split-payload exact-dedup layouts on real locators."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_practical_admission import (
    _SPLIT_PAYLOAD_INSERT,
    _SPLIT_PAYLOAD_UPSERT,
    _UPSERT,
    _open_database,
    _output_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-admission-layout-benchmark-v1"


def _values(path: Path, row_limit: int) -> Any:
    observed = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, use_threads=False):
        values = []
        for row in batch.to_pylist():
            if observed >= row_limit:
                break
            observed += 1
            values.append(
                (
                    row["content_sha256"],
                    row["source_row_identity_sha256"],
                    _output_shard(row["source_path"], 128),
                    row["text_utf8_bytes"],
                    row["source_token_count"],
                    row["license"],
                    json.dumps(row, sort_keys=True, separators=(",", ":")),
                )
            )
        if values:
            yield values
        if observed >= row_limit:
            return


def _run_layout(
    input_path: Path,
    row_limit: int,
    scratch_root: Path,
    split_payload_dedup: bool,
) -> dict[str, Any]:
    started = time.monotonic_ns()
    with tempfile.TemporaryDirectory(
        prefix="sai-pleias-layout-benchmark-", dir=scratch_root
    ) as temporary_directory:
        database_path = Path(temporary_directory) / "exact-dedup.sqlite3"
        database = _open_database(
            database_path, split_payload_dedup=split_payload_dedup
        )
        input_rows = 0
        next_payload_id = 0
        try:
            for values in _values(input_path, row_limit):
                input_rows += len(values)
                if split_payload_dedup:
                    payload_values = []
                    index_values = []
                    for value in values:
                        next_payload_id += 1
                        payload_values.append((next_payload_id, value[-1]))
                        index_values.append((*value[:-1], next_payload_id))
                    database.executemany(_SPLIT_PAYLOAD_INSERT, payload_values)
                    database.executemany(_SPLIT_PAYLOAD_UPSERT, index_values)
                else:
                    database.executemany(_UPSERT, values)
            database.commit()
            if split_payload_dedup:
                cursor = database.execute(
                    "SELECT winners.content_sha256, winners.identity_sha256, "
                    "winners.output_shard, winners.text_utf8_bytes, "
                    "winners.source_token_count, winners.license, "
                    "payloads.row_json FROM winners "
                    "JOIN payloads USING (payload_id) "
                    "ORDER BY winners.content_sha256"
                )
            else:
                cursor = database.execute(
                    "SELECT content_sha256, identity_sha256, output_shard, "
                    "text_utf8_bytes, source_token_count, license, row_json "
                    "FROM winners ORDER BY content_sha256"
                )
            digest = hashlib.sha256()
            winner_rows = 0
            for row in cursor:
                winner_rows += 1
                digest.update(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
                )
                digest.update(b"\n")
            database_bytes = database_path.stat().st_size
        finally:
            database.close()
    return {
        "layout": (
            "split_sequential_payload_v1"
            if split_payload_dedup
            else "inline_payload_v1"
        ),
        "input_rows": input_rows,
        "winner_rows": winner_rows,
        "ordered_winners_sha256": digest.hexdigest(),
        "database_bytes": database_bytes,
        "elapsed_nanoseconds": time.monotonic_ns() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.input.is_file()
        or args.input.is_symlink()
        or args.rows < 1
        or not args.scratch_root.is_dir()
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise SystemExit("unsafe benchmark arguments")
    legacy = _run_layout(args.input, args.rows, args.scratch_root, False)
    split = _run_layout(args.input, args.rows, args.scratch_root, True)
    equivalent = bool(
        legacy["input_rows"] == split["input_rows"]
        and legacy["winner_rows"] == split["winner_rows"]
        and legacy["ordered_winners_sha256"]
        == split["ordered_winners_sha256"]
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete_equivalent_layout_benchmark" if equivalent else "failed",
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
            "requested_rows": args.rows,
        },
        "legacy": legacy,
        "split_payload": split,
        "exact_ordered_winner_equivalence": equivalent,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
