from __future__ import annotations

import json
from pathlib import Path

from sai.data.pleias_parent_disjoint_audit_aggregate import build_aggregate
from sai.data.pleias_parent_disjoint_audit_population import EXPECTED_ROWS, SOURCE_ID
from sai.data.reservoir_audit_population import SCHEMA, _write_jsonl
from sai.data.token_stream import canonical_sha256, sha256_file


def _shards(root: Path) -> Path:
    root.mkdir()
    for shard_index in range(8):
        shard = root / f"shard_{shard_index:03d}"
        shard.mkdir()
        lineage = []
        candidates = []
        for ordinal in range(shard_index, EXPECTED_ROWS, 8):
            identity = f"{ordinal + 1:064x}"
            candidates.append(
                {
                    "schema": "test-candidate",
                    "candidate_identity_sha256": identity,
                    "text": f"row {ordinal}",
                }
            )
            lineage.append(
                {
                    "ordinal": ordinal,
                    "candidate_identity_sha256": identity,
                    "repository": "PleIAs/common_corpus",
                    "revision": "a" * 40,
                    "path": f"common_corpus_{ordinal % 10 + 1}/p-{ordinal}.parquet",
                    "stratum": f"open_corpus_partition:{ordinal % 10 + 1}",
                    "full_file_content_verified": True,
                }
            )
        candidate_path = shard / "candidates.jsonl"
        lineage_path = shard / "lineage.jsonl"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "source_id": SOURCE_ID,
            "global_plan_rows": EXPECTED_ROWS,
            "logical_shards": 8,
            "shard_index": shard_index,
            "acquisition_mode": "full_verified_parent",
            "maximum_simultaneous_parent_files": 1,
            "temporary_parent_removed_after_each_row": True,
            "fully_verified_parent_files": 128,
            "benchmark_decontamination_complete": False,
            "hermes_judgments_complete": False,
            "training_ready": False,
            "population": {
                "path": candidate_path.name,
                "rows": 128,
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": 128,
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        (shard / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    return root


def test_combines_all_eight_shards_in_global_ordinal_order(tmp_path: Path) -> None:
    output = tmp_path / "aggregate"
    result = build_aggregate(_shards(tmp_path / "shards"), output)
    assert result["population"]["rows"] == EXPECTED_ROWS
    assert result["lineage"]["rows"] == EXPECTED_ROWS
    assert result["unique_parent_files"] == EXPECTED_ROWS
    assert result["fully_verified_parent_files"] == EXPECTED_ROWS
    assert result["benchmark_decontamination_complete"] is False
    values = [
        json.loads(line)
        for line in (output / "lineage.jsonl").read_text().splitlines()
    ]
    assert [row["ordinal"] for row in values] == list(range(EXPECTED_ROWS))
