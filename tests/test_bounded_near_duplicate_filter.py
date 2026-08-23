from __future__ import annotations

import json
from pathlib import Path

from sai.data.bounded_near_duplicate_filter import build_filter, find_groups
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256


def _document(text: str, row: int) -> dict:
    payload = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "test/source",
            "row_id": f"row-{row}",
            "license": "CC0-1.0",
            "domain": "english",
        },
        "verification": {"benchmark_disjoint": True, "evidence_sha256": "f" * 64},
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def test_exact_sparse_join_finds_near_copy_and_ignores_unrelated() -> None:
    base = " ".join(f"token{index}" for index in range(120))
    near = " ".join(
        f"changed{index}" if index < 5 else f"token{index}" for index in range(120)
    )
    unrelated = " ".join(f"other{index}" for index in range(120))
    documents = [_document(base, 0), _document(near, 1), _document(unrelated, 2)]
    groups, evidence, geometry = find_groups(documents, block_rows=1)
    assert len(groups) == 1
    assert groups[0]["member_count"] == 2
    assert evidence["candidate_pairs_logically_covered"] == 3
    assert evidence["threshold_pair_matches"] == 1
    assert geometry["total_shingle_occurrences"] == 348


def test_filter_keeps_deterministic_canonical_survivor(tmp_path: Path) -> None:
    repeated = " ".join(f"concept{index}" for index in range(100))
    documents = [
        _document(repeated, 0),
        _document(repeated, 1),
        _document(" ".join(f"unique{index}" for index in range(100)), 2),
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in documents))
    output = tmp_path / "output.jsonl"
    receipt = tmp_path / "receipt.json"
    result = build_filter(source, output, receipt, block_rows=2)
    written = [json.loads(line) for line in output.read_text().splitlines()]
    survivor = min(documents[0]["identity_sha256"], documents[1]["identity_sha256"])
    assert result["evidence"]["documents_dropped"] == 1
    assert len(written) == 2
    assert survivor in {row["identity_sha256"] for row in written}
    assert result[
        "bounded_pilot_exact_and_high_confidence_near_duplicate_filter_complete"
    ] is True
    assert result["global_cross_source_near_duplicate_filter_complete"] is False
