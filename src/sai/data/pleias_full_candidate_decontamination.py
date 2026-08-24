"""Filter advanced PleIAs strata against the full official benchmark boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Container
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_overlap_count,
    _normalize,
    _overlap_count,
    binary_boundary_index,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    AGGREGATE_SCHEMA as BOUNDED_AGGREGATE_SCHEMA,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    CANDIDATE_SCHEMA as BOUNDED_CANDIDATE_SCHEMA,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    SHARD_SCHEMA as BOUNDED_SHARD_SCHEMA,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_semantic_sample import _token_band
from sai.data.pleias_semantic_stratum_decision import (
    SCHEMA as STRATUM_DECISION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-pleias-full-candidate-decontamination-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-full-candidate-decontamination-aggregate-v1"


class PleiasFullCandidateDecontaminationError(RuntimeError):
    """Candidate custody, semantic advancement, or benchmark overlap differs."""


class _Union:
    def __init__(self, members: list[Container[bytes]]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasFullCandidateDecontaminationError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasFullCandidateDecontaminationError(
            "signed input is invalid"
        ) from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise PleiasFullCandidateDecontaminationError("signed input differs")
    return payload


def _advanced_strata(decision: dict[str, Any]) -> frozenset[str]:
    rows = decision.get("decisions")
    advanced = decision.get("advanced_strata")
    if (
        decision.get("status")
        != "complete_nontraining_pleias_semantic_stratum_decision"
        or not isinstance(rows, list)
        or not isinstance(advanced, list)
        or len(advanced) != len(set(advanced))
    ):
        raise PleiasFullCandidateDecontaminationError("stratum decision differs")
    replay = []
    expected = []
    for row in rows:
        unsigned = {key: value for key, value in row.items() if key != "row_sha256"}
        if (
            not isinstance(row, dict)
            or row.get("row_sha256") != canonical_sha256(unsigned)
            or row.get("automatic_training_admission") is not False
        ):
            raise PleiasFullCandidateDecontaminationError(
                "stratum decision row differs"
            )
        replay.append(row["row_sha256"])
        if row.get("decision") == "advance_to_full_candidate_decontamination":
            expected.append(row.get("stratum"))
    if (
        canonical_sha256(replay) != decision.get("ordered_decisions_sha256")
        or expected != advanced
        or any(not isinstance(value, str) or not value for value in advanced)
    ):
        raise PleiasFullCandidateDecontaminationError(
            "advanced stratum coverage differs"
        )
    return frozenset(advanced)


def screen_text(
    text: str,
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> tuple[int, int]:
    """Return exact word/code boundary overlap counts for one full candidate."""

    if not isinstance(text, str) or not text:
        raise PleiasFullCandidateDecontaminationError("candidate text differs")
    normalized = _normalize(text)
    return (
        _overlap_count(
            _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
        ),
        _code_overlap_count(_CODE.findall(normalized), code_boundary),
    )


def run_shard(
    manifest_path: Path,
    bounded_root: Path,
    bounded_aggregate_path: Path,
    decision_path: Path,
    boundary_roots: list[Path],
    output_root: Path,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Decontaminate one exact bounded shard and emit a global-dedup index."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not 0 <= shard_index < logical_shards
        or not boundary_roots
    ):
        raise PleiasFullCandidateDecontaminationError(
            "decontamination arguments differ"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasFullCandidateDecontaminationError("pyarrow is required") from error
    manifest = load_manifest(manifest_path)
    expected_parents = select_shard(manifest, logical_shards, shard_index)
    aggregate = _load_signed(bounded_aggregate_path, BOUNDED_AGGREGATE_SCHEMA)
    if (
        aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("complete_source_parent_coverage") is not True
    ):
        raise PleiasFullCandidateDecontaminationError("bounded aggregate differs")
    source_root = bounded_root / "shards" / f"shard_{shard_index:05d}"
    source_receipt = _load_signed(source_root / "receipt.json", BOUNDED_SHARD_SCHEMA)
    source_path = source_root / source_receipt.get("output", {}).get("path", "")
    if (
        source_receipt.get("logical_shards") != logical_shards
        or source_receipt.get("shard_index") != shard_index
        or source_receipt.get("source", {}).get("selected_paths_sha256")
        != canonical_sha256([row["source_path"] for row in expected_parents])
        or not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_nlink != 1
        or source_path.stat().st_size != source_receipt["output"]["bytes"]
        or sha256_file(source_path) != source_receipt["output"]["sha256"]
    ):
        raise PleiasFullCandidateDecontaminationError("bounded shard differs")
    decision = _load_signed(decision_path, STRATUM_DECISION_SCHEMA)
    advanced = _advanced_strata(decision)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    output_root.mkdir(parents=True)
    output_path = output_root / "benchmark_disjoint_candidates.parquet"
    temporary = output_root / f".candidates.partial.{uuid.uuid4().hex}.parquet"
    index_path = output_root / "global_exact_dedup_index.jsonl"
    index_temporary = output_root / f".index.partial.{uuid.uuid4().hex}.jsonl"
    index_handle = index_temporary.open("x")
    index_rows = 0
    index_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    seen_content = set()
    source = pq.ParquetFile(source_path)
    writer = pq.ParquetWriter(temporary, source.schema_arrow, compression="zstd")
    retained_text_bytes = 0
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        try:
            for batch in source.iter_batches(batch_size=16, use_threads=False):
                output_rows = []
                for row in batch.to_pylist():
                    counts["source_rows"] += 1
                    text = row.get("text")
                    content_sha256 = row.get("content_sha256")
                    if (
                        row.get("schema") != BOUNDED_CANDIDATE_SCHEMA
                        or row.get("training_ready") is not False
                        or not isinstance(text, str)
                        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
                    ):
                        raise PleiasFullCandidateDecontaminationError(
                            "bounded candidate differs"
                        )
                    stratum = "::".join(
                        (
                            row["collection"],
                            row["open_type"],
                            _token_band(row["token_count"]),
                        )
                    )
                    if stratum not in advanced:
                        counts["held_semantic_stratum"] += 1
                        continue
                    word_overlap, code_overlap = screen_text(
                        text, word_boundary, code_boundary
                    )
                    counts["word_overlap_shingles"] += word_overlap
                    counts["code_overlap_shingles"] += code_overlap
                    if word_overlap or code_overlap:
                        counts["benchmark_contaminated"] += 1
                        continue
                    if content_sha256 in seen_content:
                        counts["within_shard_exact_duplicate"] += 1
                        continue
                    seen_content.add(content_sha256)
                    output_rows.append(row)
                    counts["retained_candidates"] += 1
                    retained_text_bytes += len(text.encode())
                    index_row = {
                        "content_sha256": content_sha256,
                        "source_row_identity_sha256": row["source_row_identity_sha256"],
                        "shard_index": shard_index,
                        "source_row_index": row["source_row_index"],
                        "stratum": stratum,
                    }
                    index_handle.write(
                        json.dumps(index_row, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    index_digest.update(bytes.fromhex(canonical_sha256(index_row)))
                    index_rows += 1
                if output_rows:
                    writer.write_table(
                        pa.Table.from_pylist(output_rows, schema=source.schema_arrow)
                    )
        except BaseException:
            writer.close()
            index_handle.close()
            temporary.unlink(missing_ok=True)
            index_temporary.unlink(missing_ok=True)
            raise
        writer.close()
        index_handle.flush()
        os.fsync(index_handle.fileno())
        index_handle.close()
        os.replace(temporary, output_path)
        os.replace(index_temporary, index_path)
    finally:
        for member in [*words, *code]:
            member.close()
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_full_candidate_decontamination_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "bounded_aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "bounded_shard_receipt_sha256": source_receipt["receipt_sha256"],
            "stratum_decision_receipt_sha256": decision["receipt_sha256"],
        },
        "boundary_indexes": boundary_receipts,
        "boundary_indexes_sha256": canonical_sha256(boundary_receipts),
        "decontamination_policy": POLICY,
        "decontamination_policy_sha256": canonical_sha256(POLICY),
        "counts": dict(sorted(counts.items())),
        "retained_text_utf8_bytes": retained_text_bytes,
        "output": {
            "path": output_path.name,
            "rows": counts["retained_candidates"],
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "global_exact_dedup_index": {
            "path": index_path.name,
            "rows": index_rows,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "ordered_row_digests_sha256": index_digest.hexdigest(),
        },
        "full_candidate_benchmark_decontamination_complete": True,
        "within_shard_exact_deduplication_complete": True,
        "global_exact_deduplication_complete": False,
        "global_near_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate(
    manifest_path: Path,
    bounded_aggregate_path: Path,
    decision_path: Path,
    shards_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify every decontamination shard and seal exact aggregate custody."""

    if output.exists() or output.is_symlink():
        raise PleiasFullCandidateDecontaminationError("aggregate output exists")
    manifest = load_manifest(manifest_path)
    bounded = _load_signed(bounded_aggregate_path, BOUNDED_AGGREGATE_SCHEMA)
    decision = _load_signed(decision_path, STRATUM_DECISION_SCHEMA)
    receipts = []
    totals: Counter[str] = Counter()
    boundary_sha256 = None
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", SHARD_SCHEMA)
        output_path = root / receipt.get("output", {}).get("path", "")
        index_path = root / receipt.get("global_exact_dedup_index", {}).get("path", "")
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("bounded_aggregate_receipt_sha256")
            != bounded["receipt_sha256"]
            or receipt.get("source", {}).get("stratum_decision_receipt_sha256")
            != decision["receipt_sha256"]
            or not output_path.is_file()
            or output_path.stat().st_size != receipt["output"]["bytes"]
            or sha256_file(output_path) != receipt["output"]["sha256"]
            or not index_path.is_file()
            or index_path.stat().st_size != receipt["global_exact_dedup_index"]["bytes"]
            or sha256_file(index_path) != receipt["global_exact_dedup_index"]["sha256"]
        ):
            raise PleiasFullCandidateDecontaminationError(
                "decontamination shard differs"
            )
        current_boundary = receipt["boundary_indexes_sha256"]
        if boundary_sha256 is None:
            boundary_sha256 = current_boundary
        elif current_boundary != boundary_sha256:
            raise PleiasFullCandidateDecontaminationError(
                "decontamination boundary differs"
            )
        for key, value in receipt["counts"].items():
            totals[key] += value
        totals["retained_text_utf8_bytes"] += receipt["retained_text_utf8_bytes"]
        totals["output_bytes"] += receipt["output"]["bytes"]
        receipts.append(receipt["receipt_sha256"])
    if totals["source_rows"] != bounded.get("totals", {}).get("selected_rows"):
        raise PleiasFullCandidateDecontaminationError(
            "decontamination source coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_full_candidate_decontamination",
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "source_parent_count": len(manifest),
            "bounded_aggregate_receipt_sha256": bounded["receipt_sha256"],
            "stratum_decision_receipt_sha256": decision["receipt_sha256"],
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "boundary_indexes_sha256": boundary_sha256,
        "totals": dict(sorted(totals.items())),
        "complete_bounded_candidate_coverage": True,
        "full_candidate_benchmark_decontamination_complete": True,
        "within_shard_exact_deduplication_complete": True,
        "global_exact_deduplication_complete": False,
        "global_near_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--bounded-root", type=Path, required=True)
    shard.add_argument("--bounded-aggregate", type=Path, required=True)
    shard.add_argument("--decision", type=Path, required=True)
    shard.add_argument("--boundary-index", type=Path, action="append", required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--manifest", type=Path, required=True)
    combine.add_argument("--bounded-aggregate", type=Path, required=True)
    combine.add_argument("--decision", type=Path, required=True)
    combine.add_argument("--shards-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        result = run_shard(
            args.manifest,
            args.bounded_root,
            args.bounded_aggregate,
            args.decision,
            args.boundary_index,
            args.output_root,
            args.logical_shards,
            args.shard_index,
        )
    else:
        result = aggregate(
            args.manifest,
            args.bounded_aggregate,
            args.decision,
            args.shards_root,
            args.logical_shards,
            args.output,
        )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
