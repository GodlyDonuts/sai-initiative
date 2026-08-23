"""Select deterministic agent-label candidates from Sai pretraining documents."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import CANDIDATE_SCHEMA, SOURCE_TYPES, normalize_candidate
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-agent-data-candidate-population-v1"


class AgentCandidatePopulationError(RuntimeError):
    """A source population, selection identity, or output differs."""


def _regular(path: Path) -> os.stat_result:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise AgentCandidatePopulationError(
            "source population is missing or unsafe"
        ) from error
    try:
        result = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1 or result.st_size <= 0:
        raise AgentCandidatePopulationError("source population is missing or unsafe")
    return result


def _selection_key(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _candidate(
    document: dict[str, Any], *, source_revision: str, source_type: str
) -> dict[str, Any]:
    text = document["text"]
    source = document["source"]
    provenance = canonical_sha256(
        {
            "document_identity_sha256": document["identity_sha256"],
            "source": source,
            "verification": document["verification"],
        }
    )
    row = {
        "schema": CANDIDATE_SCHEMA,
        "text": text,
        "source": {
            "dataset": source["dataset"],
            "revision": source_revision,
            "row_id": source["row_id"],
            "license": source["license"],
            "source_type": source_type,
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": provenance,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return normalize_candidate(row)


def build_population(
    source_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    source_revision: str,
    source_type: str,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Select the lowest stable hash keys without loading the source into memory."""

    before = _regular(source_path)
    if (
        not isinstance(source_revision, str)
        or not source_revision
        or source_type not in SOURCE_TYPES
        or isinstance(sample_size, bool)
        or not 1 <= sample_size <= 1_000_000
        or isinstance(seed, bool)
        or not 0 <= seed < 2**63
    ):
        raise AgentCandidatePopulationError("candidate selection arguments differ")
    if (
        output_path.exists()
        or output_path.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise AgentCandidatePopulationError("candidate output already exists")
    heap: list[tuple[int, int, dict[str, Any]]] = []
    source_rows = 0
    with source_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                document = normalize_document(json.loads(line))
            except (json.JSONDecodeError, RuntimeError) as error:
                raise AgentCandidatePopulationError(
                    f"source row {line_number} differs"
                ) from error
            source_rows += 1
            key = int(_selection_key(seed, document["identity_sha256"]), 16)
            item = (-key, line_number, document)
            if len(heap) < sample_size:
                heapq.heappush(heap, item)
            elif key < -heap[0][0]:
                heapq.heapreplace(heap, item)
    after = source_path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise AgentCandidatePopulationError("source changed while selecting candidates")
    if source_rows < sample_size:
        raise AgentCandidatePopulationError("source has fewer rows than requested")
    selected = sorted(
        (
            _selection_key(seed, document["identity_sha256"]),
            line_number,
            _candidate(
                document, source_revision=source_revision, source_type=source_type
            ),
        )
        for _, line_number, document in heap
    )
    identities = [row[2]["candidate_identity_sha256"] for row in selected]
    if len(identities) != len(set(identities)):
        raise AgentCandidatePopulationError(
            "selected candidate identities are duplicated"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("x") as handle:
        for _, _, row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {
            "path": str(source_path.resolve()),
            "bytes": before.st_size,
            "rows": source_rows,
            "sha256": sha256_file(source_path),
            "revision": source_revision,
            "source_type": source_type,
        },
        "selection": {
            "method": "lowest_sha256_of_seed_colon_document_identity",
            "seed": seed,
            "sample_size": sample_size,
            "ordered_selection_sha256": canonical_sha256(
                [
                    {
                        "selection_key": key,
                        "source_line_number": line_number,
                        "candidate_identity_sha256": row["candidate_identity_sha256"],
                    }
                    for key, line_number, row in selected
                ]
            ),
        },
        "population": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "rows": len(selected),
            "sha256": sha256_file(output_path),
        },
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    temporary = receipt_path.parent / f".{receipt_path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("x") as handle:
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-type", choices=SOURCE_TYPES, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    receipt = build_population(
        args.source,
        args.output,
        args.receipt,
        source_revision=args.source_revision,
        source_type=args.source_type,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
