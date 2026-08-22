"""Create an exact-sequence-multiset order control for a Sai curriculum stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.token_stream import (
    SCHEMA as TOKEN_STREAM_SCHEMA,
)
from sai.data.token_stream import (
    canonical_sha256,
    sha256_file,
    validate_frozen_stream,
)

CONTROL_SCHEMA = "sai-curriculum-sequence-order-control-v1"
FROZEN_SEED = 2026082201


class CurriculumControlError(RuntimeError):
    """The parent stream, permutation, record multiset, or output differs."""


def _permutation(sequences: int, seed: int) -> list[int]:
    if (
        isinstance(sequences, bool)
        or not isinstance(sequences, int)
        or sequences <= 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise CurriculumControlError("sequence-control geometry differs")
    return sorted(
        range(sequences),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).digest(),
    )


def _permutation_sha256(permutation: list[int]) -> str:
    digest = hashlib.sha256()
    for value in permutation:
        digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


class _Records:
    def __init__(self, root: Path, report: dict[str, Any]) -> None:
        self._files = []
        self._maps: list[tuple[mmap.mmap, mmap.mmap]] = []
        self.sequence_length = report["sequence_length"]
        self.sequences_per_shard = report["sequences_per_shard"]
        self.token_bytes = self.sequence_length * 4
        self.start_bytes = (self.sequence_length + 7) // 8
        try:
            for shard in report["shards"]:
                token_file = (root / shard["tokens"]["path"]).open("rb")
                start_file = (root / shard["segment_starts"]["path"]).open("rb")
                self._files.extend((token_file, start_file))
                self._maps.append(
                    (
                        mmap.mmap(token_file.fileno(), 0, access=mmap.ACCESS_READ),
                        mmap.mmap(start_file.fileno(), 0, access=mmap.ACCESS_READ),
                    )
                )
        except BaseException:
            self.close()
            raise

    def record(self, index: int) -> tuple[bytes, bytes]:
        shard = index // self.sequences_per_shard
        local = index % self.sequences_per_shard
        if not 0 <= shard < len(self._maps):
            raise CurriculumControlError("sequence-control index differs")
        tokens, starts = self._maps[shard]
        token_offset = local * self.token_bytes
        start_offset = local * self.start_bytes
        token_record = tokens[token_offset : token_offset + self.token_bytes]
        start_record = starts[start_offset : start_offset + self.start_bytes]
        if (
            len(token_record) != self.token_bytes
            or len(start_record) != self.start_bytes
        ):
            raise CurriculumControlError("sequence-control record is truncated")
        return token_record, start_record

    def close(self) -> None:
        for pair in getattr(self, "_maps", []):
            for mapping in pair:
                mapping.close()
        self._maps = []
        for handle in getattr(self, "_files", []):
            handle.close()
        self._files = []

    def __enter__(self) -> _Records:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _record_sha256(tokens: bytes, starts: bytes) -> bytes:
    return hashlib.sha256(tokens + starts).digest()


def _multiset_sha256(records: _Records, sequences: int) -> str:
    identities = [_record_sha256(*records.record(index)) for index in range(sequences)]
    identities.sort()
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(identity)
    return digest.hexdigest()


def build_order_control(
    source: Path, output: Path, *, seed: int = FROZEN_SEED
) -> dict[str, Any]:
    """Permute exact packed records without changing any token or target mask."""

    parent = validate_frozen_stream(source, verify_sources=True)
    if output.exists() or output.is_symlink():
        raise CurriculumControlError("sequence-control output already exists")
    if seed != FROZEN_SEED:
        raise CurriculumControlError("sequence-control seed differs")
    sequences = parent["sequences"]
    permutation = _permutation(sequences, seed)
    stage = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    shards = []
    token_handle = start_handle = None
    shard_sequences = 0
    shard_index = 0

    def open_shard() -> tuple[Any, Any]:
        return (
            (stage / f"shard_{shard_index:05d}.tokens.u32le").open("wb"),
            (stage / f"shard_{shard_index:05d}.starts.bitset").open("wb"),
        )

    def close_shard() -> None:
        nonlocal token_handle, start_handle, shard_sequences
        if token_handle is None or start_handle is None:
            return
        token_path = Path(token_handle.name)
        start_path = Path(start_handle.name)
        token_handle.close()
        start_handle.close()
        shards.append(
            {
                "index": len(shards),
                "sequences": shard_sequences,
                "tokens": {
                    "path": token_path.name,
                    "bytes": token_path.stat().st_size,
                    "sha256": sha256_file(token_path),
                },
                "segment_starts": {
                    "path": start_path.name,
                    "bytes": start_path.stat().st_size,
                    "sha256": sha256_file(start_path),
                },
            }
        )
        token_handle = start_handle = None
        shard_sequences = 0

    try:
        with _Records(source, parent) as records:
            multiset_sha256 = _multiset_sha256(records, sequences)
            for parent_index in permutation:
                if token_handle is None:
                    token_handle, start_handle = open_shard()
                tokens, starts = records.record(parent_index)
                token_handle.write(tokens)
                start_handle.write(starts)
                shard_sequences += 1
                if shard_sequences == parent["sequences_per_shard"]:
                    close_shard()
                    shard_index += 1
            close_shard()
        parent_receipt = source / "stream_receipt.json"
        report: dict[str, Any] = {
            **{
                key: value
                for key, value in parent.items()
                if key
                not in {
                    "ordered_stream_identity_sha256",
                    "prefix_utf8_bytes",
                    "shards",
                    "curriculum",
                }
            },
            "schema": TOKEN_STREAM_SCHEMA,
            "prefix_utf8_bytes": {str(sequences): parent["admitted_utf8_bytes"]},
            "shards": shards,
            "ordering_control": {
                "schema": CONTROL_SCHEMA,
                "method": "sha256_ranked_exact_sequence_permutation",
                "seed": seed,
                "permutation_sha256": _permutation_sha256(permutation),
                "fixed_points": sum(
                    index == value for index, value in enumerate(permutation)
                ),
                "parent_stream": {
                    "path": str(source.resolve()),
                    "receipt_bytes": parent_receipt.stat().st_size,
                    "receipt_file_sha256": sha256_file(parent_receipt),
                    "ordered_stream_identity_sha256": parent[
                        "ordered_stream_identity_sha256"
                    ],
                },
                "sequence_multiset_sha256": multiset_sha256,
                "same_tokens_and_boundary_masks": True,
                "same_sequence_multiset": True,
                "only_sequence_order_changed": True,
            },
        }
        report["ordered_stream_identity_sha256"] = canonical_sha256(report)
        (stage / "stream_receipt.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        validate_order_control(stage)
        os.replace(stage, output)
        return report
    except BaseException:
        if token_handle is not None:
            token_handle.close()
        if start_handle is not None:
            start_handle.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_order_control(output: Path) -> dict[str, Any]:
    return _validate_order_control(output)


def _bind_qualified_source(
    report: dict[str, Any],
    expected: dict[str, Any],
    qualification_sha256: str,
) -> None:
    receipt = {
        "order": 0,
        "path": expected["path"],
        "bytes": expected["bytes"],
        "sha256": expected["sha256"],
    }
    if (
        report.get("source_receipts") != [receipt]
        or report.get("source_qualification_sha256") != qualification_sha256
    ):
        raise CurriculumControlError("qualified split source differs")


def _validate_order_control(
    output: Path,
    *,
    trusted_parent: tuple[Path, dict[str, Any]] | None = None,
    expected_source: dict[str, Any] | None = None,
    qualification_sha256: str | None = None,
) -> dict[str, Any]:
    report = validate_frozen_stream(
        output,
        verify_sources=trusted_parent is None,
    )
    if trusted_parent is not None:
        if expected_source is None or qualification_sha256 is None:
            raise CurriculumControlError("qualified split source differs")
        _bind_qualified_source(report, expected_source, qualification_sha256)
    ordering = report.get("ordering_control")
    if (
        not isinstance(ordering, dict)
        or ordering.get("schema") != CONTROL_SCHEMA
        or ordering.get("method") != "sha256_ranked_exact_sequence_permutation"
        or ordering.get("seed") != FROZEN_SEED
        or ordering.get("same_tokens_and_boundary_masks") is not True
        or ordering.get("same_sequence_multiset") is not True
        or ordering.get("only_sequence_order_changed") is not True
    ):
        raise CurriculumControlError("sequence-control receipt differs")
    parent_row = ordering.get("parent_stream")
    if not isinstance(parent_row, dict):
        raise CurriculumControlError("sequence-control parent differs")
    declared_parent_root = Path(parent_row.get("path", ""))
    if trusted_parent is None:
        parent_root = declared_parent_root
        parent = validate_frozen_stream(parent_root, verify_sources=True)
    else:
        parent_root, parent = trusted_parent
        if declared_parent_root != parent_root.resolve():
            raise CurriculumControlError("sequence-control parent differs")
        _bind_qualified_source(parent, expected_source, qualification_sha256)
    parent_receipt = parent_root / "stream_receipt.json"
    if (
        not parent_receipt.is_file()
        or parent_receipt.is_symlink()
        or parent_row.get("receipt_bytes") != parent_receipt.stat().st_size
        or parent_row.get("receipt_file_sha256") != sha256_file(parent_receipt)
    ):
        raise CurriculumControlError("sequence-control parent receipt differs")
    sequences = report["sequences"]
    permutation = _permutation(sequences, FROZEN_SEED)
    if (
        ordering.get("permutation_sha256") != _permutation_sha256(permutation)
        or ordering.get("fixed_points")
        != sum(index == value for index, value in enumerate(permutation))
        or parent_row.get("ordered_stream_identity_sha256")
        != parent["ordered_stream_identity_sha256"]
        or report["sequence_length"] != parent["sequence_length"]
        or report["admitted_utf8_bytes"] != parent["admitted_utf8_bytes"]
        or report["documents"] != parent["documents"]
        or report["source_receipts"] != parent["source_receipts"]
        or report.get("source_qualification_sha256")
        != parent.get("source_qualification_sha256")
    ):
        raise CurriculumControlError("sequence-control parent geometry differs")
    with (
        _Records(parent_root, parent) as parent_records,
        _Records(output, report) as control_records,
    ):
        parent_multiset = _multiset_sha256(parent_records, sequences)
        control_multiset = _multiset_sha256(control_records, sequences)
        if (
            parent_multiset != control_multiset
            or ordering.get("sequence_multiset_sha256") != parent_multiset
        ):
            raise CurriculumControlError("sequence-control multiset differs")
        for output_index, parent_index in enumerate(permutation):
            if control_records.record(output_index) != parent_records.record(
                parent_index
            ):
                raise CurriculumControlError("sequence-control permutation differs")
    return report


def validate_curriculum_order_bundle(
    curriculum_root: Path,
    control_root: Path,
    development_root: Path,
    split_receipt: Path,
    *,
    curriculum_workers: int = 1,
) -> dict[str, dict[str, Any]]:
    """Replay the split once, then bind all three streams to its exact outputs."""

    from sai.data.curriculum_split import validate_curriculum_split

    split = validate_curriculum_split(
        split_receipt,
        curriculum_workers=curriculum_workers,
    )
    qualification_sha256 = sha256_file(split_receipt)
    curriculum = validate_frozen_stream(curriculum_root, verify_sources=False)
    development = validate_frozen_stream(development_root, verify_sources=False)
    _bind_qualified_source(curriculum, split["train"], qualification_sha256)
    _bind_qualified_source(development, split["development"], qualification_sha256)
    control = _validate_order_control(
        control_root,
        trusted_parent=(curriculum_root, curriculum),
        expected_source=split["train"],
        qualification_sha256=qualification_sha256,
    )
    return {
        "split": split,
        "curriculum": curriculum,
        "control": control,
        "development": development,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=FROZEN_SEED)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_order_control(args.source, args.output, seed=args.seed)
    else:
        payload = validate_order_control(args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ordered_stream_identity_sha256": payload[
                    "ordered_stream_identity_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
