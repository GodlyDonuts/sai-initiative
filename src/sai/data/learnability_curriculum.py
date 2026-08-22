"""Build an exact-record, model-centric learnability curriculum.

This module deliberately operates *after* source selection, decontamination,
tokenization, and packing.  It changes only the order of already frozen packed
records.  A separately frozen weak/strong checkpoint pair supplies per-record
normalized loss scores; neither the treatment checkpoint nor terminal
benchmark feedback may participate.
"""

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

from sai.data.curriculum_control import _multiset_sha256, _record_sha256, _Records
from sai.data.token_stream import (
    SCHEMA as TOKEN_STREAM_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file, validate_frozen_stream

POLICY_SCHEMA = "sai-model-centric-learnability-policy-v1"
SCORE_SCHEMA = "sai-model-centric-learnability-score-v1"
SCHEDULE_SCHEMA = "sai-model-centric-learnability-curriculum-v1"
PHASES = ("grounding", "integration", "reasoning", "specialization")
BANDS = ("ready", "developing", "challenging", "stretch")
ORDER_SEED = 2026082202
_MAX_POLICY_BYTES = 8 << 20
_MAX_SCORE_BYTES = 512 << 20
_POLICY_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "source_stream_identity_sha256",
    "source_receipt_file_sha256",
    "sequence_count",
    "sequences_per_update",
    "phase_order",
    "band_order",
    "phase_sequence_counts",
    "band_sequence_counts",
    "phase_by_band_counts",
    "scoring",
    "within_phase_order",
    "controls",
    "receipt_sha256",
}
_SCORING_KEYS = {
    "method",
    "weak_checkpoint_sha256",
    "strong_checkpoint_sha256",
    "tokenizer_sha256",
    "evaluator_sha256",
    "runtime_sha256",
    "treatment_checkpoint_used",
    "terminal_benchmark_feedback_used",
}
_ORDER_KEYS = {"method", "seed"}
_CONTROL_KEYS = {
    "same_sequence_multiset",
    "same_source_bytes",
    "tokenizer_factor_isolated",
    "architecture_factor_isolated",
    "only_score_to_order_changed",
}
_SCORE_KEYS = {
    "schema",
    "sequence_index",
    "record_sha256",
    "target_count",
    "weak_nll_microunits_per_target",
    "strong_nll_microunits_per_target",
    "preference_delta_microunits",
}


class LearnabilityCurriculumError(RuntimeError):
    """A score, policy, packed record, permutation, or receipt differs."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LearnabilityCurriculumError(f"{label} fields differ")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise LearnabilityCurriculumError(f"{label} differs")
    return value


def _nonnegative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LearnabilityCurriculumError(f"{label} differs")
    return value


def _positive(value: Any, label: str) -> int:
    value = _nonnegative(value, label)
    if value == 0:
        raise LearnabilityCurriculumError(f"{label} differs")
    return value


def _read_regular(path: Path, label: str, *, maximum_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise LearnabilityCurriculumError(f"{label} is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise LearnabilityCurriculumError(f"{label} is missing or unsafe")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)

    def identity(row: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
        )

    if len(encoded) != before.st_size or identity(before) != identity(after):
        raise LearnabilityCurriculumError(f"{label} changed while reading")
    return encoded


def validate_policy_payload(
    payload: Any,
    *,
    source_stream_identity_sha256: str,
    source_receipt_file_sha256: str,
    sequence_count: int,
) -> dict[str, Any]:
    """Validate a score-to-order policy frozen before score inspection."""

    policy = _exact(payload, _POLICY_KEYS, "learnability policy")
    unsigned = {key: value for key, value in policy.items() if key != "receipt_sha256"}
    if (
        policy["schema"] != POLICY_SCHEMA
        or policy["status"] != "prospective"
        or policy["training_authorized"] is not False
        or policy["four_b_training_authorized"] is not False
        or policy["receipt_sha256"] != canonical_sha256(unsigned)
        or _sha256(policy["source_stream_identity_sha256"], "source stream")
        != source_stream_identity_sha256
        or _sha256(policy["source_receipt_file_sha256"], "source receipt")
        != source_receipt_file_sha256
        or _positive(policy["sequence_count"], "sequence count") != sequence_count
        or policy["phase_order"] != list(PHASES)
        or policy["band_order"] != list(BANDS)
    ):
        raise LearnabilityCurriculumError("learnability policy identity differs")
    sequences_per_update = _positive(
        policy["sequences_per_update"], "sequences per update"
    )
    phase_counts = _exact(policy["phase_sequence_counts"], set(PHASES), "phase counts")
    band_counts = _exact(policy["band_sequence_counts"], set(BANDS), "band counts")
    matrix = _exact(policy["phase_by_band_counts"], set(PHASES), "phase-band counts")
    for phase in PHASES:
        matrix[phase] = _exact(matrix[phase], set(BANDS), "phase-band row")
    if (
        any(_positive(phase_counts[phase], "phase count") <= 0 for phase in PHASES)
        or any(_positive(band_counts[band], "band count") <= 0 for band in BANDS)
        or sum(phase_counts.values()) != sequence_count
        or sum(band_counts.values()) != sequence_count
    ):
        raise LearnabilityCurriculumError("learnability population differs")
    for phase in PHASES:
        if (
            any(
                _nonnegative(matrix[phase][band], "phase-band count") < 0
                for band in BANDS
            )
            or sum(matrix[phase].values()) != phase_counts[phase]
        ):
            raise LearnabilityCurriculumError("learnability phase allocation differs")
    for band in BANDS:
        if sum(matrix[phase][band] for phase in PHASES) != band_counts[band]:
            raise LearnabilityCurriculumError("learnability band allocation differs")
    if matrix["grounding"]["stretch"] != 0 or any(
        matrix[phase]["ready"] <= 0 for phase in PHASES[1:]
    ):
        raise LearnabilityCurriculumError("learnability rehearsal boundary differs")
    cumulative = 0
    means = []
    for phase in PHASES:
        cumulative += phase_counts[phase]
        if phase != PHASES[-1] and cumulative % sequences_per_update:
            raise LearnabilityCurriculumError("learnability phase boundary differs")
        numerator = sum(BANDS.index(band) * matrix[phase][band] for band in BANDS)
        means.append(numerator / phase_counts[phase])
    if any(later <= earlier for earlier, later in zip(means, means[1:], strict=False)):
        raise LearnabilityCurriculumError("learnability difficulty does not increase")
    scoring = _exact(policy["scoring"], _SCORING_KEYS, "scoring contract")
    if (
        scoring["method"] != "weak_minus_strong_normalized_nll_microunits"
        or _sha256(scoring["weak_checkpoint_sha256"], "weak checkpoint")
        == _sha256(scoring["strong_checkpoint_sha256"], "strong checkpoint")
        or scoring["treatment_checkpoint_used"] is not False
        or scoring["terminal_benchmark_feedback_used"] is not False
    ):
        raise LearnabilityCurriculumError("learnability scoring contract differs")
    for field in ("tokenizer_sha256", "evaluator_sha256", "runtime_sha256"):
        _sha256(scoring[field], field.replace("_", " "))
    ordering = _exact(policy["within_phase_order"], _ORDER_KEYS, "within-phase order")
    if (
        ordering["method"] != "sha256_ranked_without_score_order_within_phase"
        or ordering["seed"] != ORDER_SEED
    ):
        raise LearnabilityCurriculumError("within-phase order differs")
    controls = _exact(policy["controls"], _CONTROL_KEYS, "learnability controls")
    if any(controls[key] is not True for key in _CONTROL_KEYS):
        raise LearnabilityCurriculumError("learnability controls differ")
    return policy


def load_policy(
    path: Path, source: Path, report: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    encoded = _read_regular(
        path, "learnability policy", maximum_bytes=_MAX_POLICY_BYTES
    )
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnabilityCurriculumError("learnability policy JSON differs") from error
    receipt = source / "stream_receipt.json"
    return (
        validate_policy_payload(
            payload,
            source_stream_identity_sha256=report["ordered_stream_identity_sha256"],
            source_receipt_file_sha256=sha256_file(receipt),
            sequence_count=report["sequences"],
        ),
        encoded,
    )


def _load_scores(
    path: Path, records: _Records, report: dict[str, Any]
) -> tuple[list[dict[str, Any]], bytes]:
    encoded = _read_regular(path, "learnability scores", maximum_bytes=_MAX_SCORE_BYTES)
    try:
        raw_rows = [json.loads(line) for line in encoded.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnabilityCurriculumError("learnability score JSONL differs") from error
    if len(raw_rows) != report["sequences"]:
        raise LearnabilityCurriculumError("learnability score population differs")
    rows = []
    for expected_index, raw in enumerate(raw_rows):
        row = _exact(raw, _SCORE_KEYS, "learnability score row")
        weak = _positive(row["weak_nll_microunits_per_target"], "weak NLL")
        strong = _positive(row["strong_nll_microunits_per_target"], "strong NLL")
        delta = row["preference_delta_microunits"]
        if (
            row["schema"] != SCORE_SCHEMA
            or row["sequence_index"] != expected_index
            or isinstance(delta, bool)
            or not isinstance(delta, int)
            or delta != weak - strong
            or not 0
            < _positive(row["target_count"], "target count")
            <= report["sequence_length"] - 1
        ):
            raise LearnabilityCurriculumError("learnability score value differs")
        record_identity = _record_sha256(*records.record(expected_index)).hex()
        if _sha256(row["record_sha256"], "record identity") != record_identity:
            raise LearnabilityCurriculumError("learnability score record differs")
        rows.append(row)
    return rows, encoded


def _rank_bands(
    rows: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, list[int]]:
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index]["preference_delta_microunits"],
            rows[index]["strong_nll_microunits_per_target"],
            rows[index]["record_sha256"],
        ),
    )
    bands: dict[str, list[int]] = {}
    offset = 0
    for band in BANDS:
        count = policy["band_sequence_counts"][band]
        bands[band] = ranked[offset : offset + count]
        offset += count
    if offset != len(rows):
        raise LearnabilityCurriculumError("learnability band population differs")
    return bands


def _hash_rank(seed: int, label: str, record_sha256: str) -> bytes:
    return hashlib.sha256(f"{seed}:{label}:{record_sha256}".encode()).digest()


def _derive_permutation(
    rows: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[list[int], dict[str, dict[str, int]]]:
    bands = _rank_bands(rows, policy)
    seed = policy["within_phase_order"]["seed"]
    selected: dict[str, list[int]] = {phase: [] for phase in PHASES}
    realized = {phase: {band: 0 for band in BANDS} for phase in PHASES}
    for band in BANDS:
        band_rows = sorted(
            bands[band],
            key=lambda index: _hash_rank(seed, band, rows[index]["record_sha256"]),
        )
        offset = 0
        for phase in PHASES:
            count = policy["phase_by_band_counts"][phase][band]
            selected[phase].extend(band_rows[offset : offset + count])
            realized[phase][band] = count
            offset += count
        if offset != len(band_rows):
            raise LearnabilityCurriculumError("learnability allocation differs")
    permutation = []
    for phase in PHASES:
        phase_rows = sorted(
            selected[phase],
            key=lambda index: _hash_rank(seed, phase, rows[index]["record_sha256"]),
        )
        if len(phase_rows) != policy["phase_sequence_counts"][phase]:
            raise LearnabilityCurriculumError("learnability phase population differs")
        permutation.extend(phase_rows)
    if len(permutation) != len(rows) or len(set(permutation)) != len(rows):
        raise LearnabilityCurriculumError("learnability permutation differs")
    return permutation, realized


def _permutation_sha256(permutation: list[int]) -> str:
    digest = hashlib.sha256()
    for value in permutation:
        digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


def _score_summary(rows: list[dict[str, Any]], indices: list[int]) -> dict[str, int]:
    deltas = [rows[index]["preference_delta_microunits"] for index in indices]
    strong = [rows[index]["strong_nll_microunits_per_target"] for index in indices]
    return {
        "sequences": len(indices),
        "minimum_preference_delta_microunits": min(deltas),
        "maximum_preference_delta_microunits": max(deltas),
        "mean_preference_delta_microunits": sum(deltas) // len(deltas),
        "mean_strong_nll_microunits_per_target": sum(strong) // len(strong),
    }


def _copy_permutation(
    source: Path,
    stage: Path,
    parent: dict[str, Any],
    records: _Records,
    permutation: list[int],
) -> list[dict[str, Any]]:
    shards = []
    token_handle = start_handle = None
    shard_sequences = 0
    shard_index = 0

    def open_shard():
        return (
            (stage / f"shard_{shard_index:05d}.tokens.u32le").open("wb"),
            (stage / f"shard_{shard_index:05d}.starts.bitset").open("wb"),
        )

    def close_shard() -> None:
        nonlocal token_handle, start_handle, shard_sequences, shard_index
        if token_handle is None or start_handle is None:
            return
        token_path = Path(token_handle.name)
        start_path = Path(start_handle.name)
        token_handle.close()
        start_handle.close()
        shards.append(
            {
                "index": shard_index,
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
        shard_index += 1

    try:
        for parent_index in permutation:
            if token_handle is None:
                token_handle, start_handle = open_shard()
            tokens, starts = records.record(parent_index)
            token_handle.write(tokens)
            start_handle.write(starts)
            shard_sequences += 1
            if shard_sequences == parent["sequences_per_shard"]:
                close_shard()
        close_shard()
    except BaseException:
        if token_handle is not None:
            token_handle.close()
        if start_handle is not None:
            start_handle.close()
        raise
    return shards


def _descriptor(path: Path, encoded: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _schedule_descriptor(
    *,
    source: Path,
    parent: dict[str, Any],
    policy_path: Path,
    policy: dict[str, Any],
    policy_bytes: bytes,
    scores_path: Path,
    scores: list[dict[str, Any]],
    scores_bytes: bytes,
    permutation: list[int],
    multiset_sha256: str,
    realized: dict[str, dict[str, int]],
) -> dict[str, Any]:
    phase_summaries = {}
    offset = 0
    for phase in PHASES:
        count = policy["phase_sequence_counts"][phase]
        indices = permutation[offset : offset + count]
        phase_summaries[phase] = {
            **_score_summary(scores, indices),
            "by_band": realized[phase],
        }
        offset += count
    payload = {
        "schema": SCHEDULE_SCHEMA,
        "status": "complete",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "parent_stream": {
            "path": str(source.resolve()),
            "receipt_file_sha256": sha256_file(source / "stream_receipt.json"),
            "ordered_stream_identity_sha256": parent["ordered_stream_identity_sha256"],
        },
        "policy": {
            **_descriptor(policy_path, policy_bytes),
            "receipt_sha256": policy["receipt_sha256"],
        },
        "scores": {
            **_descriptor(scores_path, scores_bytes),
            "ordered_population_sha256": canonical_sha256(scores),
        },
        "permutation_sha256": _permutation_sha256(permutation),
        "sequence_multiset_sha256": multiset_sha256,
        "phases": phase_summaries,
        "same_tokens_and_boundary_masks": True,
        "same_sequence_multiset": True,
        "only_sequence_order_changed": True,
        "semantic_prerequisite_order_proven": False,
        "limitations": [
            "model_centric_learnability_is_not_semantic_prerequisite_evidence",
            "frozen_probe_checkpoints_may_have_checkpoint_specific_preferences",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def build_learnability_curriculum(
    source: Path, scores_path: Path, policy_path: Path, output: Path
) -> dict[str, Any]:
    """Materialize one exact-record curriculum from frozen score evidence."""

    parent = validate_frozen_stream(source, verify_sources=True)
    if output.exists() or output.is_symlink():
        raise LearnabilityCurriculumError("learnability output already exists")
    policy, policy_bytes = load_policy(policy_path, source, parent)
    stage = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    try:
        with _Records(source, parent) as records:
            scores, scores_bytes = _load_scores(scores_path, records, parent)
            permutation, realized = _derive_permutation(scores, policy)
            multiset_sha256 = _multiset_sha256(records, parent["sequences"])
            shards = _copy_permutation(source, stage, parent, records, permutation)
        schedule = _schedule_descriptor(
            source=source,
            parent=parent,
            policy_path=policy_path,
            policy=policy,
            policy_bytes=policy_bytes,
            scores_path=scores_path,
            scores=scores,
            scores_bytes=scores_bytes,
            permutation=permutation,
            multiset_sha256=multiset_sha256,
            realized=realized,
        )
        report = {
            **{
                key: value
                for key, value in parent.items()
                if key
                not in {
                    "ordered_stream_identity_sha256",
                    "prefix_utf8_bytes",
                    "shards",
                    "curriculum",
                    "ordering_control",
                    "learnability_curriculum",
                }
            },
            "schema": TOKEN_STREAM_SCHEMA,
            "prefix_utf8_bytes": {
                str(parent["sequences"]): parent["admitted_utf8_bytes"]
            },
            "shards": shards,
            "learnability_curriculum": schedule,
        }
        report["ordered_stream_identity_sha256"] = canonical_sha256(report)
        (stage / "stream_receipt.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        _validate_learnability_curriculum(
            stage,
            source=source,
            scores_path=scores_path,
            policy_path=policy_path,
        )
        os.replace(stage, output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_learnability_curriculum(
    output: Path, *, source: Path, scores_path: Path, policy_path: Path
) -> dict[str, Any]:
    report = validate_frozen_stream(output, verify_sources=True)
    parent = validate_frozen_stream(source, verify_sources=True)
    policy, policy_bytes = load_policy(policy_path, source, parent)
    with _Records(source, parent) as parent_records:
        scores, scores_bytes = _load_scores(scores_path, parent_records, parent)
        permutation, realized = _derive_permutation(scores, policy)
        multiset_sha256 = _multiset_sha256(parent_records, parent["sequences"])
        with _Records(output, report) as output_records:
            if _multiset_sha256(output_records, report["sequences"]) != multiset_sha256:
                raise LearnabilityCurriculumError("learnability multiset differs")
            for output_index, parent_index in enumerate(permutation):
                if output_records.record(output_index) != parent_records.record(
                    parent_index
                ):
                    raise LearnabilityCurriculumError(
                        "learnability permutation differs"
                    )
    expected = _schedule_descriptor(
        source=source,
        parent=parent,
        policy_path=policy_path,
        policy=policy,
        policy_bytes=policy_bytes,
        scores_path=scores_path,
        scores=scores,
        scores_bytes=scores_bytes,
        permutation=permutation,
        multiset_sha256=multiset_sha256,
        realized=realized,
    )
    if (
        report.get("learnability_curriculum") != expected
        or report["sequences"] != parent["sequences"]
        or report["admitted_utf8_bytes"] != parent["admitted_utf8_bytes"]
        or report["tokenizer_identity_sha256"] != parent["tokenizer_identity_sha256"]
        or report["source_receipts"] != parent["source_receipts"]
        or report["prefix_utf8_bytes"]
        != {str(parent["sequences"]): parent["admitted_utf8_bytes"]}
    ):
        raise LearnabilityCurriculumError("learnability receipt differs")
    return report


def validate_learnability_curriculum(
    output: Path, *, source: Path, scores_path: Path, policy_path: Path
) -> dict[str, Any]:
    """Replay scores, policy, source records, and the full output permutation."""

    return _validate_learnability_curriculum(
        output,
        source=source,
        scores_path=scores_path,
        policy_path=policy_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--scores", type=Path, required=True)
        child.add_argument("--policy", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        payload = build_learnability_curriculum(
            args.source, args.scores, args.policy, args.output
        )
    else:
        payload = validate_learnability_curriculum(
            args.output,
            source=args.source,
            scores_path=args.scores,
            policy_path=args.policy,
        )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "ordered_stream_identity_sha256": payload[
                    "ordered_stream_identity_sha256"
                ],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
