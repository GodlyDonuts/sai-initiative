"""Validate direct or curriculum-derived benchmark-disjoint training lineage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file


class TrainingSourceLineageError(RuntimeError):
    """The training source does not descend from the audited source population."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingSourceLineageError(message)


def _load_receipt(path: Path) -> tuple[dict[str, Any], str]:
    _require(path.is_absolute(), "receipt path must be absolute")
    _require(path.is_file() and not path.is_symlink(), "receipt is missing or unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingSourceLineageError("receipt is unreadable") from error
    _require(isinstance(payload, dict), "receipt payload must be an object")
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    _require(claimed == canonical_sha256(unsigned), "receipt self hash differs")
    return payload, claimed


def validate_training_source_lineage(
    *,
    stream_source: dict[str, Any],
    benchmark_training_source_sha256: str,
    split_receipt: Path | None = None,
    split_receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind a frozen stream to the source used by benchmark decontamination.

    The direct path preserves the original contract. The derived path proves the
    exact chain: decontaminated source -> qualified curriculum -> qualified
    train/development split -> frozen train stream.
    """

    _require(
        set(stream_source) == {"order", "path", "bytes", "sha256"},
        "stream source schema differs",
    )
    _require(stream_source["order"] == 0, "stream source order differs")
    _require(
        isinstance(benchmark_training_source_sha256, str)
        and len(benchmark_training_source_sha256) == 64,
        "benchmark training source SHA256 differs",
    )
    if split_receipt is None:
        _require(
            split_receipt_file_sha256 is None,
            "split receipt SHA256 was supplied without a receipt",
        )
        _require(
            stream_source["sha256"] == benchmark_training_source_sha256,
            "direct stream source differs from benchmark training source",
        )
        lineage = {
            "method": "direct_decontaminated_source",
            "training_source_sha256": benchmark_training_source_sha256,
            "stream_source_sha256": stream_source["sha256"],
        }
    else:
        _require(
            isinstance(split_receipt_file_sha256, str)
            and len(split_receipt_file_sha256) == 64,
            "split receipt file SHA256 is missing",
        )
        _require(
            sha256_file(split_receipt) == split_receipt_file_sha256,
            "split receipt file SHA256 differs",
        )
        split, split_claimed = _load_receipt(split_receipt)
        _require(
            split.get("schema") == "sai-curriculum-train-development-split-v1",
            "split receipt schema differs",
        )
        _require(
            split.get("status") == "qualified" and split.get("split_qualified") is True,
            "split receipt is not qualified",
        )
        _require(
            split.get("checks")
            == {
                "all_curriculum_documents_emitted_once": True,
                "both_populations_have_every_phase": True,
                "exact_identity_assignment_disjoint": True,
                "train_progression_qualified": True,
            },
            "split qualification checks differ",
        )
        train = split.get("train")
        _require(isinstance(train, dict), "split train output is missing")
        train_path = Path(train.get("path", ""))
        _require(
            train_path.is_absolute()
            and train_path.is_file()
            and not train_path.is_symlink(),
            "split train output is missing or unsafe",
        )
        _require(
            stream_source["path"] == str(train_path.resolve())
            and stream_source["bytes"] == train.get("bytes")
            and stream_source["sha256"] == train.get("sha256"),
            "stream source differs from split train output",
        )
        source_curriculum = split.get("source_curriculum")
        _require(
            isinstance(source_curriculum, dict),
            "split source curriculum is missing",
        )
        curriculum_receipt_path = Path(source_curriculum.get("receipt_path", ""))
        _require(
            curriculum_receipt_path.is_absolute()
            and curriculum_receipt_path.is_file()
            and not curriculum_receipt_path.is_symlink(),
            "curriculum receipt is missing or unsafe",
        )
        _require(
            curriculum_receipt_path.stat().st_size
            == source_curriculum.get("receipt_bytes")
            and sha256_file(curriculum_receipt_path)
            == source_curriculum.get("receipt_file_sha256"),
            "curriculum receipt file differs",
        )
        curriculum, curriculum_claimed = _load_receipt(curriculum_receipt_path)
        _require(
            curriculum_claimed == source_curriculum.get("receipt_sha256"),
            "curriculum receipt identity differs",
        )
        _require(
            curriculum.get("schema") == "sai-curriculum-order-receipt-v1"
            and curriculum.get("status") == "qualified"
            and curriculum.get("curriculum_qualified") is True,
            "curriculum is not qualified",
        )
        _require(
            curriculum.get("source", {}).get("sha256")
            == benchmark_training_source_sha256,
            "curriculum parent differs from benchmark training source",
        )
        _require(
            curriculum.get("output", {}).get("sha256")
            == source_curriculum.get("output_sha256"),
            "curriculum output identity differs",
        )
        lineage = {
            "method": "qualified_curriculum_train_split",
            "decontaminated_training_source_sha256": (benchmark_training_source_sha256),
            "curriculum_receipt_sha256": curriculum_claimed,
            "curriculum_output_sha256": curriculum["output"]["sha256"],
            "split_receipt_file_sha256": split_receipt_file_sha256,
            "split_receipt_sha256": split_claimed,
            "stream_source_sha256": stream_source["sha256"],
        }
    return {
        "lineage": lineage,
        "source_lineage_sha256": canonical_sha256(lineage),
    }
