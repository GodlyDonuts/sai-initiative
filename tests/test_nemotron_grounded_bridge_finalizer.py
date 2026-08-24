import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sai.data.nemotron_grounded_bridge_finalizer import (
    NemotronGroundedBridgeFinalizerError,
    finalize,
)
from sai.data.token_stream import canonical_sha256


def _write_receipt(root: Path, schema: str, status: str, value: str) -> None:
    root.mkdir()
    receipt = {"schema": schema, "status": status, "value": value}
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt))


def test_finalizer_creates_both_outputs_under_one_lock(tmp_path: Path) -> None:
    aggregate_root = tmp_path / "aggregate"
    decontamination_root = tmp_path / "decontamination"

    def aggregate(*args, **kwargs):
        aggregate_root.mkdir()
        (aggregate_root / "receipt.json").write_text("{}")
        return {"receipt_sha256": "a" * 64}

    def decontaminate(*args, **kwargs):
        assert aggregate_root.is_dir()
        decontamination_root.mkdir()
        (decontamination_root / "receipt.json").write_text("{}")
        return {"receipt_sha256": "b" * 64}

    with (
        patch(
            "sai.data.nemotron_grounded_bridge_finalizer.build_aggregate",
            side_effect=aggregate,
        ) as build_aggregate,
        patch(
            "sai.data.nemotron_grounded_bridge_finalizer.build_screen",
            side_effect=decontaminate,
        ) as build_screen,
    ):
        result = finalize(
            tmp_path / "population",
            tmp_path / "same-family",
            tmp_path / "judgments",
            aggregate_root,
            [tmp_path / "boundary"],
            decontamination_root,
            tmp_path / "finalizer.lock",
            logical_shards=64,
        )

    assert result["aggregate_created"] is True
    assert result["decontamination_created"] is True
    assert result["serialized_by_process_lock"] is True
    build_aggregate.assert_called_once()
    build_screen.assert_called_once()


def test_finalizer_rejects_unsealed_existing_output(tmp_path: Path) -> None:
    aggregate_root = tmp_path / "aggregate"
    aggregate_root.mkdir()
    with pytest.raises(
        NemotronGroundedBridgeFinalizerError,
        match="existing output lacks a sealed receipt",
    ):
        finalize(
            tmp_path / "population",
            tmp_path / "same-family",
            tmp_path / "judgments",
            aggregate_root,
            [tmp_path / "boundary"],
            tmp_path / "decontamination",
            tmp_path / "finalizer.lock",
            logical_shards=64,
        )


def test_finalizer_reuses_only_schema_valid_sealed_outputs(tmp_path: Path) -> None:
    aggregate_root = tmp_path / "aggregate"
    decontamination_root = tmp_path / "decontamination"
    _write_receipt(
        aggregate_root,
        "sai-grounded-cross-domain-independent-model-family-bridge-"
        "verification-aggregate-v1",
        "complete_independent_model_family_bridge_verification_routes",
        "aggregate",
    )
    _write_receipt(
        decontamination_root,
        "sai-grounded-bridge-decontamination-v1",
        "complete_post_generation_bridge_benchmark_screen",
        "decontamination",
    )

    with (
        patch(
            "sai.data.nemotron_grounded_bridge_finalizer.build_aggregate"
        ) as build_aggregate,
        patch(
            "sai.data.nemotron_grounded_bridge_finalizer.build_screen"
        ) as build_screen,
    ):
        result = finalize(
            tmp_path / "population",
            tmp_path / "same-family",
            tmp_path / "judgments",
            aggregate_root,
            [tmp_path / "boundary"],
            decontamination_root,
            tmp_path / "finalizer.lock",
            logical_shards=64,
        )

    assert result["aggregate_created"] is False
    assert result["decontamination_created"] is False
    build_aggregate.assert_not_called()
    build_screen.assert_not_called()
