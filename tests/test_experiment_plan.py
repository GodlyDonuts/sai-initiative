from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sai.training.experiment import (
    ExperimentPlanError,
    build_plan,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "SAI_100M_TOURNAMENT_TEMPLATE.json"
ARCHITECTURE = ROOT / "docs" / "SAI_FRONTIER_ARCHITECTURE_TOURNAMENT.json"
GEOMETRY = ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json"
ISO_DATA_SEQUENCES = 1_048_576
ISO_FLOP_SEQUENCES = {
    "gated_gqa": 678_678,
    "gdn_hybrid": 794_277,
    "kda_mla_hybrid": 797_472,
}
COMMON_FLOPS = 1_047_724_302_079_623_168


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    tokenizer_identity = "1" * 64
    tokenizer = write_json(
        tmp_path / "tokenizer.json",
        {
            "schema": "sai-tokenizer-qualification-receipt-v1",
            "status": "qualified",
            "training_authorized": False,
            "vocab_size": 48_000,
            "tokenizer_identity_sha256": tokenizer_identity,
            "corpus_identity_sha256": "2" * 64,
            "byte_fallback": True,
            "roundtrip_failures": 0,
            "special_tokens_preserved": True,
        },
    )
    prefix_counts = {ISO_DATA_SEQUENCES, *ISO_FLOP_SEQUENCES.values()}
    stream = write_json(
        tmp_path / "stream.json",
        {
            "schema": "sai-ordered-token-stream-receipt-v1",
            "status": "complete",
            "training_authorized": False,
            "tokenizer_identity_sha256": tokenizer_identity,
            "ordered_stream_identity_sha256": "3" * 64,
            "source_manifest_sha256": "4" * 64,
            "sequence_length": 2_048,
            "sequences": ISO_DATA_SEQUENCES,
            "valid_tokens": ISO_DATA_SEQUENCES * 2_048,
            "admitted_utf8_bytes": ISO_DATA_SEQUENCES * 1_000,
            "prefix_utf8_bytes": {
                str(count): count * 1_000 for count in sorted(prefix_counts)
            },
            "benchmark_disjoint": True,
            "cross_document_targets_masked": True,
        },
    )
    environment = write_json(
        tmp_path / "environment.json",
        {
            "schema": "sai-training-environment-receipt-v1",
            "status": "complete",
            "training_authorized": False,
            "environment_identity_sha256": "5" * 64,
            "versions": {
                "python": "3.13.7",
                "torch": "2.8.0+cu128",
                "cuda": "12.8",
                "triton": "3.4.0",
            },
        },
    )
    return tokenizer, stream, environment


def plan(tmp_path: Path) -> dict:
    tokenizer, stream, environment = inputs(tmp_path)
    return build_plan(
        TEMPLATE,
        ARCHITECTURE,
        GEOMETRY,
        tokenizer,
        stream,
        environment,
    )


def test_plan_builds_exact_three_family_three_seed_dual_contrast_grid(
    tmp_path: Path,
) -> None:
    payload = plan(tmp_path)
    assert validate_plan(payload) == payload
    assert len(payload["runs"]) == 18
    identities = {row["run_identity_sha256"] for row in payload["runs"]}
    assert len(identities) == 18
    assert not payload["training_authorized"]
    assert payload["gpu_jobs_submitted"] == 0
    assert payload["training_updates_completed"] == 0


def test_iso_data_and_iso_flop_are_separately_and_exactly_matched(
    tmp_path: Path,
) -> None:
    payload = plan(tmp_path)
    iso_data = [row for row in payload["runs"] if row["contrast"] == "iso_data"]
    assert {row["prefix_sequences"] for row in iso_data} == {ISO_DATA_SEQUENCES}
    assert len({row["prefix_utf8_bytes"] for row in iso_data}) == 1

    iso_flop = [row for row in payload["runs"] if row["contrast"] == "iso_flop"]
    assert {row["modeled_training_flops"] for row in iso_flop} == {COMMON_FLOPS}
    assert {
        row["mixer_family"]: row["prefix_sequences"]
        for row in iso_flop
        if row["seed"] == 20260821
    } == ISO_FLOP_SEQUENCES
    assert payload["budget_geometry"]["exact_iso_flop_common_budget"] == COMMON_FLOPS


def test_plan_is_deterministic(tmp_path: Path) -> None:
    tokenizer, stream, environment = inputs(tmp_path)
    first = build_plan(TEMPLATE, ARCHITECTURE, GEOMETRY, tokenizer, stream, environment)
    second = build_plan(
        TEMPLATE, ARCHITECTURE, GEOMETRY, tokenizer, stream, environment
    )
    assert first == second


def test_missing_exact_stream_prefix_fails_closed(tmp_path: Path) -> None:
    tokenizer, stream, environment = inputs(tmp_path)
    payload = json.loads(stream.read_text())
    del payload["prefix_utf8_bytes"][str(ISO_FLOP_SEQUENCES["gated_gqa"])]
    write_json(stream, payload)
    with pytest.raises(ExperimentPlanError, match="lacks required prefixes"):
        build_plan(TEMPLATE, ARCHITECTURE, GEOMETRY, tokenizer, stream, environment)


def test_tokenizer_stream_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    tokenizer, stream, environment = inputs(tmp_path)
    payload = json.loads(stream.read_text())
    payload["tokenizer_identity_sha256"] = "9" * 64
    write_json(stream, payload)
    with pytest.raises(ExperimentPlanError, match="ordered token stream differs"):
        build_plan(TEMPLATE, ARCHITECTURE, GEOMETRY, tokenizer, stream, environment)


def test_template_hyperparameter_drift_fails_closed(tmp_path: Path) -> None:
    tokenizer, stream, environment = inputs(tmp_path)
    template = tmp_path / "template.json"
    payload = json.loads(TEMPLATE.read_text())
    payload["optimizer"]["learning_rate"] = 0.001
    write_json(template, payload)
    with pytest.raises(ExperimentPlanError, match="template differs"):
        build_plan(template, ARCHITECTURE, GEOMETRY, tokenizer, stream, environment)


def test_plan_tampering_or_input_mutation_fails_closed(tmp_path: Path) -> None:
    payload = plan(tmp_path)
    tampered = copy.deepcopy(payload)
    tampered["runs"][0]["prefix_sequences"] -= 1
    with pytest.raises(ExperimentPlanError, match="plan or bound inputs differ"):
        validate_plan(tampered)

    environment_path = Path(payload["inputs"]["environment_receipt"]["path"])
    environment_path.write_text("{}\n")
    with pytest.raises(ExperimentPlanError):
        validate_plan(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_hold", False),
        ("training_authorized", True),
        ("official_training_order_received", True),
        ("gpu_jobs_submitted", 1),
        ("training_updates_completed", 1),
    ],
)
def test_any_execution_or_authorization_claim_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = plan(tmp_path)
    payload[field] = value
    with pytest.raises(ExperimentPlanError, match="no-training boundary"):
        validate_plan(payload)
