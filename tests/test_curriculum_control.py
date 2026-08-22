from pathlib import Path

import pytest

from sai.data.curriculum_control import (
    FROZEN_SEED,
    CurriculumControlError,
    build_order_control,
    validate_order_control,
)
from sai.data.token_stream import freeze
from tests.test_token_stream import CharacterTokenizer, source


def test_order_control_preserves_exact_records_and_changes_only_order(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent_report = freeze(
        CharacterTokenizer(),
        [source(tmp_path)],
        parent,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={1, 2, 3},
        sequences_per_shard=2,
        source_qualification_sha256="2" * 64,
    )
    control = tmp_path / "control"
    report = build_order_control(parent, control)

    assert report["sequences"] == parent_report["sequences"]
    assert report["admitted_utf8_bytes"] == parent_report["admitted_utf8_bytes"]
    assert report["source_qualification_sha256"] == "2" * 64
    assert report["prefix_utf8_bytes"] == {"3": parent_report["admitted_utf8_bytes"]}
    assert report["ordering_control"]["same_sequence_multiset"] is True
    assert report["ordering_control"]["only_sequence_order_changed"] is True
    assert (
        report["ordered_stream_identity_sha256"]
        != parent_report["ordered_stream_identity_sha256"]
    )
    assert validate_order_control(control) == report


def test_order_control_is_create_only_and_tamper_fails(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    freeze(
        CharacterTokenizer(),
        [source(tmp_path)],
        parent,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={3},
        sequences_per_shard=2,
    )
    control = tmp_path / "control"
    build_order_control(parent, control, seed=FROZEN_SEED)
    with pytest.raises(CurriculumControlError, match="already exists"):
        build_order_control(parent, control)
    token_file = next(control.glob("*.tokens.u32le"))
    payload = bytearray(token_file.read_bytes())
    payload[0] ^= 1
    token_file.write_bytes(payload)
    with pytest.raises(Exception, match="content differs|multiset differs"):
        validate_order_control(control)


def test_order_control_refuses_posthoc_seed(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    freeze(
        CharacterTokenizer(),
        [source(tmp_path)],
        parent,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={3},
        sequences_per_shard=2,
    )
    with pytest.raises(CurriculumControlError, match="seed differs"):
        build_order_control(parent, tmp_path / "control", seed=FROZEN_SEED + 1)
