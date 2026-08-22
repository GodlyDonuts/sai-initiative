from pathlib import Path

import pytest

from sai.data.curriculum_control import (
    FROZEN_SEED,
    CurriculumControlError,
    build_order_control,
    validate_curriculum_order_bundle,
    validate_order_control,
)
from sai.data.token_stream import freeze, sha256_file
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


def test_bundle_reuses_replayed_split_without_weakening_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_source = source(tmp_path)
    development_source = tmp_path / "development.jsonl"
    development_source.write_bytes(train_source.read_bytes())
    split_receipt = tmp_path / "split.json"
    split_receipt.write_text('{"qualified":true}\n')
    qualification = sha256_file(split_receipt)
    parent = tmp_path / "parent"
    parent_report = freeze(
        CharacterTokenizer(),
        [train_source],
        parent,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={1, 2, 3},
        sequences_per_shard=2,
        source_qualification_sha256=qualification,
    )
    development = tmp_path / "development"
    development_report = freeze(
        CharacterTokenizer(),
        [development_source],
        development,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={1, 2, 3},
        sequences_per_shard=2,
        source_qualification_sha256=qualification,
    )
    control = tmp_path / "control"
    control_report = build_order_control(parent, control)
    split = {
        "train": {
            "path": str(train_source.resolve()),
            "bytes": train_source.stat().st_size,
            "sha256": sha256_file(train_source),
        },
        "development": {
            "path": str(development_source.resolve()),
            "bytes": development_source.stat().st_size,
            "sha256": sha256_file(development_source),
        },
    }
    monkeypatch.setattr(
        "sai.data.curriculum_split.validate_curriculum_split",
        lambda *_args, **_kwargs: split,
    )

    bundle = validate_curriculum_order_bundle(
        parent,
        control,
        development,
        split_receipt,
    )

    assert bundle == {
        "split": split,
        "curriculum": parent_report,
        "control": control_report,
        "development": development_report,
    }

    split["train"] = {**split["train"], "sha256": "f" * 64}
    with pytest.raises(CurriculumControlError, match="qualified split source"):
        validate_curriculum_order_bundle(
            parent,
            control,
            development,
            split_receipt,
        )
