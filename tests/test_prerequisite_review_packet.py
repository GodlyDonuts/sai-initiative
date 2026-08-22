from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sai.data.prerequisite_review_packet as packet
from sai.data.prerequisite_review_packet import (
    BLIND_ANNOTATION_SCHEMA,
    PrerequisiteReviewPacketError,
    build_packet,
    compile_annotations,
    validate_compilation,
    validate_packet,
)
from sai.data.token_stream import canonical_sha256


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    phases = ("grounding", "integration", "reasoning", "specialization")
    bands = ("foundation", "composition", "reasoning", "specialization")
    cells = [
        (phase, band)
        for phase in phases
        for band in bands
        if (phase, band) != ("grounding", "specialization")
    ]
    rows = []
    for cell_index, (phase, band) in enumerate(cells):
        for rank in range(8):
            index = cell_index * 8 + rank
            text = f"lesson {index} about colors and composition"
            identity = hashlib.sha256(f"document:{index}".encode()).hexdigest()
            rows.append(
                {
                    "schema": "sai-semantic-prerequisite-audit-document-v1",
                    "document_identity_sha256": identity,
                    "document_index": index,
                    "phase": phase,
                    "surface_band": band,
                    "source": {
                        "dataset": "fixture",
                        "domain": "english",
                        "license": "fixture",
                        "row_id": str(index),
                    },
                    "selection_rank_sha256": hashlib.sha256(
                        f"rank:{index}".encode()
                    ).hexdigest(),
                    "text": text,
                }
            )
    output = tmp_path / "population.jsonl"
    encoded = _jsonl(rows)
    output.write_bytes(encoded)
    receipt = tmp_path / "population.receipt.json"
    receipt.write_text("{}\n")
    payload = {
        "receipt_sha256": "1" * 64,
        "selection": {
            "per_stratum": 8,
            "strata": 15,
            "selected_documents": 120,
            "excluded_structurally_empty_strata": ["grounding:specialization"],
        },
        "output": {
            "path": str(output),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_population_sha256": canonical_sha256(rows),
        },
    }
    monkeypatch.setattr(
        packet, "_validate_population", lambda *_args, **_kwargs: payload
    )
    return receipt


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "blind.jsonl",
        tmp_path / "hidden-key.jsonl",
        tmp_path / "packet.receipt.json",
    )


def _blind_annotations(
    tmp_path: Path, review: Path, *, corrupt_span: bool = False
) -> tuple[Path, Path]:
    concept_list = tmp_path / "concepts.json"
    concept_list.write_text(
        json.dumps(
            {
                "schema": "sai-semantic-prerequisite-concept-list-v1",
                "status": "candidate",
                "concepts": [{"concept_id": "color_composition"}],
            },
            sort_keys=True,
        )
        + "\n"
    )
    rows = []
    for review_row in map(json.loads, review.read_text().splitlines()):
        text = review_row["text"]
        start = text.index("colors")
        end = start + len("colors")
        text_sha256 = hashlib.sha256(text[start:end].encode()).hexdigest()
        if corrupt_span:
            text_sha256 = "f" * 64
        rows.append(
            {
                "schema": BLIND_ANNOTATION_SCHEMA,
                "review_identity_sha256": review_row["review_identity_sha256"],
                "concepts": [
                    {
                        "concept_id": "color_composition",
                        "confidence_ppm": 900_000,
                        "evidence_spans": [
                            {"start": start, "end": end, "text_sha256": text_sha256}
                        ],
                    }
                ],
            }
        )
    blind = tmp_path / "blind-annotations.jsonl"
    blind.write_bytes(_jsonl(rows))
    return blind, concept_list


def test_packet_hides_curriculum_labels_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population_receipt = _population(tmp_path, monkeypatch)
    review, key, receipt = _paths(tmp_path)
    payload = build_packet(population_receipt, review, key, receipt)

    review_rows = [json.loads(line) for line in review.read_text().splitlines()]
    key_rows = [json.loads(line) for line in key.read_text().splitlines()]
    assert payload["selection"] == {
        "rows": 120,
        "ordering": "ascending_sha256_of_salt_plus_document_identity",
        "review_hides_phase_surface_band_source_and_document_order": True,
        "hidden_key_is_separate": True,
    }
    assert len(review_rows) == len(key_rows) == 120
    assert [row["review_identity_sha256"] for row in review_rows] == sorted(
        row["review_identity_sha256"] for row in review_rows
    )
    assert all(
        not ({"phase", "surface_band", "source", "document_index"} & set(row))
        for row in review_rows
    )
    assert all(
        {"phase", "surface_band", "source", "document_index"}.issubset(row)
        for row in key_rows
    )
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate_packet(population_receipt, review, key, receipt) == payload


def test_packet_rejects_tamper_and_resigned_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population_receipt = _population(tmp_path, monkeypatch)
    review, key, receipt = _paths(tmp_path)
    build_packet(population_receipt, review, key, receipt)

    review.chmod(0o600)
    review.write_bytes(review.read_bytes().replace(b"lesson", b"poison", 1))
    with pytest.raises(PrerequisiteReviewPacketError):
        validate_packet(population_receipt, review, key, receipt)

    review.unlink()
    key.unlink()
    receipt.unlink()
    payload = build_packet(population_receipt, review, key, receipt)
    receipt.chmod(0o600)
    payload["selection"]["rows"] = 119
    unsigned = {
        name: value for name, value in payload.items() if name != "receipt_sha256"
    }
    payload["receipt_sha256"] = canonical_sha256(unsigned)
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(PrerequisiteReviewPacketError):
        validate_packet(population_receipt, review, key, receipt)


def test_packet_is_create_only_and_population_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population_receipt = _population(tmp_path, monkeypatch)
    review, key, receipt = _paths(tmp_path)
    build_packet(population_receipt, review, key, receipt)
    with pytest.raises(PrerequisiteReviewPacketError, match="boundary"):
        build_packet(population_receipt, review, key, receipt)

    population = tmp_path / "population.jsonl"
    population.write_bytes(population.read_bytes().replace(b"colors", b"shapes", 1))
    with pytest.raises(PrerequisiteReviewPacketError):
        validate_packet(population_receipt, review, key, receipt)


def test_blind_annotations_compile_to_canonical_population_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population_receipt = _population(tmp_path, monkeypatch)
    review, key, packet_receipt = _paths(tmp_path)
    build_packet(population_receipt, review, key, packet_receipt)
    blind, concepts = _blind_annotations(tmp_path, review)
    output = tmp_path / "canonical-annotations.jsonl"
    compilation_receipt = tmp_path / "compilation.receipt.json"

    payload = compile_annotations(
        population_receipt=population_receipt,
        review_output=review,
        key_output=key,
        packet_receipt=packet_receipt,
        blind_annotations=blind,
        concept_list=concepts,
        output=output,
        compilation_receipt=compilation_receipt,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    population_rows = [
        json.loads(line)
        for line in (tmp_path / "population.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 120
    assert [row["document_identity_sha256"] for row in rows] == [
        row["document_identity_sha256"] for row in population_rows
    ]
    assert [row["phase"] for row in rows] == [row["phase"] for row in population_rows]
    assert all("review_identity_sha256" not in row for row in rows)
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert (
        validate_compilation(
            population_receipt=population_receipt,
            review_output=review,
            key_output=key,
            packet_receipt=packet_receipt,
            blind_annotations=blind,
            concept_list=concepts,
            output=output,
            compilation_receipt=compilation_receipt,
        )
        == payload
    )


def test_compilation_rejects_bad_evidence_and_blind_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population_receipt = _population(tmp_path, monkeypatch)
    review, key, packet_receipt = _paths(tmp_path)
    build_packet(population_receipt, review, key, packet_receipt)
    blind, concepts = _blind_annotations(tmp_path, review, corrupt_span=True)
    output = tmp_path / "canonical-annotations.jsonl"
    compilation_receipt = tmp_path / "compilation.receipt.json"
    with pytest.raises(PrerequisiteReviewPacketError, match="evidence"):
        compile_annotations(
            population_receipt=population_receipt,
            review_output=review,
            key_output=key,
            packet_receipt=packet_receipt,
            blind_annotations=blind,
            concept_list=concepts,
            output=output,
            compilation_receipt=compilation_receipt,
        )
    assert not output.exists() and not compilation_receipt.exists()

    blind, concepts = _blind_annotations(tmp_path, review)
    rows = [json.loads(line) for line in blind.read_text().splitlines()]
    rows[0]["review_identity_sha256"] = "a" * 64
    blind.write_bytes(_jsonl(rows))
    with pytest.raises(PrerequisiteReviewPacketError, match="annotations"):
        compile_annotations(
            population_receipt=population_receipt,
            review_output=review,
            key_output=key,
            packet_receipt=packet_receipt,
            blind_annotations=blind,
            concept_list=concepts,
            output=output,
            compilation_receipt=compilation_receipt,
        )
