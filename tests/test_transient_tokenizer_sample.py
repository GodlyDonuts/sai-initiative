import hashlib
import io
import json
from pathlib import Path

import pytest

from sai.data.pleias_virtual_transient_stream import (
    ENVELOPE_SCHEMA,
)
from sai.data.pleias_virtual_transient_stream import (
    RECEIPT_SCHEMA as SOURCE_RECEIPT_SCHEMA,
)
from sai.data.pleias_virtual_transient_stream import (
    STATUS as SOURCE_STATUS,
)
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, normalize_document
from sai.data.transient_tokenizer_sample import (
    SAMPLE_NAME,
    TransientTokenizerSampleError,
    build_sample,
)


def _envelope(index: int, domain: str, phase: str, split: str = "train") -> dict:
    text = f"Verified lesson {index} explains evidence and prerequisite reasoning " * 20
    document = normalize_document(
        {
            "schema": ROW_SCHEMA,
            "text": text,
            "source": {
                "dataset": "PleIAs/common_corpus@revision",
                "row_id": f"parent.parquet#{index}",
                "license": "Public Domain",
                "domain": "english",
            },
            "verification": {
                "benchmark_disjoint": True,
                "evidence_sha256": f"{index + 1:064x}",
            },
        }
    )
    value = {
        "schema": ENVELOPE_SCHEMA,
        "document": document,
        "corpus_split": split,
        "semantic_curriculum_phase": phase,
        "semantic_difficulty_mean_milli": 2_000,
        "semantic_prerequisite_burden_mean_milli": 1_000,
        "semantic_quality_floor_milli": 8_000 + index,
        "semantic_domains": [domain],
        "semantic_recurring_concepts": ["evidence"],
        "semantic_recurring_prerequisites": ["language"],
        "final_locator_sha256": f"{index + 100:064x}",
        "tokenization_ready": True,
        "training_ready": False,
    }
    value["envelope_sha256"] = canonical_sha256(value)
    return value


def _source(tmp_path: Path, envelopes: list[dict]) -> tuple[io.StringIO, Path]:
    payload = "".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        for value in envelopes
    )
    receipt = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "status": SOURCE_STATUS,
        "counts": {"documents": len(envelopes)},
        "ordered_jsonl_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_text_persisted_by_compiler": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    path = tmp_path / "source-receipt.json"
    path.write_text(json.dumps(receipt))
    return io.StringIO(payload), path


def test_sample_is_bounded_broad_and_excludes_development(tmp_path: Path) -> None:
    envelopes = [
        *[_envelope(i, "mathematics", "foundation") for i in range(6)],
        *[_envelope(i + 6, "literature", "synthesis") for i in range(6)],
        _envelope(12, "science", "depth", "development"),
    ]
    source, receipt = _source(tmp_path, envelopes)
    output = tmp_path / "sample"
    result = build_sample(
        source,
        receipt,
        output,
        maximum_utf8_bytes=1_000_000,
    )
    assert result["sample"]["bytes"] <= 1_000_000
    assert result["selection"]["strata"] == 2
    assert result["input_counts"]["development_documents_excluded"] == 1
    rows = [
        json.loads(line) for line in (output / SAMPLE_NAME).read_text().splitlines()
    ]
    assert rows
    assert all(row["source"]["row_id"] != "parent.parquet#12" for row in rows)
    assert all(row["verification"]["benchmark_disjoint"] for row in rows)
    assert not any(
        "Verified lesson" in line
        for line in (output / "receipt.json").read_text().splitlines()
    )


def test_sample_rejects_stream_receipt_mismatch(tmp_path: Path) -> None:
    source, receipt = _source(tmp_path, [_envelope(0, "mathematics", "foundation")])
    value = json.loads(receipt.read_text())
    value["ordered_jsonl_sha256"] = "f" * 64
    value.pop("receipt_sha256")
    value["receipt_sha256"] = canonical_sha256(value)
    receipt.write_text(json.dumps(value))
    with pytest.raises(
        TransientTokenizerSampleError, match="transient source receipt differs"
    ):
        build_sample(
            source,
            receipt,
            tmp_path / "sample",
            maximum_utf8_bytes=1_000_000,
        )


def test_sample_rejects_mutated_envelope(tmp_path: Path) -> None:
    envelope = _envelope(0, "mathematics", "foundation")
    envelope["semantic_domains"] = ["tampered"]
    source, receipt = _source(tmp_path, [envelope])
    with pytest.raises(
        TransientTokenizerSampleError, match="transient envelope differs"
    ):
        build_sample(
            source,
            receipt,
            tmp_path / "sample",
            maximum_utf8_bytes=1_000_000,
        )
