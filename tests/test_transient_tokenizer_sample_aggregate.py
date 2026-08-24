import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import (
    ROW_SCHEMA,
    canonical_sha256,
    normalize_document,
    sha256_file,
)
from sai.data.transient_tokenizer_sample import SAMPLE_NAME
from sai.data.transient_tokenizer_sample import SCHEMA as SHARD_SCHEMA
from sai.data.transient_tokenizer_sample import STATUS as SHARD_STATUS
from sai.data.transient_tokenizer_sample_aggregate import (
    TransientTokenizerSampleAggregateError,
    aggregate,
)


def _document(index: int, text: str | None = None) -> dict:
    return normalize_document(
        {
            "schema": ROW_SCHEMA,
            "text": text or (f"Verified broad tokenizer lesson {index}. " * 30),
            "source": {
                "dataset": "PleIAs/common_corpus@revision",
                "row_id": f"parent-{index}.parquet#0",
                "license": "Public Domain",
                "domain": "math" if index % 2 else "english",
            },
            "verification": {
                "benchmark_disjoint": True,
                "evidence_sha256": f"{index + 1:064x}",
            },
        }
    )


def _shard(root: Path, index: int, documents: list[dict]) -> None:
    shard = root / f"shard_{index:05d}"
    shard.mkdir(parents=True)
    sample = shard / SAMPLE_NAME
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in documents
    )
    sample.write_text(payload)
    receipt = {
        "schema": SHARD_SCHEMA,
        "status": SHARD_STATUS,
        "source_receipt_sha256": f"{index + 100:064x}",
        "sample": {
            "path": SAMPLE_NAME,
            "documents": len(documents),
            "bytes": sample.stat().st_size,
            "sha256": sha256_file(sample),
            "ordered_jsonl_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        },
        "selected_counts": {
            "documents": len(documents),
            "jsonl_bytes": sample.stat().st_size,
        },
        "source_text_persisted_only_in_bounded_sample": True,
        "tokenizer_measurement_only": True,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (shard / "receipt.json").write_text(json.dumps(receipt))


def test_aggregate_replays_unique_bounded_samples(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    _shard(samples, 0, [_document(0), _document(2)])
    _shard(samples, 1, [_document(1), _document(3)])
    result = aggregate(
        samples,
        tmp_path / "aggregate.json",
        logical_shards=2,
        maximum_bytes_per_shard=1_000_000,
        scratch_root=tmp_path,
    )
    assert result["totals"]["documents"] == 4
    assert result["totals"]["domain::english::documents"] == 2
    assert result["totals"]["domain::math::documents"] == 2
    assert result["exact_document_identity_unique"] is True
    assert result["exact_text_content_unique"] is True
    assert "Verified broad" not in (tmp_path / "aggregate.json").read_text()


def test_aggregate_rejects_cross_shard_text_duplicate(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    duplicate = "The same independently identified source text. " * 30
    _shard(samples, 0, [_document(0, duplicate)])
    _shard(samples, 1, [_document(1, duplicate)])
    with pytest.raises(
        TransientTokenizerSampleAggregateError,
        match="cross-shard duplicate",
    ):
        aggregate(
            samples,
            tmp_path / "aggregate.json",
            logical_shards=2,
            maximum_bytes_per_shard=1_000_000,
            scratch_root=tmp_path,
        )


def test_aggregate_job_is_cpu_only_and_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "scripts" / "aggregate_transient_tokenizer_samples_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --cpus-per-task=1" in job
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "--logical-shards 128" in job
    assert "--maximum-bytes-per-shard 64000000" in job
    assert "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}" in job
