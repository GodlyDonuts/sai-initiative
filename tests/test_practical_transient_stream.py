import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data import practical_transient_stream as stream
from sai.data.pleias_practical_admission import SCHEMA as ADMISSION_SCHEMA
from sai.data.pleias_practical_locator_scan import LOCATOR_SCHEMA, _schema
from sai.data.token_stream import canonical_sha256, sha256_file


def test_transient_stream_reconstructs_and_verifies_text(
    tmp_path: Path, monkeypatch
) -> None:
    text = "A coherent English source document with useful context. " * 20
    content_sha256 = hashlib.sha256(text.encode()).hexdigest()
    parent_path = tmp_path / "parent.parquet"
    parent_row = {
        "identifier": "doc-1",
        "collection": "books",
        "open_type": "open",
        "license": "public domain",
        "language": "English",
        "word_count": 180,
        "token_count": 220,
        "text": text,
    }
    pq.write_table(pa.Table.from_pylist([parent_row]), parent_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest_row = {
        "source_id": "pleias_common_corpus",
        "source_path": "data/part.parquet",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "bytes": 123,
        "sha256": "1" * 64,
        "raw_source_is_training_ready": False,
    }
    manifest.write_text(json.dumps(manifest_row) + "\n")
    admission = tmp_path / "admission"
    shard = admission / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    locator = {
        "schema": LOCATOR_SCHEMA,
        "source_id": "pleias_common_corpus",
        "source_repository": manifest_row["source_repository"],
        "source_revision": manifest_row["source_revision"],
        "source_path": manifest_row["source_path"],
        "source_parent_sha256": manifest_row["sha256"],
        "source_row_index": 0,
        "source_row_identity_sha256": "2" * 64,
        "identifier": "doc-1",
        "collection": "books",
        "open_type": "open",
        "license": "public domain",
        "language": "English",
        "word_count": 180,
        "source_token_count": 220,
        "text_utf8_bytes": len(text.encode()),
        "content_sha256": content_sha256,
    }
    locators = shard / "locators.parquet"
    pq.write_table(pa.Table.from_pylist([locator], schema=_schema()), locators)
    descriptor = {
        "shard_index": 0,
        "path": str(locators.relative_to(admission)),
        "rows": 1,
        "text_utf8_bytes": len(text.encode()),
        "source_token_count": 220,
        "bytes": locators.stat().st_size,
        "sha256": sha256_file(locators),
    }
    receipt = {
        "schema": ADMISSION_SCHEMA,
        "status": "complete_practical_pleias_pretraining_admission",
        "policy": {"output_partition_policy": "canonical_source_path_sha256_modulo"},
        "outputs": {"descriptors": [descriptor]},
        "practical_pretraining_ready": True,
        "training_ready": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (admission / "receipt.json").write_text(json.dumps(receipt))

    monkeypatch.setattr(stream, "_download", lambda parent, token, scratch: parent_path)
    output = io.StringIO()
    result = stream.stream_shard(
        manifest,
        admission,
        0,
        "token",
        output,
        tmp_path / "stream-receipt.json",
        tmp_path,
    )
    emitted = json.loads(output.getvalue())
    assert emitted["text"] == text
    assert emitted["training_ready"] is True
    assert result["counts"] == {
        "source_parents": 1,
        "documents": 1,
        "text_utf8_bytes": len(text.encode()),
        "source_token_count": 220,
    }
    assert result["source_text_persisted"] is False
