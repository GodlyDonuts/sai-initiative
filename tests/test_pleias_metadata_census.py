import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.pleias_metadata_census import (
    PleiasMetadataCensusError,
    census_local_file,
    select_shard,
)
from sai.data.token_stream import sha256_file


def _manifest(path):
    return {
        "source_path": "common_corpus_1/a.parquet",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_censuses_metadata_without_persisting_text(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table(
            {
                "identifier": ["a", None],
                "collection": ["Books", "Books"],
                "open_type": ["Public Domain", "Public Domain"],
                "license": ["Public Domain", "Public Domain"],
                "language": ["English", "French"],
                "word_count": [100, 0],
                "token_count": [140, 0],
                "text": ["x" * 200, None],
            }
        ),
        path,
    )
    result = census_local_file(path, _manifest(path))
    assert result["rows"] == 2
    assert result["word_count"] == 100
    assert result["token_count"] == 140
    assert result["axes"]["collection"]["Books"]["rows"] == 2
    pair = json.dumps(["Books", "French"], separators=(",", ":"))
    assert result["axes"]["collection_language"][pair]["rows"] == 1
    assert result["structural_counts"]["text_null"] == 1
    assert result["source_text_read"] is False
    assert result["source_text_persisted"] is False
    assert result["training_ready"] is False


def test_rejects_parent_hash_drift(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"text": ["x"]}), path)
    manifest = _manifest(path)
    manifest["sha256"] = "0" * 64
    with pytest.raises(PleiasMetadataCensusError, match="identity"):
        census_local_file(path, manifest)


def test_shard_partition_is_disjoint_and_complete():
    rows = [{"source_path": str(index)} for index in range(17)]
    shards = [select_shard(rows, 4, index) for index in range(4)]
    flattened = [row["source_path"] for shard in shards for row in shard]
    assert sorted(flattened, key=int) == [str(index) for index in range(17)]
    assert len(flattened) == len(set(flattened))
