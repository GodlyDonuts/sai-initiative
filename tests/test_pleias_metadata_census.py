import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.pleias_metadata_census import (
    PleiasMetadataCensusError,
    aggregate_shards,
    census_local_file,
    merge_segments,
    run_segment,
    run_shard,
    select_segment,
    select_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file


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


def test_segment_partition_is_disjoint_and_reconstructs_each_shard():
    rows = [{"source_path": str(index)} for index in range(33)]
    shard = select_shard(rows, 2, 1)
    segments = [select_segment(rows, 2, 1, 4, index) for index in range(4)]
    flattened = [row["source_path"] for segment in segments for row in segment]
    assert sorted(flattened, key=int) == sorted(
        (row["source_path"] for row in shard), key=int
    )
    assert len(flattened) == len(set(flattened))


def test_segment_recovery_reconstructs_canonical_shard_bytes(tmp_path, monkeypatch):
    manifest_rows = []
    for index in range(8):
        path = tmp_path / f"parent-{index:02d}.parquet"
        pq.write_table(
            pa.table(
                {
                    "identifier": [f"id-{index}"],
                    "collection": ["Books"],
                    "open_type": ["Public Domain"],
                    "license": ["Public Domain"],
                    "language": ["English"],
                    "word_count": [100 + index],
                    "token_count": [140 + index],
                    "text": ["source text is never retained by the census"],
                }
            ),
            path,
        )
        manifest_rows.append(
            {
                "source_id": "pleias_common_corpus",
                "source_path": str(path),
                "source_repository": "PleIAs/common_corpus",
                "source_revision": "a" * 40,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "raw_source_is_training_ready": False,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    )

    def local_census(row, _token, _scratch):
        return census_local_file(Path(row["source_path"]), row)

    monkeypatch.setattr(
        "sai.data.pleias_metadata_census._download_and_census", local_census
    )
    segments_root = tmp_path / "segments"
    for segment_index in range(2):
        run_segment(
            manifest,
            segments_root / f"segment_{segment_index:05d}",
            1,
            0,
            2,
            segment_index,
            "token",
        )
    recovered_root = tmp_path / "recovered" / "shards" / "shard_00000"
    recovered = merge_segments(manifest, segments_root, recovered_root, 1, 0, 2)
    direct_root = tmp_path / "direct" / "shards" / "shard_00000"
    direct = run_shard(manifest, direct_root, 1, 0, "token")
    assert recovered["receipt_sha256"] == direct["receipt_sha256"]
    assert (recovered_root / "files.jsonl").read_bytes() == (
        direct_root / "files.jsonl"
    ).read_bytes()
    recovery = json.loads((recovered_root / "recovery.json").read_text())
    assert recovery["canonical_shard_receipt_sha256"] == recovered["receipt_sha256"]
    assert recovery["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in recovery.items() if key != "receipt_sha256"}
    )
    aggregate = aggregate_shards(
        manifest, tmp_path / "recovered" / "shards", 1, tmp_path / "aggregate.json"
    )
    assert aggregate["totals"]["files"] == 8
    assert aggregate["source_text_persisted"] is False

    segment_receipt_path = segments_root / "segment_00000" / "receipt.json"
    segment_receipt = json.loads(segment_receipt_path.read_text())
    segment_receipt["selected_paths_sha256"] = "f" * 64
    segment_receipt["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in segment_receipt.items()
            if key != "receipt_sha256"
        }
    )
    segment_receipt_path.write_text(json.dumps(segment_receipt, sort_keys=True))
    with pytest.raises(PleiasMetadataCensusError, match="segment differs"):
        merge_segments(
            manifest,
            segments_root,
            tmp_path / "tampered-recovery",
            1,
            0,
            2,
        )
