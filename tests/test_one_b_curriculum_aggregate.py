from __future__ import annotations

from sai.data.one_b_curriculum_aggregate import _shard_paths


def test_aggregate_requires_exact_162_shard_identities(tmp_path) -> None:
    paths = _shard_paths(tmp_path)
    assert len(paths) == 162
    assert paths[0] == tmp_path / "books" / "receipt.json"
    assert paths[1] == tmp_path / "pleias" / "shard_00000" / "receipt.json"
    assert paths[128] == tmp_path / "pleias" / "shard_00127" / "receipt.json"
    assert paths[129] == tmp_path / "code" / "shard_00000" / "receipt.json"
    assert paths[-1] == tmp_path / "connections" / "receipt.json"
