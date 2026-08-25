from __future__ import annotations

from sai.data.one_b_hf_publish import PREFIX, REPOSITORY


def test_packed_release_uses_content_addressed_immutable_prefix() -> None:
    assert REPOSITORY == "Godlydonuts/Sai"
    assert PREFIX == "training/packed/one-b/20260826-r2"
