from sai.data.pleias_quality_evidence_mirror import SAFE_FILES, mirror_evidence
from sai.data.token_stream import sha256_file


def test_mirrors_only_fixed_source_safe_allowlist(tmp_path):
    source = tmp_path / "source"
    expected = {}
    for index, (label, relative) in enumerate(SAFE_FILES):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"safe-evidence-{index}\n".encode())
        expected[label] = sha256_file(path)
    unsafe = source / "pleias-semantic-sample-20260826-r1" / "candidates.jsonl"
    unsafe.write_text("private source excerpt\n")
    output = tmp_path / "durable"
    receipt = mirror_evidence(source, output)
    assert receipt["file_count"] == len(SAFE_FILES)
    assert receipt["source_text_persisted"] is False
    assert receipt["training_ready"] is False
    assert not list(output.rglob("candidates.jsonl"))
    assert {row["label"]: row["sha256"] for row in receipt["files"]} == expected
