from pathlib import Path


def test_finemath_semantic_audit_uses_one_resumable_rate_safe_hermes_stream() -> None:
    script = Path("scripts/run_finemath_semantic_audit_local.sh").read_text()
    assert "sai.data.nous_label_worker" in script
    assert "stealth/ox-alpha" in script
    assert "http://127.0.0.1:8645/v1" in script
    assert "--logical-shards \"${sai_logical_shards}\"" in script
    assert "--judgments-per-candidate 3" in script
    assert "--concurrency 1" in script
    assert "run_segment 0 63" in script
    assert "run_segment 0 15" not in script
    assert 'segment-00-63.log" 2>&1' in script
    assert "&\nsai_" not in script
    assert "[[ -f \"${sai_summary}\" ]] && continue" in script
    assert "sk-" not in script
    assert "nvapi-" not in script


def test_finemath_semantic_finalizer_requires_all_shard_summaries() -> None:
    script = Path("scripts/finalize_finemath_semantic_audit_local.sh").read_text()
    assert "sai_summaries" in script
    assert '"${sai_summaries}" = "${sai_logical_shards}"' in script
    assert "sai.data.semantic_audit_aggregate" in script
    assert "--expected-model stealth/ox-alpha" in script
    assert "--logical-shards \"${sai_logical_shards}\"" in script
    assert "sk-" not in script
