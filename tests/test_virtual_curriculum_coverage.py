import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sai.data.data_compiler_labeling import DOMAINS
from sai.data.token_stream import canonical_sha256
from sai.data.virtual_curriculum_coverage import (
    POLICY,
    POLICY_SHA256,
    SCHEMA,
    VirtualCurriculumCoverageError,
    _ppm,
    measure_coverage,
)
from sai.data.virtual_spiral_curriculum_index import (
    AGGREGATE_SCHEMA,
    AGGREGATE_STATUS,
    BANDS,
    _index_row,
    _write_shard,
)


def test_coverage_policy_requires_breadth_without_claiming_bridges() -> None:
    assert POLICY_SHA256 == canonical_sha256(POLICY)
    assert POLICY["required_domains"] == list(DOMAINS)
    assert POLICY["required_bands"] == list(BANDS)
    assert POLICY["cross_domain_labels_are_not_verified_bridges"] is True


def test_coverage_job_is_cpu_only_immutable_and_non_requeueing() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (
        root / "scripts" / "analyze_virtual_curriculum_coverage_stokes.sbatch"
    ).read_text()
    assert "#SBATCH --no-requeue" in job
    assert "#SBATCH --gres" not in job
    assert "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}" in job
    assert "--durable-output" in job
    assert "rm " not in job


def test_ppm_is_exact_and_rejects_invalid_denominators() -> None:
    assert _ppm(1, 4) == 250_000
    with pytest.raises(VirtualCurriculumCoverageError, match="denominator differs"):
        _ppm(1, 0)


def _build_index(root: Path) -> None:
    totals: Counter[str] = Counter()
    receipts = []
    ordered = hashlib.sha256()
    document_index = 0
    for component, directory, shards in (
        ("institutional_books", "books", 64),
        ("pleias_common_corpus", "pleias", 128),
    ):
        for shard_index in range(shards):
            domain = DOMAINS[document_index % len(DOMAINS)]
            second_domain = DOMAINS[(document_index + 1) % len(DOMAINS)]
            band_index = document_index % len(BANDS)
            difficulty = (band_index * 1_000) + 500
            split = "development" if shard_index == 0 else "train"
            row = _index_row(
                component=component,
                component_shard=shard_index,
                component_row_index=0,
                document_identity_sha256=f"{document_index + 1:064x}",
                content_sha256=f"{document_index + 1_000:064x}",
                output_text_utf8_bytes=10_000,
                corpus_split=split,
                source_group_sha256=f"{document_index + 2_000:064x}",
                rights_label="Public Domain",
                quality_floor_milli=8_000,
                difficulty_milli=difficulty,
                prerequisite_burden_milli=0,
                semantic_phase_hint=BANDS[band_index],
                semantic_domains=[domain, second_domain],
                concepts=[f"concept-{document_index}"],
                prerequisites=[f"prerequisite-{document_index // 2}"],
                source_custody_sha256=f"{document_index + 3_000:064x}",
            )
            receipt = _write_shard(
                root / directory / "shards" / f"shard_{shard_index:05d}",
                [row],
                component=component,
                logical_shards=shards,
                shard_index=shard_index,
                source_receipt_sha256=f"{document_index + 4_000:064x}",
            )
            receipts.append(receipt["receipt_sha256"])
            ordered.update(bytes.fromhex(row["curriculum_priority_sha256"]))
            size = row["output_text_utf8_bytes"]
            totals["documents"] += 1
            totals["output_text_utf8_bytes"] += size
            totals[f"component::{component}::documents"] += 1
            totals[f"component::{component}::output_text_utf8_bytes"] += size
            totals[f"split::{split}::documents"] += 1
            totals[f"split::{split}::output_text_utf8_bytes"] += size
            totals[f"band::{row['spiral_band']}::documents"] += 1
            totals[f"band::{row['spiral_band']}::output_text_utf8_bytes"] += size
            for semantic_domain in row["semantic_domains"]:
                totals[f"domain::{semantic_domain}::documents"] += 1
            document_index += 1
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": AGGREGATE_STATUS,
        "curriculum_policy_sha256": _index_row(
            component="institutional_books",
            component_shard=0,
            component_row_index=0,
            document_identity_sha256="1" * 64,
            content_sha256="2" * 64,
            output_text_utf8_bytes=1,
            corpus_split="train",
            source_group_sha256="3" * 64,
            rights_label="Public Domain",
            quality_floor_milli=8_000,
            difficulty_milli=0,
            prerequisite_burden_milli=0,
            semantic_phase_hint="foundation",
            semantic_domains=[DOMAINS[0]],
            concepts=[],
            prerequisites=[],
            source_custody_sha256="4" * 64,
        )["curriculum_policy_sha256"],
        "index_shards": {
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "ordered_curriculum_priority_digests_sha256": ordered.hexdigest(),
        "exact_document_identity_unique": True,
        "exact_content_identity_unique": True,
        "source_disjoint_split_preserved": True,
        "source_text_persisted": False,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    (root / "aggregate.json").write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_measurement_replays_all_shards_and_fails_only_small_volume(
    tmp_path: Path,
) -> None:
    index_root = tmp_path / "index"
    _build_index(index_root)
    output = tmp_path / "coverage.json"
    durable = tmp_path / "evidence" / "coverage.json"
    result = measure_coverage(index_root, output, durable)
    assert result["schema"] == SCHEMA
    assert result["failed_requirements"] == ["two_terabyte_window"]
    assert result["coverage_gate_passed"] is False
    assert result["cross_domain_metadata_is_verified_bridge_evidence"] is False
    assert result["training_ready"] is False
    assert output.read_bytes() == durable.read_bytes()
