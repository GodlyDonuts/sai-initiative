from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPTS = (
    "admit_pleias_practical_core_stokes.sbatch",
    "publish_pleias_practical_hf_stokes.sbatch",
    "aggregate_pleias_practical_hf_stokes.sbatch",
    "publish_practical_metadata_hf_stokes.sbatch",
    "audit_practical_corpus_stokes.sbatch",
    "verify_practical_transient_stream_stokes.sbatch",
    "reconcile_practical_bridge_foundation_stokes.sbatch",
)


def test_practical_release_graph_uses_one_verified_immutable_runtime() -> None:
    for name in SCRIPTS:
        script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert '#SBATCH --no-requeue' in script
        assert '#SBATCH --gres' not in script
        assert 'SAI_RUNTIME_ROOT:?' in script
        assert 'SAI_RUNTIME_COMMIT:?' in script
        assert 'rev-parse HEAD' in script
        assert 'status --porcelain' in script
        assert 'export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"' in script
        assert 'export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src' not in script


def test_practical_admission_binds_manifest_and_quarantine_to_runtime() -> None:
    script = (
        ROOT / "scripts/admit_pleias_practical_core_stokes.sbatch"
    ).read_text(encoding="utf-8")
    assert '--manifest "${SAI_RUNTIME_ROOT}/artifacts/' in script
    assert '--quarantine-registry-root "${SAI_RUNTIME_ROOT}/artifacts/' in script
    assert '--total-text-byte-ceiling 2000000000000' in script


def test_stream_smoke_binds_manifest_to_runtime() -> None:
    script = (
        ROOT / "scripts/verify_practical_transient_stream_stokes.sbatch"
    ).read_text(encoding="utf-8")
    assert '--manifest "${SAI_RUNTIME_ROOT}/artifacts/' in script
