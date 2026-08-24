from pathlib import Path


def test_nemotron_bridge_runner_is_bounded_resumable_and_nonduplicating() -> None:
    script = Path(
        "scripts/run_nemotron_grounded_bridge_verification_local.sh"
    ).read_text()
    assert "sai_lanes=9" in script
    assert "--concurrency 1" in script
    assert "if [[ -f \"${summary}\" ]]" in script
    assert "while ! python3 -m sai.data.nemotron_grounded_bridge_verifier" in script
    assert "nvidia/nemotron-3-ultra-550b-a55b" in script
    assert "https://integrate.api.nvidia.com/v1" in script
    assert "--api-key-env NVIDIA_API_KEY" in script
    assert "nemotron_grounded_bridge_verification_aggregate" in script
    assert "sai.data.grounded_bridge_decontamination" in script
